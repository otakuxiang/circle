import sparseconvnet as scn 
import torch
from torch import nn
import time
import pdb


class InstanceNormReLU(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(latent_dim,affine = True,eps = 1e-04, momentum=0.99,track_running_stats=False)
        self.relu = nn.LeakyReLU(0.05)

    def forward(self,x):
        features = x.features
        features = self.norm(features)
        features = self.relu(features)
        x.features = features
        return x


class SparseUNet(nn.Module):
    def __init__(self,layer_num,latent_dim):
        super().__init__()
        self.use_rgb = use_rgb
        encoders = []
        predictors = []
        dense_convertors = []
        for i in range(layer_num):
            if i == 0:
                num_in = latent_dim
                num_out = latent_dim
            else:
                num_in = latent_dim * 2 ** (i - 1)
                num_out = latent_dim * 2 ** i 
            encoders.append(UNetEncoderBlock(num_in,num_out,True if i!= 0 else False))
            dense_convertors.insert(0,scn.SparseToDense(3,num_out))
        self.dense_convertors = torch.nn.ModuleList(dense_convertors)
        self.encoders = torch.nn.ModuleList(encoders)
        decoders = []
        for i in range(layer_num - 1):

            num_out = latent_dim * 2 ** i
            num_in = latent_dim * 2 ** (i + 1) 
            predictors.insert(0,predictLayer(2,num_out,32))
            decoders.insert(0,UNetDecoderBlock(num_in,num_out))
        
        # TODO: make it more complex
        self.corner_maker = scn.Sequential().add(
            scn.Convolution(3,latent_dim,latent_dim * 2,2,1,False)).add(
                InstanceNormReLU(latent_dim* 2)
            )
        
        self.corner_linear = nn.Linear(latent_dim * 2,latent_dim,bias = True)
        self.decoders = torch.nn.ModuleList(decoders)
        self.predictors = torch.nn.ModuleList(predictors)
        self.upsampler = nn.Upsample(scale_factor=2)
        self.offset = torch.tensor([
            [0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]
        ],dtype = torch.int32).cuda()
        self.structure = None

    def get_params_without_predictor(self):
        for name,params in self.named_parameters():
            if name.find("predictors") == -1:
                yield params

    def subdivide(self,cords):
        son = cords.unsqueeze(1).repeat(1,8,1) * 2 + self.offset
        return son.view(-1,3)

    def set_input_dim(self,input_dim):
        self.input_layer = scn.InputLayer(3,torch.LongTensor(input_dim))
        self.corner_inputer = scn.InputLayer(3,torch.LongTensor([input_dim[0]+2,input_dim[1]+2,input_dim[2]+2]))
               
    def forward(self,input):
        x = self.input_layer(input)
        sparse_feature = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i != len(self.encoders) - 1:
                sparse_feature.insert(0,x)
        logits = []
        cords = x.get_spatial_locations().cuda()
        cords = cords[:,0:3]
        xyz = []
        l = len(self.decoders)
        for i in range(len(self.decoders)): 
            x = self.dense_convertors[i](x)
            print(x.shape,x.nelement() * 4 / 1024 /1024)
            cords = self.subdivide(cords)
            skip = self.dense_convertors[i + 1](sparse_feature[i])

            x = self.decoders[i](x,skip,cords)       

            mask = x.get_spatial_locations().cuda()
            cords = mask[:,0:3]
            xyz.insert(0,mask)
            logit_mask = self.predictors[i](x)
            logits.insert(0,logit_mask)
            if self.structure is not None:
                gt = self.structure[l - i - 1]
                divide_mask = (gt[:,cords[:,0],cords[:,1],cords[:,2]] == 1).view(-1)
            else:
                divide_mask = torch.argmax(logit_mask,dim = 1).squeeze(0) == 1

            cords = cords[divide_mask,:]


        x = self.corner_inputer([cords + 1,x.features[divide_mask,:].contiguous()])

        x = self.corner_maker(x)
        c_cords = x.get_spatial_locations().cuda()
       
        geo_latent = self.corner_linear(x.features)
        return geo_latent,logits,xyz,cords,c_cords


        



        
class SparseDoubleConv(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        self.conv_block1 = scn.Sequential().add(
            scn.SubmanifoldConvolution(3,num_in,num_out,3,False,1)
        ).add(
            InstanceNormReLU(num_out)
        )
        self.conv_block2 = scn.Sequential().add(
            scn.SubmanifoldConvolution(3,num_out,num_out,3,False,1)
        ).add(
            InstanceNormReLU(num_out)
        )
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x
    
class SparseResBlock(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        self.conv_block1 = scn.Sequential().add(
            scn.SubmanifoldConvolution(3,num_in,num_out,3,False,1)
        ).add(
            InstanceNormReLU(num_out)
        )
        self.conv_block2 = scn.Sequential().add(
            scn.SubmanifoldConvolution(3,num_out,num_out,3,False,1)
        ).add(
            InstanceNormReLU(num_out)
        )
        self.conv_block3 = scn.Sequential().add(
            scn.SubmanifoldConvolution(3,num_out,num_out,3,False,1)
        )
        self.norm = torch.nn.BatchNorm1d(num_out,affine = True,eps = 1e-04, momentum=0.99,track_running_stats=False)
        self.add = scn.AddTable()
        self.non_linear = scn.LeakyReLU()

    def forward(self,x):
        x = self.conv_block1(x)
        res = x
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x.features = self.norm(x.features)
        x = self.non_linear(self.add([x,res]))
        return x
    
        


class UNetEncoderBlock(nn.Module):
    def __init__(self,num_in,num_out,is_pooling):
        super().__init__()
        self.is_pooling = is_pooling
        self.res_block = SparseResBlock(num_in,num_out)
        self.downsample = scn.Convolution(3,num_in,num_in,2,2,True)
    
    def forward(self,x):
        if self.is_pooling:
            x = self.downsample(x)
        x = self.res_block(x)
        
        return x

class UNetDecoderBlock(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        
        self.conv_block = SparseResBlock(num_out,num_out)
        self.input_layer = None
        self.upsample = nn.ConvTranspose3d(num_in,num_out,2,2,bias = False)
        
    def forward(self,x,skip,cords):
        x = self.upsample(x)
        x = x + skip
        if self.input_layer is None:
            self.input_size = torch.LongTensor([x.size(2),x.size(3),x.size(4)])
            self.input_layer = scn.InputLayer(3,torch.LongTensor([x.size(2),x.size(3),x.size(4)]))
        else:
            if self.input_size[0] != x.size(2) or self.input_size[1] != x.size(3) \
                or self.input_size[2] != x.size(4):
                self.input_size = torch.LongTensor([x.size(2),x.size(3),x.size(4)])
                self.input_layer = scn.InputLayer(3,torch.LongTensor([x.size(2),x.size(3),x.size(4)]))
        features = x[0,:,cords[:,0],cords[:,1],cords[:,2]].squeeze(0).permute(1,0)
        sparse_x = self.input_layer([cords,features])
        x = self.conv_block(sparse_x)
        return x

class predictLayer(nn.Module):
    def __init__(self,num_output,num_input,hidden):
        super().__init__()
        self.block = nn.Sequential(
                nn.Conv1d(num_input,hidden,bias = False,kernel_size = 1),
                nn.InstanceNorm1d(hidden),
                nn.LeakyReLU(),
            nn.Conv1d(hidden,num_output,bias = True,kernel_size = 1)
        )
    def forward(self,x):
        features = x.features.permute(1,0).unsqueeze(0).contiguous()
        logit = self.block(features)
        return logit
