import sparseconvnet as scn 
import torch
from torch import nn
import time
import pdb,traceback

def sparse_add(x,skip):

    spatial_size = skip.spatial_size.tolist()
    cord_x = x.get_spatial_locations().cuda()[:,0:3]
    cord_s = skip.get_spatial_locations().cuda()[:,0:3]

    id_x = cord_x[:,0] * spatial_size[1] * spatial_size[2] + cord_x[:,1] * spatial_size[2] + cord_x[:,2]
    id_s = cord_s[:,0] * spatial_size[1] * spatial_size[2] + cord_s[:,1] * spatial_size[2] + cord_s[:,2]
    _,inverse_1,count_1 = torch.unique(torch.cat([id_x,id_s]),return_inverse=True,return_counts=True)
    inds_1 = torch.arange(0,id_x.size(0) + id_s.size(0),device=cord_x.device,dtype= torch.long)
    mask_1 = count_1[inverse_1[inds_1]] > 1
    mask_x = mask_1[:id_x.size(0)]
    mask_s = mask_1[id_x.size(0):]
    inds_x = inds_1[:id_x.size(0)][mask_x]
    inds_s = inds_1[id_x.size(0):][mask_s] - id_x.size(0)
    _,inv_x = torch.sort(id_x[inds_x])
    _,inv_s = torch.sort(id_s[inds_s])
    inds_x = inds_x[inv_x]
    inds_s = inds_s[inv_s]
    x.features[inds_x,:] = x.features[inds_x,:] + skip.features[inds_s,:]

    return x

class InstanceNormReLU(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(latent_dim,affine = True,eps = 1e-04, momentum=0.99,track_running_stats=False)
        self.relu = nn.LeakyReLU(0.05)

    def forward(self,x):
        # print(x.features)
        features = x.features
        features = self.norm(features)
        features = self.relu(features)
        x.features = features
        return x


        
class SparseUNet(nn.Module):
    def __init__(self,layer_num,latent_dim,use_rgb = False):
        super().__init__()
        self.use_rgb = use_rgb
        encoders = []
        predictors = []

        for i in range(layer_num):
            if i == 0:
                num_in = latent_dim
                num_out = latent_dim
            else:
                num_in = latent_dim * 2 ** (i - 1)
                num_out = latent_dim * 2 ** i 
            encoders.append(UNetEncoderBlock(num_in,num_out,True if i!= 0 else False))
            
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

    def subdivide(self,cords,x):
        x = x.unsqueeze(1).repeat(1,8,1)
        son = cords.unsqueeze(1).repeat(1,8,1) * 2 + self.offset
        return son.view(-1,3),x.reshape(-1,x.shape[-1])

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
        features = x.features
        for i in range(len(self.decoders)): 
            skip = sparse_feature[i]

            x = self.decoders[i](features,skip,cords,x.spatial_size)       
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
            features = x.features[divide_mask,:]
        x = self.corner_inputer([cords + 1,x.features[divide_mask,:].contiguous()])
        x = self.corner_maker(x)
        c_cords = x.get_spatial_locations().cuda()
        # print(location_time)

        geo_latent = self.corner_linear(x.features)
        return geo_latent,logits,xyz,cords,c_cords


        


    
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
        self.deconv_input = None
        self.input_layer = None
        self.upsample = scn.Convolution(3,num_in,num_out,2,1,False)

    def prepare_deconv(self,x,cords,spatial_size):
        sn = spatial_size * 2 + 1
        self.input_size = sn
        if self.input_layer is None:
            self.input_layer = scn.InputLayer(3,self.input_size)
        else:
            input_size = sn.tolist()
            if self.input_size[0] != input_size[0] or self.input_size[1] != input_size[1] \
                or self.input_size[2] != input_size[2]:
                self.input_size = torch.LongTensor(input_size)
                self.input_layer = scn.InputLayer(3,self.input_size)
        cords = cords * 2 + 1
        return self.input_layer([cords,x])

    def forward(self,x,skip,cords,spatial_size):

        x = self.prepare_deconv(x,cords,spatial_size)
        x = self.upsample(x)
        x = sparse_add(x,skip)

        
        x = self.conv_block(x)
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
