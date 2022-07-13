import sparseconvnet as scn 
import torch
from torch import nn
from torch.autograd import grad

class InstanceNormLeakyReLU(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(latent_dim,affine = True,eps = 1e-04, momentum=0.99,track_running_stats=False)
        self.relu = nn.LeakyReLU(0.333)

    def forward(self,x):
        # print(x.features)
        
        features = x.features
        features = self.norm(features)
        features = self.relu(features)
        x.features = features
        
        return x


class InstanceNormReLU(nn.Module):
    def __init__(self,latent_dim):
        super().__init__()
        self.norm = torch.nn.InstanceNorm1d(latent_dim,affine = True,eps = 1e-04, momentum=0.99)
        self.relu = nn.LeakyReLU(0.333)

    def forward(self,x):
        # print(x.features)
        features = x.features.permute(1,0).unsqueeze(0)
        features = self.norm(features)
        features = self.relu(features)
        x.features = features.squeeze(0).permute(1,0)
        return x

class SparseUNet2D(nn.Module):
    def __init__(self,layer_num,latent_dim,input_dim):
        super().__init__()
        self.input_layer = scn.InputLayer(2,torch.LongTensor([input_dim,input_dim]))
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
            dense_convertors.insert(0,scn.SparseToDense(2,num_out))
        self.dense_convertors = torch.nn.ModuleList(dense_convertors)
        self.encoders = torch.nn.ModuleList(encoders)
        decoders = []
        for i in range(layer_num - 1):
            num_out = latent_dim * 2 ** i
            num_in = latent_dim * 2 ** (i + 1) 
            predictors.insert(0,predictLayer(2,num_out,32))
            decoders.insert(0,UNetDecoderBlock(num_in,num_out))
        self.decoders = torch.nn.ModuleList(decoders)
        self.predictors = torch.nn.ModuleList(predictors)
        self.upsampler = nn.Upsample(scale_factor=2)
        self.offset = torch.tensor([
            [0,0],[0,1],[1,0],[1,1]
        ],dtype = torch.int32).cuda()

    def subdivide(self,cords):
        son = cords.unsqueeze(1).repeat(1,4,1) * 2 + self.offset
        return son.view(-1,2)

    def forward(self,input):
        x = self.input_layer(input)
        sparse_features = []
        encoder_results = []
        decoder_results = []
        sparse_feature = []
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i != len(self.encoders) - 1:
                sparse_features.insert(0,x)
            sparse_feature.append(x)   
        logits = []
        cords = x.get_spatial_locations().cuda()
        cords = cords[:,0:2]
        x = self.dense_convertors[0](x)
        encoder_results.insert(0,x)
        label_result = []
        for i in range(len(self.decoders)): 
            x = self.dense_convertors[i](x)
            cords = self.subdivide(cords)
            skip = self.dense_convertors[i + 1](sparse_features[i])
            encoder_results.insert(0,skip)
            if i == 0:
                label = torch.ones_like(skip)
            # skip = skip * label

            x = self.decoders[i](x,skip,cords)
            decoder_results.insert(0,x)
            features = x.features.permute(1,0).unsqueeze(0)
            mask = x.get_spatial_locations()
            
            
            
            logit = torch.zeros([x.size(0),2,x.size(2),x.size(3)]).cuda()
            # print(features.shape)
            logit_mask = self.predictors[i](features)
            # print(logit_mask)
            logit[:,0,:,:] = 1.0
            logit[:,:,mask[:,0],mask[:,1]] = logit_mask

            logits.insert(0,logit)
            label = torch.argmax(logit,dim = 1,keepdim = True).float()
            # print(label.shape)
            # label = label.permute(1,0).view(1,-1,x.size(2),x.size(3))
            # print(label.shape)
            label_result.insert(0,label)
            # x = x * label  # to make zero node no longger subdivide 
            
            label = self.upsampler(label)
        return x,logits,sparse_feature,decoder_results

class SparseResBlock(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        self.conv_block1 = scn.Sequential().add(
            scn.SubmanifoldConvolution(2,num_in,num_out,3,False,1)
        ).add(
            InstanceNormLeakyReLU(num_out)
        )
        self.conv_block2 = scn.Sequential().add(
            scn.SubmanifoldConvolution(2,num_out,num_out,3,False,1)
        ).add(
            InstanceNormLeakyReLU(num_out)
        )
        self.conv_block3 = scn.Sequential().add(
            scn.SubmanifoldConvolution(2,num_out,num_out,3,False,1)
        )
        self.norm = torch.nn.InstanceNorm1d(num_out)
        self.add = scn.AddTable()
        self.non_linear = scn.LeakyReLU(0.01)

    def forward(self,x):
        x = self.conv_block1(x)
        res = x
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        features = x.features.view(1,x.features.size(0),x.features.size(1))
        features = features.permute(0,2,1)
        features = self.norm(features)
        x.features = features.view(x.features.size(1),x.features.size(0)).permute(1,0)
        x = self.non_linear(self.add([x,res]))
        return x
    
        


class UNetEncoderBlock(nn.Module):
    def __init__(self,num_in,num_out,is_pooling):
        super().__init__()
        self.is_pooling = is_pooling
        self.res_block = SparseResBlock(num_in,num_out)
        # self.double_conv = scn.Sequential().add(
        #     scn.SubmanifoldConvolution(3,latent_dim,latent_dim*2,f_size,False,1)
        # ).add(scn.InstanceNormReLU(latent_dim*2)
        # ).add(scn.SubmanifoldConvolution(3,latent_dim*2,latent_dim*2,3,False,1)
        # ).add(scn.InstanceNormReLU(latent_dim*2))
        # self.downsample = scn.Sequential(
        # ).add(scn.Convolution(3,latent_dim * 2,latent_dim * 2,2,2,False)
        # ).add(scn.BatchNormLeakyReLU(latent_dim * 2))
        self.downsample = scn.MaxPooling(2,2,2)
        
    def forward(self,x):
        if self.is_pooling:
            x = self.downsample(x)
        x = self.res_block(x)
        
        return x

class UNetDecoderBlock(nn.Module):
    def __init__(self,num_in,num_out):
        super().__init__()
        # self.conv_block1 = nn.Sequential(
        #     nn.Conv2d(num_out, num_out, 3 , bias = False, padding=1),
        #     nn.InstanceNorm2d(num_out),
        #     nn.LeakyReLU()        
        # )
        # self.conv_block2 = nn.Sequential(
        #     nn.Conv2d(num_out, num_out, 3 , bias = False, padding=1),
        #     nn.InstanceNorm2d(num_out),
        #     nn.LeakyReLU()        
        # )
        # self.conv_block3 = nn.Sequential(
        #     nn.Conv2d(num_out, num_out, 3 , bias = False, padding=1),
        #     nn.InstanceNorm2d(num_out)
        # )
        self.conv_block = SparseResBlock(num_out,num_out)
        self.input_layer = None
        # self.non_linear = nn.LeakyReLU()
        # self.upsample = scn.Deconvolution(3,latent_dim * 2, latent_dim,
        #                     2, 2, False)
        self.upsample = nn.ConvTranspose2d(num_in,num_out,2,2,bias = False)


    def forward(self,x,skip,cords):
        x = self.upsample(x)
        # print(x)
        x = x + skip
        if self.input_layer is None:
            self.input_layer = scn.InputLayer(2,torch.LongTensor([x.size(2),x.size(3)]))
        features = x[0,:,cords[:,0],cords[:,1]].squeeze(0).permute(1,0)
        sparse_x = self.input_layer([cords,features])
        x = self.conv_block(sparse_x)
        # x = self.conv_block1(x)
        # res = x
        # x = self.conv_block2(x)
        # x = self.conv_block3(x)
        # x += res
        # x = self.non_linear(x)
        return x

class predictLayer(nn.Module):
    def __init__(self,num_output,num_input,hidden):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(num_input,hidden,bias = False,kernel_size = 1),
            nn.InstanceNorm1d(hidden),
            nn.LeakyReLU(),
            nn.Conv1d(hidden,2,bias = False,kernel_size = 1)
        )
    def forward(self,x):
        logit = self.block(x)
        return logit
