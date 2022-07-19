import torch.nn as nn
import torch
import torch.nn.functional as F
from network.deepsdf_decoder_cnp import Model as sdf_decoder

class Model(nn.Module):
    def __init__(self,layer_num,latent_dim,dims = [64,64,128,64]):
        super().__init__()
        dropout = None
        dropout_prob = 0.2
        norm_layers = [0,1,2,3,4,5]
        latent_in = [3]
        input_ch = 3
        self.base_decoder = sdf_decoder(position_size=input_ch,latent_size = latent_dim,dims = dims,dropout = dropout,
                dropout_prob = dropout_prob,latent_in = latent_in,norm_layers = norm_layers,weight_norm = True)
    def forward(self,input):
        x,_ = self.base_decoder(input)
        return x
    