import math
import torch
import torch.nn as nn
import logging
from utils import exp_util
from pathlib import Path
import importlib
# from pytorch_memlab import profile
import os
import network.sparseConvNet
import network.sparseConvNetAlter

import network.hierachicalDecoder
import network.cnp_encoder
from system.ext import groupby_sum_backward,groupby_sum
from torch.autograd import Function
class Networks():
    def __init__(self):
        self.decoder = None
        self.encoder = None
        self.conv_kernels = None
        self.rgb_decoder = None
    def eval(self):
        if self.encoder is not None:
            self.encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.conv_kernels is not None:
            self.conv_kernels.eval()
    def enc_eval(self):
        self.encoder.eval()

def load_model(training_hyper_path: str, use_epoch: int = -1,layer = None):
    """
    Load in the model and hypers used.
    :param training_hyper_path:
    :param use_epoch: if -1, will load the latest model.
    :return: Networks
    """
    training_hyper_path = Path(training_hyper_path)
    # print(training_hyper_path.parent)
    if str(training_hyper_path.parent) == "config":
        args = exp_util.parse_config_json(training_hyper_path)
        model = Networks()
        model.decoder = network.hierachicalDecoder.Model(8,29,embeding=args.encoder_specs['embed']).cuda()
        model.conv_kernels = network.sparseConvNetAlter.HierachicalSparseConv(8,29).cuda()
        model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
        if use_epoch == -1:
            print("there")
            return model, args
        state_dict = torch.load("config/encoder_500.pth.tar")["model_state"]
        model.encoder.load_state_dict(state_dict)
        state_dict = torch.load("config/model_500.pth.tar")["model_state"]
        model.decoder.base_decoder.load_state_dict(state_dict)

       

        # print("there")
        return model, args
    if training_hyper_path.name.split(".")[-1] == "json":
        args = exp_util.parse_config_json(training_hyper_path)
        exp_dir = training_hyper_path.parent
        model_paths = exp_dir.glob('model_*.pth.tar')
        model_paths = {int(str(t).split("model_")[-1].split(".pth")[0]): t for t in model_paths}
        # assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {sorted(list(model_paths.keys()))}"
        args.checkpoint = os.path.join(exp_dir,f"model_{use_epoch}.pth.tar")
    else:
        args = exp_util.parse_config_yaml(Path('configs/training_defaults.yaml'))
        args = exp_util.parse_config_yaml(training_hyper_path, args)
        logging.warning("Loaded a un-initialized model.")
        args.checkpoint = None

    model = Networks()
    model.decoder = network.hierachicalDecoder.Model(8,29).cuda()
    model.conv_kernels = network.sparseConvNetAlter.HierachicalSparseConv(8,29).cuda()
    # if args.encoder_name is not None:
    #     encoder_module = importlib.import_module("network." + args.encoder_name)
    model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
    if args.checkpoint is not None:
        if layer is not None:
            ckp = torch.load(Path(args.checkpoint).parent / f"model_{layer}_{use_epoch}.pth.tar")
        else:
            ckp = torch.load(Path(args.checkpoint).parent / f"model_{use_epoch}.pth.tar")
        if model.decoder is not None:
            if "decoder_state" in ckp.keys():
                # model.decoder.load_state_dict(state_dict)
                model.decoder.load_state_dict(ckp["decoder_state"])
            
        if model.encoder is not None:
            if "encoder_state" in ckp.keys():
                model.encoder.load_state_dict(ckp["encoder_state"])
        if model.conv_kernels is not None:
            if "sparse_state" in ckp.keys():
                model.conv_kernels.load_state_dict(ckp["sparse_state"])
        
    return model, args

def load_unet_model(training_hyper_path: str, use_epoch: int = -1,use_nerf = False,layer = 4):
    training_hyper_path = Path(training_hyper_path)
    if use_epoch == -1:
        args = exp_util.parse_config_json(training_hyper_path)
        model = Networks()
        model.decoder = network.hierachicalDecoder.Model(1,args.network_specs["latent_size"],embeding = args.embed,dims = args.network_specs["dims"],if_xyz=args.network_specs["if_xyz"]).cuda()
        model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
        if use_nerf:
            model.conv_kernels = network.sparseConvNetAlter.SparseUNet(5,args.encoder_specs["latent_size"],use_rgb=True).cuda()
            model.rgb_decoder = network.nerf.NeRF(D = 4,W=32,input_ch=args.color_latent_dim + 3,output_ch=3,skips=[2])
        else:
            model.conv_kernels = network.sparseConvNetAlter.SparseUNet(layer,args.encoder_specs["latent_size"]).cuda()
        return model, args

    # print(training_hyper_path.parent)
    if str(training_hyper_path.parent) == "config":
        args = exp_util.parse_config_json(training_hyper_path)
        model = Networks()
        model.decoder = network.hierachicalDecoder.Model(1,args.encoder_specs["latent_size"],embeding = args.embed,dims = args.network_specs["dims"]).cuda()
        model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
        if use_nerf:
            model.conv_kernels = network.sparseConvNetAlter.SparseUNet(layer,args.encoder_specs["latent_size"],use_rgb=True).cuda()
            model.rgb_decoder = network.nerf.NeRF(D = 4,W=32,input_ch=args.color_latent_dim + 3,output_ch=3,skips=[2])
        else:
            model.conv_kernels = network.sparseConvNetAlter.SparseUNet(layer,args.encoder_specs["latent_size"]).cuda()

        state_dict = torch.load("config/encoder_500.pth.tar")["model_state"]
        model.encoder.load_state_dict(state_dict)
        state_dict = torch.load("config/model_500.pth.tar")["model_state"]
        model.decoder.base_decoder.load_state_dict(state_dict)
        return model, args
    if training_hyper_path.name.split(".")[-1] == "json":
        args = exp_util.parse_config_json(training_hyper_path)
        exp_dir = training_hyper_path.parent
        model_paths = exp_dir.glob('model_*.pth.tar')
        model_paths = {int(str(t).split("model_")[-1].split(".pth")[0]): t for t in model_paths}
        # assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {sorted(list(model_paths.keys()))}"
        args.checkpoint = os.path.join(exp_dir,f"model_{use_epoch}.pth.tar")
    model = Networks()
    model.decoder = network.hierachicalDecoder.Model(1,args.network_specs["latent_size"],embeding = args.embed,dims = args.network_specs["dims"],if_xyz=args.network_specs["if_xyz"]).cuda()
    if use_nerf:
        model.conv_kernels = network.sparseConvNetAlter.SparseUNet(layer,args.encoder_specs["latent_size"],use_rgb=True).cuda()
        model.rgb_decoder = network.nerf.NeRF(D = args.nerf["D"],W=args.nerf["W"],input_ch=args.color_latent_dim + 3,output_ch=3,skips=args.nerf["skips"])
    else:
        model.conv_kernels = network.sparseConvNetAlter.SparseUNet(layer,args.encoder_specs["latent_size"]).cuda()
    model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
    if args.checkpoint is not None:
        ckp = torch.load(Path(args.checkpoint).parent / f"model_{use_epoch}.pth.tar")
        if model.decoder is not None:
            if "decoder_state" in ckp.keys():
                # model.decoder.load_state_dict(state_dict)
                model.decoder.load_state_dict(ckp["decoder_state"])
        if model.encoder is not None:
            if "encoder_state" in ckp.keys():
                model.encoder.load_state_dict(ckp["encoder_state"])
        if model.conv_kernels is not None:
            if "sparse_state" in ckp.keys():
                model.conv_kernels.load_state_dict(ckp["sparse_state"])
        if model.rgb_decoder is not None:
            if "nerf_state" in ckp.keys():
                model.rgb_decoder.load_state_dict(ckp["nerf_state"])
    return model, args


def load_origin_unet_model(training_hyper_path: str, use_epoch: int = -1,use_nerf = False,layer = 4):
    training_hyper_path = Path(training_hyper_path)
    if use_epoch == -1:
        args = exp_util.parse_config_json(training_hyper_path)
        model = Networks()
        model.decoder = network.hierachicalDecoder.Model(1,args.network_specs["latent_size"],dims = args.network_specs["dims"]).cuda()
        model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
        if use_nerf:
            model.conv_kernels = network.sparseConvNet.SparseUNet(5,args.encoder_specs["latent_size"],use_rgb=True).cuda()
            model.rgb_decoder = network.nerf.NeRF(D = 4,W=32,input_ch=args.color_latent_dim + 3,output_ch=3,skips=[2])
        else:
            model.conv_kernels = network.sparseConvNet.SparseUNet(layer,args.encoder_specs["latent_size"]).cuda()
        return model, args

    # print(training_hyper_path.parent)
    if str(training_hyper_path.parent) == "config":
        args = exp_util.parse_config_json(training_hyper_path)
        model = Networks()
        model.decoder = network.hierachicalDecoder.Model(1,args.encoder_specs["latent_size"],dims = args.network_specs["dims"]).cuda()
        model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
        if use_nerf:
            model.conv_kernels = network.sparseConvNet.SparseUNet(layer,args.encoder_specs["latent_size"],use_rgb=True).cuda()
            model.rgb_decoder = network.nerf.NeRF(D = 4,W=32,input_ch=args.color_latent_dim + 3,output_ch=3,skips=[2])
        else:
            model.conv_kernels = network.sparseConvNet.SparseUNet(layer,args.encoder_specs["latent_size"]).cuda()

        state_dict = torch.load("config/encoder_500.pth.tar")["model_state"]
        model.encoder.load_state_dict(state_dict)
        state_dict = torch.load("config/model_500.pth.tar")["model_state"]
        model.decoder.base_decoder.load_state_dict(state_dict)
        return model, args
    if training_hyper_path.name.split(".")[-1] == "json":
        args = exp_util.parse_config_json(training_hyper_path)
        exp_dir = training_hyper_path.parent
        model_paths = exp_dir.glob('model_*.pth.tar')
        model_paths = {int(str(t).split("model_")[-1].split(".pth")[0]): t for t in model_paths}
        # assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {sorted(list(model_paths.keys()))}"
        args.checkpoint = os.path.join(exp_dir,f"model_{use_epoch}.pth.tar")
    model = Networks()
    model.decoder = network.hierachicalDecoder.Model(1,args.network_specs["latent_size"],dims = args.network_specs["dims"]).cuda()
    
    model.conv_kernels = network.sparseConvNet.SparseUNet(layer,args.encoder_specs["latent_size"]).cuda()
    model.encoder = network.cnp_encoder.Model(**args.encoder_specs).cuda()
    if args.checkpoint is not None:
        ckp = torch.load(Path(args.checkpoint).parent / f"model_{use_epoch}.pth.tar")
        if model.decoder is not None:
            if "decoder_state" in ckp.keys():
                # model.decoder.load_state_dict(state_dict)
                model.decoder.load_state_dict(ckp["decoder_state"])
        if model.encoder is not None:
            if "encoder_state" in ckp.keys():
                model.encoder.load_state_dict(ckp["encoder_state"])
        if model.conv_kernels is not None:
            if "sparse_state" in ckp.keys():
                model.conv_kernels.load_state_dict(ckp["sparse_state"])
        if model.rgb_decoder is not None:
            if "nerf_state" in ckp.keys():
                model.rgb_decoder.load_state_dict(ckp["nerf_state"])
    return model, args


def load_latent_vecs(training_hyper_path: str, use_epoch: int = -1):
    """
    Load in the trained latent vectors.
    :param training_hyper_path:
    :param use_epoch: if -1, will load the latest model.
    :return:
    """
    training_hyper_path = Path(training_hyper_path)

    assert training_hyper_path.name.split(".")[-1] == "json"

    exp_dir = training_hyper_path.parent
    model_paths = exp_dir.glob('training_*.pth.tar')
    model_paths = {int(str(t).split("training_")[-1].split(".pth")[0]): t for t in model_paths}
    assert use_epoch in model_paths.keys(), f"{use_epoch} not found in {list(model_paths.keys())}"

    ckpt_path = model_paths[use_epoch]
    latent_vecs = torch.load(ckpt_path)["latent_vec"]

    if isinstance(latent_vecs, torch.Tensor):
        return latent_vecs.detach()
    else:
        return latent_vecs['weight']


# @profile
def forward_model(model: nn.Module, network_input: torch.Tensor = None,
                  latent_input: torch.Tensor = None,
                  xyz_input: torch.Tensor = None,
                  loss_func=None, max_sample: int = 2 ** 32,
                  no_detach: bool = False,
                  verbose: bool = True):
    """
    Forward the neural network model. (if loss_func is not None, will also compute the gradient w.r.t. the loss)
    Either network_input or (latent_input, xyz_input) tuple could be provided.
    :param model:           MLP model.
    :param network_input:   (N, 128)
    :param latent_input:    (N, 125)
    :param xyz_input:       (N, 3)
    :param loss_func:
    :param max_sample
    :return: [(N, X)] several values
    """
    combine=False
    if latent_input is not None and xyz_input is not None:
        combine=True
        assert network_input is None
        assert latent_input.ndimension() == 2
        assert xyz_input.ndimension() == 2
        n_chunks = math.ceil(latent_input.size(0) / max_sample)
        #network_input = torch.cat((latent_input, xyz_input), dim=1)
        latent_input = torch.chunk(latent_input, n_chunks)
        xyz_input = torch.chunk(xyz_input, n_chunks)
    else:
        assert network_input.ndimension() == 2
        n_chunks = math.ceil(network_input.size(0) / max_sample)
        latent_input = torch.chunk(network_input, n_chunks)
    # print(n_chunks)
    # print(n_chunks == 1)
    # assert not no_detach or n_chunks == 1

    # if verbose:
    #     logging.debug(f"Network input chunks = {n_chunks}, each chunk = {latent_input[0].size()},{xyz_input[0].size()}")

    head = 0
    output_chunks = None
    for chunk_i, input_latent_chunk in enumerate(latent_input):
        # (N, 1)
        if combine == True:
            input_xyz_chunk = xyz_input[chunk_i]
            input_chunk = torch.cat((input_latent_chunk, input_xyz_chunk), dim=1)
        else:
            input_chunk = input_latent_chunk

        network_output = model(input_chunk)

        if not isinstance(network_output, tuple):
            network_output = [network_output, ]

        if chunk_i == 0:
            output_chunks = [[] for _ in range(len(network_output))]
        if loss_func is not None:
            # The 'graph' in pytorch stores how the final variable is computed to its current form.
            # Under normal situations, we can delete this path right after the gradient is computed because the path
            #   will be re-constructed on next forward call.
            # However, in our case, self.latent_vec is the leaf node requesting the gradient, the specific computation:
            #   vec = self.latent_vec[inds] && cat(vec, xyz)
            #   will be forgotten, too. if we delete the entire graph.
            # Indeed, the above computation is the ONLY part that we do not re-build during the next forwarding.
            # So, we set retain_graph to True.
            # According to https://github.com/pytorch/pytorch/issues/31185, if we delete the head loss immediately
            #   after the backward(retain_graph=True), the un-referenced part graph will be deleted too,
            #   hence keeping only the needed part (a sub-graph). Perfect :)
            loss_func(network_output,
                      torch.arange(head, head + network_output[0].size(0), device=network_output[0].device)
                      ).backward(retain_graph=(chunk_i != n_chunks - 1))
        if not no_detach:
            network_output = [t.detach() for t in network_output]

        for payload_i, payload in enumerate(network_output):
            output_chunks[payload_i].append(payload)
        head += network_output[0].size(0)
    output_chunks = [torch.cat(t, dim=0) for t in output_chunks]
    return output_chunks


def get_samples(r: int, device: torch.device, a: float = 0.0, b: float = None):
    """
    Get samples within a cube, the voxel size is (b-a)/(r-1). range is from [a, b]
    :param r: num samples
    :param a: bound min
    :param b: bound max
    :return: (r*r*r, 3)
    """
    overall_index = torch.arange(0, r ** 3, 1, device=device, dtype=torch.long)
    r = int(r)

    if b is None:
        b = 1. - 1. / r

    vsize = (b - a) / (r - 1)
    samples = torch.zeros(r ** 3, 3, device=device, dtype=torch.float32)
    samples[:, 0] = (overall_index // (r * r)) * vsize + a
    samples[:, 1] = ((overall_index // r) % r) * vsize + a
    samples[:, 2] = (overall_index % r) * vsize + a
    return samples




def pack_samples(sample_indexer: torch.Tensor, count: int,
                 sample_values: torch.Tensor = None):
    """
    Pack a set of samples into batches. Each element in the batch is a random subsampling of the sample_values
    :param sample_indexer: (N, )
    :param count: C
    :param sample_values: (N, L), if None, will return packed_inds instead of packed.
    :return: packed (B, C, L) or packed_inds (B, C), mapping: (B, ).
    """
    from system.ext import pack_batch

    # First shuffle the samples to avoid biased samples.
    shuffle_inds = torch.randperm(sample_indexer.size(0), device=sample_indexer.device)
    sample_indexer = sample_indexer[shuffle_inds]

    mapping, pinds, pcount = torch.unique(sample_indexer, return_inverse=True, return_counts=True)

    n_batch = mapping.size(0)
    packed_inds = pack_batch(pinds, n_batch, count * 2)         # (B, 2C)

    pcount.clamp_(max=count * 2 - 1)
    packed_inds_ind = torch.floor(torch.rand((n_batch, count), device=pcount.device) * pcount.unsqueeze(-1)).long()  # (B, C)

    packed_inds = torch.gather(packed_inds, 1, packed_inds_ind)     # (B, C)
    packed_inds = shuffle_inds[packed_inds]                         # (B, C)

    if sample_values is not None:
        assert sample_values.size(0) == sample_indexer.size(0)
        packed = torch.index_select(sample_values, 0, packed_inds.view(-1)).view(n_batch, count, sample_values.size(-1))
        return packed, mapping
    else:
        return packed_inds, mapping


def groupby_reduce(sample_indexer: torch.Tensor, sample_values: torch.Tensor, op: str = "max"):
    """
    Group-By and Reduce sample_values according to their indices, the reduction operation is defined in `op`.
    :param sample_indexer: (N,). An index, must start from 0 and go to the (max-1), can be obtained using torch.unique.
    :param sample_values: (N, L)
    :param op: have to be in 'max', 'mean'
    :return: reduced values: (C, L)
    """
    C = sample_indexer.max() + 1
    n_samples = sample_indexer.size(0)

    assert n_samples == sample_values.size(0), "Indexer and Values must agree on sample count!"

    if op == 'mean':
        from system.ext import groupby_sum
        values_sum, values_count = groupby_sum(sample_values, sample_indexer, C)
        return values_sum / values_count.unsqueeze(-1)
    elif op == 'sum':
        from system.ext import groupby_sum
        values_sum = GroupSumFunction.apply(sample_values, sample_indexer, C)
        # values_sum, _ = groupby_sum(sample_values, sample_indexer, C)
        return values_sum
    elif op == 'max':
        from system.ext import groupby_max
        return groupby_max(sample_values, sample_indexer, C)
    else:
        raise NotImplementedError

    

class GroupSumFunction(Function):
    @staticmethod
    def forward(ctx,sample_values,sample_indexer,C):
        ctx.save_for_backward(sample_values,sample_indexer)
        values_sum, _ = groupby_sum(sample_values, sample_indexer, C)
        return values_sum

    @staticmethod
    def backward(ctx, grad_output):
        sample_values,sample_indexer = ctx.saved_tensors
        grad_input = torch.zeros_like(sample_values)
        grad_input = groupby_sum_backward(grad_output,grad_input,sample_indexer)
        return grad_input,None,None



class StepLearningRateSchedule():
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return self.initial * (self.factor ** (epoch // self.interval))



def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

if __name__ == '__main__':
    # sv = torch.tensor([[0.2, 0.7], [0.3, 0.5], [-0.1, 0.5], [0.0, 7.0], [1.5, 6.1], [3.5, 4.1]]).float().cuda()
    # si = torch.tensor([12, 15, 15, 12, 12, 12]).long().cuda()
    # p, m = pack_samples(sv, si, 2)
    # print("p =", p)
    # print("m =", m)
    # si = torch.tensor([0, 0, 1, 2, 0, 1]).long().cuda()

    sv = torch.rand((100000, 29)).float().cuda()
    si = torch.randint(0, 128, (100000, )).long().cuda()

    from pycg import exp
    timer = exp.Timer()

    for k in range(20):
        print(groupby_reduce(si, sv, "max"))
        timer.toc("max")
        print(groupby_reduce(si, sv, "mean"))
        timer.toc("mean")
    timer.report()