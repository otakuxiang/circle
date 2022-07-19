# Layer class warpped around pytorch layer with customized name and combinations
import torch
import torch.nn as nn

# There is no global switch to allow for better flexibility.
# In order to use other normalization instead of batch-norm
# You should check all Seq layers and SharedMLP to input the following classes as the norm_layer param.


class _INBase(nn.Sequential):
    def __init__(self, in_size, instance_norm=None, name=""):
        super(_INBase, self).__init__()
        self.add_module(name + "in", instance_norm(in_size))
        # Instance norm do not have learnable affine parameters.
        # They only have running metrics, which do need initialization


class InstanceNorm1d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm1d, self).__init__(in_size, instance_norm=nn.InstanceNorm1d, name=name)


class InstanceNorm2d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm2d, self).__init__(in_size, instance_norm=nn.InstanceNorm2d, name=name)


class InstanceNorm3d(_INBase):
    def __init__(self, in_size, name=""):
        super(InstanceNorm3d, self).__init__(in_size, instance_norm=nn.InstanceNorm3d, name=name)


class GroupNorm(nn.Sequential):
    def __init__(self, in_size, num_groups, name=""):
        super(GroupNorm, self).__init__()
        self.add_module(name + "gn", nn.GroupNorm(num_groups, in_size))
        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0.0)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super(_BNBase, self).__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm1d, self).__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm2d, self).__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):
    def __init__(self, in_size, name=""):
        super(BatchNorm3d, self).__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


def get_norm_layer(layer_def, dimension, **kwargs):
    # Layer def is given by user.
    # kwargs are other necessary parameters needed.
    if layer_def is None:
        return nn.Identity()
    class_name = layer_def["class"]
    kwargs.update(layer_def)
    del kwargs["class"]
    return {
        "InstanceNorm": [InstanceNorm1d, InstanceNorm2d, InstanceNorm3d][dimension - 1],
        "GroupNorm": GroupNorm,
        "BatchNorm": [BatchNorm1d, BatchNorm2d, BatchNorm3d][dimension - 1]
    }[class_name](**kwargs)


class _ConvBase(nn.Sequential):

    def __init__(self, in_size, out_size, kernel_size, stride, padding, dilation,
                 activation, bn, bn_dim, init, conv=None,
                 bias=True, preact=False, name=""):
        super(_ConvBase, self).__init__()

        bias = bias and (bn is None)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn is not None:
            if not preact:
                bn_unit = get_norm_layer(bn, bn_dim, in_size=out_size)
            else:
                bn_unit = get_norm_layer(bn, bn_dim, in_size=in_size)

        if preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=1, stride=1, padding=0, dilation=1,
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv1d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 1,
            init, conv=nn.Conv1d,
            bias=bias, preact=preact, name=name)


class Conv2d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1),
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv2d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 2,
            init, conv=nn.Conv2d,
            bias=bias, preact=preact, name=name)


class Conv3d(_ConvBase):
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
                 activation=nn.ReLU(inplace=True), bn=None, init=nn.init.kaiming_normal_,
                 bias=True, preact=False, name=""):
        super(Conv3d, self).__init__(
            in_size, out_size, kernel_size, stride, padding, dilation,
            activation, bn, 3,
            init, conv=nn.Conv3d,
            bias=bias, preact=preact, name=name)


class FC(nn.Sequential):

    def __init__(self,
                 in_size,
                 out_size,
                 activation=nn.ReLU(inplace=True),
                 bn=None,
                 init=None,
                 preact=False,
                 name=""):
        super(FC, self).__init__()

        fc = nn.Linear(in_size, out_size, bias=bn is None)
        if init is not None:
            init(fc.weight)
        if bn is None:
            nn.init.constant_(fc.bias, 0)

        if bn is not None:
            if not preact:
                bn_unit = get_norm_layer(bn, 1, in_size=out_size)
            else:
                bn_unit = get_norm_layer(bn, 1, in_size=in_size)

        if preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn is not None:
                self.add_module(name + 'normlayer', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class SharedMLP(nn.Sequential):
    def __init__(self, args,
                 bn=None, activation=nn.ReLU(inplace=True), last_act=True, name=""):
        super(SharedMLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv1d(
                    args[i],
                    args[i + 1],
                    bn=bn if (last_act or (i != len(args) - 2)) else None,
                    activation=activation if (last_act or (i != len(args) - 2)) else None
                ))


class MLP(nn.Sequential):
    def __init__(self, args,
                 bn=None, activation=nn.ReLU(inplace=True), last_act=True, name=""):
        super(MLP, self).__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                FC(
                    args[i],
                    args[i + 1],
                    bn=bn if (last_act or (i != len(args) - 2)) else None,
                    activation=activation if (last_act or (i != len(args) - 2)) else None
                ))


class PointDecoderFCDeconv(nn.Module):
    """
    Decode a point set from 1D feature using two branches. (i.e. Fully-Connected + De-convolution)
    Ref: Fan et al. 2017 PSG-Net, Yi et al. 2019 GSPN
    """
    def __init__(self, bn, in_feat=1024, n_points=512, out_feat=3):
        super().__init__()

        # DeConv Formula: Nout = (Nin - 1) * stride - 2 * padding + dilation * (kernel - 1) + 1 + output_padding
        #                      = (Nin - 1) * stride + kernel
        # https://pytorch.org/docs/stable/nn.html#torch.nn.ConvTranspose1d

        if 384 < n_points <= 896:
            self.upconv = nn.Sequential(
                nn.ConvTranspose2d(in_feat, 512, kernel_size=3, stride=1),   # 512 x 3 x 3
                get_norm_layer(bn, 2, in_size=512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2),      # 256 x 7 x 7
                get_norm_layer(bn, 2, in_size=256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),      # 128 x 16 x 16
                get_norm_layer(bn, 2, in_size=128), nn.ReLU(),
                nn.ConvTranspose2d(128, out_feat, kernel_size=1, stride=1),        # 3 x 16 x 16
            )
            num_point_conv = 256
        elif 896 < n_points <= 1536:
            self.upconv = nn.Sequential(
                nn.ConvTranspose2d(in_feat, 512, kernel_size=2, stride=1),   # 512 x 2 x 2
                get_norm_layer(bn, 2, in_size=512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1),      # 256 x 3 x 3
                get_norm_layer(bn, 2, in_size=256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2),      # 256 x 7 x 7
                get_norm_layer(bn, 2, in_size=256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=3),      # 128 x 22 x 22
                get_norm_layer(bn, 2, in_size=128), nn.ReLU(),
                nn.ConvTranspose2d(128, out_feat, kernel_size=1, stride=1),        # 3 x 22 x 22
            )
            num_point_conv = 484
        elif 1536 < n_points <= 3072:
            self.upconv = nn.Sequential(
                nn.ConvTranspose2d(in_feat, 512, kernel_size=2, stride=1),   # 512 x 2 x 2
                get_norm_layer(bn, 2, in_size=512), nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1),      # 256 x 4 x 4
                get_norm_layer(bn, 2, in_size=256), nn.ReLU(),
                nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2),      # 256 x 10 x 10
                get_norm_layer(bn, 2, in_size=256), nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=5, stride=3),      # 128 x 32 x 32
                get_norm_layer(bn, 2, in_size=128), nn.ReLU(),
                nn.ConvTranspose2d(128, out_feat, kernel_size=1, stride=1),        # 128 x 32 x 32
            )
            num_point_conv = 1024
        else:
            raise NotImplementedError

        num_point_fc = n_points - num_point_conv

        self.fc = Seq(in_feat).fc(512, bn=bn).fc(512, bn=bn).fc(
            num_point_fc * out_feat, activation=None)
        self.num_output = out_feat

    def forward(self, x):
        """
        Input: [B x F]
        Output: [B x 3 x N]
        """
        num_batches = x.size(0)
        points_upconv = self.upconv(x.unsqueeze(2).unsqueeze(2))    # B x 3 x 32 x 32
        points_upconv = points_upconv.view(num_batches, self.num_output, -1)
        points_fc = self.fc(x)
        points_fc = points_fc.view(num_batches, self.num_output, -1)
        x = torch.cat((points_upconv, points_fc), dim=2)
        return x


class Seq(nn.Sequential):

    def __init__(self, input_channels):
        super(Seq, self).__init__()
        self.count = 0
        self.current_channels = input_channels

    def conv1d(self,
               out_size,
               kernel_size=1,
               stride=1,
               padding=0,
               dilation=1,
               activation=nn.ReLU(inplace=True),
               bn=None,
               init=nn.init.kaiming_normal_,
               bias=True,
               preact=False,
               name=""):

        self.add_module(
            str(self.count),
            Conv1d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv2d(self,
               out_size,
               kernel_size=(1, 1),
               stride=(1, 1),
               padding=(0, 0),
               dilation=(1, 1),
               activation=nn.ReLU(inplace=True),
               bn=None,
               init=nn.init.kaiming_normal_,
               bias=True,
               preact=False,
               name=""):

        self.add_module(
            str(self.count),
            Conv2d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def conv3d(self,
               out_size,
               kernel_size=(1, 1, 1),
               stride=(1, 1, 1),
               padding=(0, 0, 0),
               dilation=(1, 1, 1),
               activation=nn.ReLU(inplace=True),
               bn=None,
               init=nn.init.kaiming_normal_,
               bias=True,
               preact=False,
               name=""):
        self.add_module(
            str(self.count),
            Conv3d(
                self.current_channels,
                out_size,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                activation=activation,
                bn=bn,
                init=init,
                bias=bias,
                preact=preact,
                name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def fc(self,
           out_size,
           activation=nn.ReLU(inplace=True),
           bn=None,
           init=None,
           preact=False,
           name=""):

        self.add_module(
            str(self.count),
            FC(self.current_channels,
               out_size,
               activation=activation,
               bn=bn,
               init=init,
               preact=preact,
               name=name))
        self.count += 1
        self.current_channels = out_size

        return self

    def dropout(self, p=0.5):
        # type: (Seq, float) -> Seq

        self.add_module(str(self.count), nn.Dropout(p=0.5))
        self.count += 1

        return self

    def maxpool2d(self,
                  kernel_size,
                  stride=None,
                  padding=0,
                  dilation=1,
                  return_indices=False,
                  ceil_mode=False):
        self.add_module(
            str(self.count),
            nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                return_indices=return_indices,
                ceil_mode=ceil_mode))
        self.count += 1

        return self