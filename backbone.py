import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    # 残差BottleNeck结构的类,没写可变形卷积
    expansion = 4  # 不知道为啥->每一个残差块，输出的通道是输入的4倍

    def __init__(self, inplanes, planes, downsample=None, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 use_dcn=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, \
                               dilation=dilation, padding=dilation, bias=False, )
        self.bn2 = norm_layer(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4)

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # if not self.downsample==None:
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 参数前面加上*号 ，意味着参数的个数不止一个，另外带一个星号（*）参数的函数传入的参数存储为一个元组（tuple），
# 带两个（*）号则是表示字典（dict）

class ResNetBackbone(nn.Module):
    """ Adapted from torchvision.models.resnet """

    def __init__(self, layers, dcn_layers=[0, 0, 0, 0], dcn_interval=1, \
                 atrous_layers=[], block=Bottleneck, norm_layer=nn.BatchNorm2d):
        super().__init__()
        # These will be populated by _make_layer
        self.num_base_layers = len(layers)
        self.layers = nn.ModuleList()
        self.channels = []
        self.norm_layer = norm_layer
        self.dilation = 1
        self.atrous_layers = atrous_layers

        # From torchvision.models.resnet.Resnet
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # in_channles=3->R,G,B
        self.bn = norm_layer
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 函数前面加_::
        # #单下划线：_add()	是一种私有函数的命名约定，即提示程序员该函数只能在类或者该文件内部使用，但实际上也可以在外部使用。
        # 双下划线__add()	私有函数，只能在内部使用
        self._make_layer(block, 64, layers[0], dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 128, layers[1], stride=2, dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 256, layers[1], stride=2, dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)
        self._make_layer(block, 512, layers[1], stride=2, dcn_layers=dcn_layers[0], dcn_interval=dcn_interval)

        # This contains every module that should be initialized by loading in pretrained weights.
        # Any extra layers added onto this that won't be initialized by init_backbone will not be
        # in this list. That way, Yolact::init_weights knows which backbone weights to initialize
        # with xavier, and which ones to leave alone.
        # 如果m是二维卷积层的话，就要加入backbone_modules进行初始化权重
        self.backbone_modules = [m for m in self.modules() if isinstance(m, nn.Conv2d)]

    def _make_layer(self, block, planes, blocks, stride=1, dcn_layers=0, dcn_interval=1):
        """ Here one layer means a string of n Bottleneck blocks. """
        downsample = None

        # This is actually just to create the connection between layers, and not necessarily to
        # downsample. Even if the second condition is met, it only downsamples when stride != 1
        if stride != 1 or self.inplanes != planes * block.expansion:  # block就是前面定义的残差块，expansion=4
            if len(self.layers) in self.atrous_layers:
                self.dilation += 1
                stride = 1

            # 下采样的方式主要有两种：
            # 1、采用stride为2的池化层，如Max-pooling和Average-pooling，目前通常使用Max-pooling，
            # 因为他计算简单而且能够更好的保留纹理特征；
            # 2、采用stride为2的卷积层，下采样的过程是一个信息损失的过程，而池化层是不可学习的，
            # 用stride为2的可学习卷积层来代替pooling可以得到更好的效果，当然同时也增加了一定的计算量。
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False,
                          dilation=self.dilation),
                self.norm_layer(planes * block.expansion)
            )

        layers = []
        use_dcn = (dcn_layers >= blocks)  # 判断如果dcn的层大于等于残差块层，则使用可变形卷积
        # 加入第一个残差块
        layers.append(block(self.inplanes, planes, downsample=downsample, dilation=self.dilation, stride=stride,
                            norm_layer=self.norm_layer,
                            use_dcn=use_dcn))
        # 更新输入,一开始是64,第一层卷积时，inplanes=planes
        self.inplanes = planes * block.expansion
        # block=[3,4,23,3]
        '''在config中的resnet101_backbone.args中体现'''
        for i in range(1, blocks):
            use_dcn = ((i + dcn_layers) >= blocks) and (i % dcn_interval == 0)
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer, use_dcn=use_dcn))
            # 默认:downsample=None, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,use_dcn=False
        layer = nn.Sequential(*layers)

        self.channels.append(planes * block.expansion)
        self.layers.append(layer)  # nn.ModuleList()

        return layer

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)

        return tuple(outs)

    def init_backbone(self, path):
        """ Initializes the backbone weights for training. """
        '''这部分之后看'''
        state_dict = torch.load(path)

        # Replace layer1 -> layers.0 etc.
        keys = list(state_dict)
        for key in keys:
            if key.startswith('layer'):
                idx = int(key[5])
                new_key = 'layers.' + str(idx - 1) + key[6:]
                state_dict[new_key] = state_dict.pop(key)

        # Note: Using strict=False is berry scary. Triple check this.
        self.load_state_dict(state_dict, strict=False)

    def add_layer(self,conv_channels=1024, downsample=2, depth=1, block=Bottleneck):
        """ Add a downsample layer to the backbone as per what SSD does. """
        self._make_layer(block, conv_channels // block.expansion, blocks=depth, stride=downsample)


def construct_backbone(cfg):
    """ Constructs a backbone given a backbone config object (see config.py).
    在给定主干配置对象（请参阅config.py）的情况下构造主干"""
    backbone = cfg.type(*cfg.args)
    ###包裹参数传递的实现是在定义函数时在形参前面加上*或**，
    # *所对应的形参会被解释为一个元组（tuple），而**所对应的形参会被解释为一个字典

    # 'type': ResNetBackbone,  cfg.type()表示使用残差骨干网络
    # 'args': ([3, 4, 23, 3],),  *cfg.args传递[3, 4, 23, 3]元组，即使用101层残差网络

    # Add downsampling layers until we reach the number we need
    num_layers = max(cfg.selected_layers) + 1 #'selected_layers': list(range(1, 4))->[1, 2, 3]->num_layers=4,

    while len(backbone.layers) < num_layers:
        backbone.add_layer()

    return backbone
