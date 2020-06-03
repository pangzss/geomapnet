import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.p1 = torch.ones(1, planes, 1, 1).cuda()
        self.p2 = torch.ones(1, planes, 1, 1).cuda()
        self.p1.requires_grad = True
        self.p2.requires_grad = True
        #self.p1 = torch.ones(1,1,planes,1,1).cuda()
        #self.p2 = torch.ones(1,1,planes,1,1).cuda()
        #self.p1.requires_grad = True
        #self.p2.requires_grad = True

    def calc_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def StyleIntegrationConv1(self,x):
        
        self.p1.data.clamp_(0.,1.,)

        size_x = x.size()

        means,stds = self.calc_mean_std(x, eps=1e-5)                     # (4xN)xCX1X1

        means = means.reshape(-1,4,size_x[1],1,1)                        # Nx4xCx1x1
        stds = stds.reshape(-1,4,size_x[1],1,1)                          # Nx4xCx1x1
        c_means = means[:,:3]                                       # Nx3xCx1x1
        c_stds = stds[:, :3]                                        # Nx3xCx1x1
        size_stats = c_means.size()

        s_means = means[:,3][:,None].expand(size_stats)         # Nx1XCx1x1 --expand--> Nx3xCx1x1
        s_stds = stds[:,3][:,None].expand(size_stats)            # Nx1XCx1x1 --expand--> Nx3xCx1x1

        x = x.reshape(-1, 4, size_x[-3], size_x[-2], size_x[-1]) # Nx4xCxHxW
        c = x[:, :3]                              # Nx3xCxHxW
        s = x[:, 3][:,None]                       # Nx1xCxHxW
        size_feats = c.size()
        weighted_means = self.p1.expand(size_stats)*c_means + (torch.ones_like(c_means)- self.p1.expand(size_stats))*s_means    # Nx3xCx1x1
        weighted_stds = self.p1.expand(size_stats)*c_stds + (torch.ones_like(c_stds)- self.p1.expand(size_stats))*s_stds       # Nx3xCx1x1

        normalized_feats = (c - c_means.expand(size_feats)) / c_stds.expand(size_feats) # Nx3xCxHxW
        integrated_feats = normalized_feats * weighted_stds.expand(size_feats) + weighted_means.expand(size_feats) # Nx3xCxHxW
        integrated_feats = torch.cat([integrated_feats,s],dim=1).reshape(size_x)

        return integrated_feats

    def StyleIntegrationConv2(self, x):
        self.p2.data.clamp_(0.,1.)

        size_x = x.size()

        means, stds = self.calc_mean_std(x,
                                         eps=1e-5)  # (4xN)xCX1X1                                                        

        means = means.reshape(-1, 4, size_x[1], 1,
                              1)  # Nx4xCx1x1                                                          
        stds = stds.reshape(-1, 4, size_x[1], 1,
                            1)  # Nx4xCx1x1                                                          
        c_means = means[:, :3]  # Nx3xCx1x1                                                               
        c_stds = stds[:, :3]  # Nx3xCx1x1                                                               
        size_stats = c_means.size()

        s_means = means[:, 3][:, None].expand(
            size_stats)  # Nx1XCx1x1 --expand--> Nx3xCx1x1                                             
        s_stds = stds[:, 3][:, None].expand(
            size_stats)  # Nx1XCx1x1 --expand--> Nx3xCx1x1                                            

        x = x.reshape(-1, 4, size_x[-3], size_x[-2],
                      size_x[-1])  # Nx4xCxHxW                                                                  
        c = x[:, :3]  # Nx3xCxHxW                                                                                 
        s = x[:, 3][:,
            None]  # Nx1xCxHxW                                                                                 
        size_feats = c.size()
        weighted_means = self.p2.expand(size_stats) * c_means + (
                    torch.ones_like(c_means) - self.p2.expand(size_stats)) * s_means  # Nx3xCx1x1   
        weighted_stds = self.p2.expand(size_stats) * c_stds + (
                    torch.ones_like(c_stds) - self.p2.expand(size_stats)) * s_stds  # Nx3xCx1x1    

        normalized_feats = (c - c_means.expand(size_feats)) / c_stds.expand(
            size_feats)  # Nx3xCxHxW                                           
        integrated_feats = normalized_feats * weighted_stds.expand(size_feats) + weighted_means.expand(
            size_feats)  # Nx3xCxHxW
        integrated_feats = torch.cat([integrated_feats, s], dim=1).reshape(size_x)

        return integrated_feats

    def BIN1(self, x, eps=1e-5):

        self.p1.data.clamp_(0., 1., )

        size_x = x.size()

        # batch norm
        batch_mean = torch.mean(x, dim=(0, 2, 3))
        batch_std = torch.sqrt(torch.mean((x - batch_mean.reshape(1, size_x[1], 1, 1)) ** 2, dim=(0, 2, 3)))
        x_hat_batch = (x - batch_mean.reshape((1, size_x[1], 1, 1))) / (batch_std.reshape(1, size_x[1], 1, 1) + eps)

        # instance norm
        instance_mean,instance_std = self.calc_mean_std(x)
        x_hat_instance = (x - instance_mean.expand(size_x)) / (instance_std.expand(size_x) + eps)

        # BIN
        y = self.p1*x_hat_batch + (torch.ones_like(self.p1)-self.p1)*x_hat_instance
        #y = self.bn1.weight.reshape(1,size_x[1],1,1) * y + self.bn1.bias.reshape(1,size_x[1],1,1)
        return self.bn1.weight.reshape(1,size_x[1],1,1) * y + self.bn1.bias.reshape(1,size_x[1],1,1)

    def BIN2(self, x, eps=1e-5):

        self.p2.data.clamp_(0., 1., )

        size_x = x.size()

        # batch norm
        batch_mean = torch.mean(x, dim=(0, 2, 3))
        batch_std = torch.sqrt(torch.mean((x - batch_mean.reshape(1, size_x[1], 1, 1)) ** 2, dim=(0, 2, 3)))
        x_hat_batch = (x - batch_mean.reshape((1, size_x[1], 1, 1))) / (batch_std.reshape(1, size_x[1], 1, 1) + eps)

        # instance norm
        instance_mean, instance_std = self.calc_mean_std(x)
        x_hat_instance = (x - instance_mean.expand(size_x)) / (instance_std.expand(size_x) + eps)

        # BIN
        y = self.p2 * x_hat_batch + (torch.ones_like(self.p2) - self.p2) * x_hat_instance
        #y = self.bn2.weight.reshape(1, size_x[1], 1, 1) * y + self.bn2.bias.reshape(1, size_x[1], 1, 1)
        return self.bn2.weight.reshape(1, size_x[1], 1, 1) * y + self.bn2.bias.reshape(1, size_x[1], 1, 1)

    def forward(self, x):

        s = x.size()
        identity = x

        out = self.conv1(x)
        if s[0] != 1:
            out = self.BIN1(out)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if s[0] != 1:
            out = self.BIN2(out)
        else:
            out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #s = x.size()
        #if s[0] != 1:
        #    x = x.reshape(-1,4,s[-3],s[-2],s[-1])
        #    x = x[:,:3].reshape(-1,s[-3],s[-2],s[-1])


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


if __name__ == '__main__':
    model = resnet34(pretrained=False)
    num_blocks = [3,4,6,3]
    for layer in range(1,5):
        for block in range(num_blocks[layer-1]):
            print(model._modules['layer'+str(layer)][block].p2.shape)