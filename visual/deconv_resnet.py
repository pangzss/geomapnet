import torch
import torch.nn as nn

def deconv3x3(in_planes,out_planes,stride=1, output_padding=0):
    """ 3x3 deconvolution """
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                            padding=1, output_padding=output_padding, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convoluton for later downsampling"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                                             bias=False)

class Block(nn.Module):

    def __init__(self, in_planes, out_planes, identity, stride=1,output_padding=0):
        super(Block, self).__init__()

        self.identity = identity
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv1 = deconv3x3(in_planes,in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.relu2 = nn.ReLU(inplace=True)
        # may need to upsample
        self.deconv2 = deconv3x3(in_planes,out_planes,stride = stride, output_padding = output_padding)
        self.bn2 = nn.BatchNorm2d(out_planes)
        # for downsampling the act map from the previous conv block (later deconv block)
        #self.downsample = downsample
    def forward(self,x): # id_map
        out = x - self.identity
        out = self.relu1(x)
        out = self.deconv1(x)
        out = self.bn1(x)

        out = self.relu2(x)
        out = self.deconv2(x)
        out = self.bn2(out)
        
        #if self.downsample is not None:
        #    self.activ_blk_next = self.downsample(activ_blk_next)

        return out

class Deconv_ResNet(nn.Module):
    def __init__(self, identity_maps):
        super(Deconv_ResNet, self).__init__()
        
        self.identity_maps = identity_maps
        block = Block
        num_blocks_layers = [3, 6, 4, 3]

        self.layer1 = self._make_layer(block, num_blocks_layers[0],512, 256,1, stride = 2)
        self.layer2 = self._make_layer(block, num_blocks_layers[1],256, 128,2,  stride = 2)
        self.layer3 = self._make_layer(block, num_blocks_layers[2],128, 64,3, stride = 2)
        self.layer4 = self._make_layer(block, num_blocks_layers[3],64, 64,4)
        self.unpool = nn.MaxUnpool2d(2,stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_last = nn.ConvTranspose2d(64,3,kernel_size=7,stride=2,
                                            padding=1, output_padding=1)
        
    def _make_layer(self, block, num_blocks, planes, out_planes, layer_idx, stride=1):
        downsample = None
        output_padding = 0

        identity_layer = self.identity_maps['layer'+str(4-layer_idx+1)]
        #if layer_idx != 4:
        #    id_layer_curr = self.identity_maps['layer'+str(4-layer_idx+1)] 
        #    id_layer_next = self.identity_maps['layer'+str(4-layer_idx)]
        #else:
        #    id_layer_curr = self.identity_maps['layer'+str(4-layer_idx+1)]
        #    id_layer_next = self.identity_maps['maxpool']['activation']
     

        if stride !=1 :
            output_padding = 1
            
        layers = []
        for i in range(num_blocks-1):
            id_for_blk_i =  identity_layer['block'+str(num_blocks-1-i)]
            layers.append(block(planes,planes,id_for_blk_i))
        
        # the second to last needs downsampling
        #activ_blk_next = activ_layer_curr['block'+str(num_blocks-2+1)]
        #layers.append(block(planes,planes,activ_blk_next,downsample=downsample))
        # the last needs upsampling (via stride)

        layers.append(block(planes,out_planes,identity_layer['block0'],stride=stride,output_padding = output_padding))
  
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        pool_idx = self.identity_maps['maxpool']['pool_indices']
        x = self.unpool(x,pool_idx)
        x = self.relu(x)
        x = self.conv_last(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

