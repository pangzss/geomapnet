import torch
import torch.nn as nn

def deconv3x3(in_planes,out_planes,stride=1):
    """ 3x3 deconvolution """
    resize_conv = nn.Sequential(
         nn.Upsample(scale_factor = stride, mode='bilinear',align_corners=True),
         nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
        kernel_size=3, stride=1, padding=0,bias=False)
    )
    return resize_conv

class Block(nn.Module):

    def __init__(self, in_planes, out_planes, identity, stride=1):
        super(Block, self).__init__()

        self.identity = identity
        self.relu1 = nn.ReLU(inplace=True)
        self.deconv1 = deconv3x3(in_planes,in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.relu2 = nn.ReLU(inplace=True)
        # may need to upsample
        self.deconv2 = deconv3x3(in_planes,out_planes,stride = stride)
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

class Resize_ResNet(nn.Module):
    def __init__(self, identity_maps):
        super(Resize_ResNet, self).__init__()
        
        self.identity_maps = identity_maps
        block = Block
        self.num_blocks_layers = [3, 6, 4, 3]

        self.layer1 = self._make_layer(block, self.num_blocks_layers[0],512, 256,1, stride = 2)
        self.layer2 = self._make_layer(block, self.num_blocks_layers[1],256, 128,2,  stride = 2)
        self.layer3 = self._make_layer(block, self.num_blocks_layers[2],128, 64,3, stride = 2)
        self.layer4 = self._make_layer(block, self.num_blocks_layers[3],64, 64,4)
        self.layers = [self.layer1,self.layer2,self.layer3,self.layer4]
        self.unpool = nn.MaxUnpool2d(2,stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_last = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode='bilinear',align_corners=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 3,
                kernel_size=7, stride=1, padding=0,bias=False)
        )
       
        
        #self.conv2deconv_layer_indices = {1:4, 2:3, 3:2, 4,1}
        #self.
    def _make_layer(self, block, num_blocks, planes, out_planes, layer_idx, stride=1):
        downsample = None

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

        layers.append(block(planes,out_planes,identity_layer['block0'],stride=stride))
  
        return nn.Sequential(*layers)

    def _forward_impl(self, x,conv_layer_idx,conv_blk_idx):
        #conv_layer_idx : the layer index the chosen activation map is in 
        #conv_blk_idx : the blk index the chosen activation map is in
        deconv_layer_idx = 4 - conv_layer_idx + 1
        num_blks_curr_layer = self.num_blocks_layers[deconv_layer_idx-1]
        deconv_blk_idx =  num_blks_curr_layer - conv_blk_idx -1
        for curr_layer in range(deconv_layer_idx,4+1):
            layer = self.layers[curr_layer-1]
            for curr_blk in range(deconv_blk_idx,num_blks_curr_layer):
                blk = layer[curr_blk]
            
                x = blk(x)
                
            deconv_blk_idx = 0
            if curr_layer < 4:
                num_blks_curr_layer = self.num_blocks_layers[curr_layer]

        
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)

        pool_idx = self.identity_maps['maxpool']['pool_indices']
        x = self.unpool(x,pool_idx)
        x = self.relu(x)
        x = self.conv_last(x)

        return x

    def forward(self, x,conv_layer_idx,conv_blk_idx):
        return self._forward_impl(x,conv_layer_idx,conv_blk_idx)

