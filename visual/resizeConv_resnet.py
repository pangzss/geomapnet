import torch
import torch.nn as nn

def deconv3x3(in_planes,out_planes,stride=1):
    """ 3x3 deconvolution """
    resize_conv = nn.Sequential(
         nn.Upsample(scale_factor = stride, mode='nearest'),
         nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
        kernel_size=3, stride=1, padding=0,bias=False)
    )
    return resize_conv
def up1x1(in_planes,out_planes,stride=2):
    """ 3x3 deconvolution """
    resize_conv = nn.Sequential(
         nn.Upsample(scale_factor = stride, mode='nearest'),
         nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes,
        kernel_size=1, stride=1, padding=0,bias=False)
    )
class Block(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, upsample=None):
        super(Block, self).__init__()

        self.relu1 = nn.ReLU(inplace=True)
        self.deconv1 = deconv3x3(in_planes,in_planes)
    
        self.relu2 = nn.ReLU(inplace=True)
        # may need to upsample
        self.deconv2 = deconv3x3(in_planes,out_planes,stride = stride)
        # for downsampling the act map from the previous conv block (later deconv block)
        self.upsample = upsample
    def forward(self,x): # id_map
      
        out = self.relu1(x)
        out = self.deconv1(out)
    

        out = self.relu2(out)
        out = self.deconv2(out)
   
        
        if self.upsample is not None:
            out += self.upsample(x)

        return out

class resizeConv_resnet(nn.Module):
    def __init__(self):
        super(resizeConv_resnet, self).__init__()
        
        self.name = 'resizeConv'

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
                nn.Upsample(scale_factor = 2, mode='nearest'),
                nn.ReflectionPad2d(1),
                nn.Conv2d(64, 3,
                kernel_size=7, stride=1, padding=0,bias=False)
        )

    def _make_layer(self, block, num_blocks, planes, out_planes, layer_idx, stride=1):
    
        upsample = None

        if stride !=1 :
            upsample = up1x1(planes,out_planes,stride=stride)
            
        layers = []
        for i in range(num_blocks-1):
            layers.append(block(planes,planes))

        layers.append(block(planes,out_planes,stride=stride, upsample=upsample))
  
        return nn.Sequential(*layers)

    def _forward_impl(self, x,conv_layer_idx,conv_blk_idx,pool_idces):
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

       # pool_idx = self.identity_maps['maxpool']['pool_indices']
        
        x = self.unpool(x,pool_idces)
        x = self.relu(x)
        x = self.conv_last(x)

        return x

    def forward(self, x,conv_layer_idx,conv_blk_idx,pool_idces):
        return self._forward_impl(x,conv_layer_idx,conv_blk_idx,pool_idces)

