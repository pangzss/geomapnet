import torch
from conv_resnet import resnet34
from deconv_resnet import Deconv_ResNet
from utils import *
conv_model = resnet34()
layers = conv_model._modules
activations = {}
identity_maps = {}

def get_activation(layer,blk_index):
    def hook(module,input,output):
        if 'maxpool' in layer:
            activations[layer]['actvation']= output[0]
            activations[layer]['pool_indices']= output[1]
            identity_maps[layer]['activation'] = output[0]
            identity_maps[layer]['pool_indices'] = output[1]
        else:
            activations[layer]['block'+str(blk_index)]= output
            identity_maps[layer]['block'+str(blk_index)]= module.identity
    return hook
# registering hooks for the conv resnet
for layer in layers:
    if 'layer' in layer :
        activations[layer] = {}
        identity_maps[layer] = {}
        #blk_hooks[layer] = {}
        for blk_index,block in enumerate(layers[layer]):
            # blk_hooks[layer]['block'+str(blk_index)] = 
            block.register_forward_hook(get_activation(layer,blk_index))
    elif 'maxpool' in layer:
        activations[layer] = {}
        identity_maps[layer] = {}
        #blk_hooks[layer] = {}
        #blk_hooks[layer] = 
        layers[layer].register_forward_hook(get_activation(layer,0))

img = torch.ones((1,3,255,255))
out = conv_model(img)
deconv_model = Deconv_ResNet(identity_maps) 
deconv_layers = deconv_model._modules

def print_shape(layer,blk_index):
    def hook(module,input,output):
        print(layer+': block'+str(blk_index)+': ',output.shape)
        if 'conv' not in layer:
            print(module.identity.shape)
    return hook
# registering hook for deconv model for printing
for layer in deconv_layers:
    if 'layer' in layer :
        for blk_index,block in enumerate(deconv_layers[layer]):
            block.register_forward_hook(print_shape(layer,blk_index))
    elif 'conv' in layer:
        deconv_layers[layer].register_forward_hook(print_shape(layer,blk_index))


