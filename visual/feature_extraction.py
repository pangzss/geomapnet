import torch
from conv_resnet import resnet34
from deconv_resnet import Deconv_ResNet

#from utils import *
class conv_forward():
    def __init__(self):
        self.activations = {}
        self.identity_maps = {}

        #self.deconv_model=deconv_model

    def get_activation(self,layer,blk_index):
        def hook(module,input,output):
            if 'maxpool' in layer:
                self.activations[layer]['actvation']= output[0]
                self.activations[layer]['pool_indices']= output[1]
                self.identity_maps[layer]['activation'] = output[0]
                self.identity_maps[layer]['pool_indices'] = output[1]
            else:
                self.activations[layer]['block'+str(blk_index)]= output
                self.identity_maps[layer]['block'+str(blk_index)]= module.identity
        return hook
    # registering hooks for the conv resnet
    def store_maps_hook(self,layers):
        for layer in layers:
            if 'layer' in layer :
                self.activations[layer] = {}
                self.identity_maps[layer] = {}
                #blk_hooks[layer] = {}
                for blk_index,block in enumerate(layers[layer]):
                    # blk_hooks[layer]['block'+str(blk_index)] = 
                    block.register_forward_hook(self.get_activation(layer,blk_index))
            elif 'maxpool' in layer:
                self.activations[layer] = {}
                self.identity_maps[layer] = {}
                #blk_hooks[layer] = {}
                #blk_hooks[layer] = 
                layers[layer].register_forward_hook(self.get_activation(layer,0))


    #def print_shape(self,layer,blk_index):
    #    def hook(module,input,output):
    #        print(layer+': block'+str(blk_index)+': ',output.shape)
    #        if 'conv' not in layer:
    #            print(module.identity.shape)
    #    return hook
    # registering hook for deconv model for printing
    #def print_shape_hook(self,layers):
    #    if self.deconv_model != None:
    #        layers = self.deconv_model._modules
    #        for layer in deconv_layers:
    #            if 'layer' in layer :
    #                for blk_index,block in enumerate(deconv_layers[layer]):
    #                    block.register_forward_hook(print_shape(layer,blk_index))
    #            elif 'conv' in layer:
    #                deconv_layers[layer].register_forward_hook(print_shape(layer,blk_index))

    #if __name__ == '__main__':
    def get_conv_maps(self,conv_model,img):
        '''this function only takes resnet34 module'''
        layers = conv_model._modules
    
        self.store_maps_hook(layers)

        out = conv_model(img)

        #if deconv_model:
        #    deconv_model = Deconv_ResNet(identity_maps) 
        #    deconv_layers = deconv_model._modules
        #    print_shape_hook(deconv_layers)
        #    out_deconv = deconv_model(out)
        return out, self.activations, self.identity_maps 

