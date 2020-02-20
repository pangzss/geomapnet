import torch


#from utils import *
class conv_forward():
    def __init__(self):
        self.activations = {}

    def get_activation(self,layer,blk_index):
        def hook(module,input,output):
            if 'maxpool' in layer:
                self.activations[layer]['actvation']= output[0]
                self.activations[layer]['pool_indices']= output[1]

            else:
                self.activations[layer]['block'+str(blk_index)]= output
        return hook
    # registering hooks for the conv resnet
    def store_maps_hook(self,layers):
        for layer in layers:
            if 'layer' in layer :
                self.activations[layer] = {}
        
                for blk_index,block in enumerate(layers[layer]):
                    block.register_forward_hook(self.get_activation(layer,blk_index))
                    
            elif 'maxpool' in layer:
                self.activations[layer] = {}
                layers[layer].register_forward_hook(self.get_activation(layer,0))

    def get_conv_maps(self,conv_model,img):
        '''this function only takes resnet34 module'''
        layers = conv_model._modules
    
        self.store_maps_hook(layers)

        out = conv_model(img)

 
        return out, self.activations

