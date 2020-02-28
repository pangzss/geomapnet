import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

class compute_RF():
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def __init__(self, model, layer, block, neuron_idx):
        self.model = model
        self.selected_layer = layer
        self.selected_block = block
        self.neuron_idx = neuron_idx
        self.conv_output = None
        self.gradients = None
        self.weights_init(self.model)
        self.hook_layer_forward()
        self.hook_layer_backward()
    def deactivate_batchnorm(self,bn):
        bn.reset_parameters()
        bn.eval()
        with torch.no_grad():
            bn.weight.fill_(1.0)
            bn.bias.zero_()

    def weights_init(self,model):
        model._modules['conv1'].weight.data.fill_(1)
        self.deactivate_batchnorm(model._modules['bn1'])

        model._modules['maxpool'] = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        num_blocks = [3,4,6,3]
        for layer in range(1,4+1):
            for block in range(0,num_blocks[layer-1]):
                 model._modules['layer'+str(layer)][block]._modules['conv1'].weight.data.fill_(1)
                 model._modules['layer'+str(layer)][block]._modules['conv2'].weight.data.fill_(1)
                 self.deactivate_batchnorm(model._modules['layer'+str(layer)][block]._modules['bn1'])
                 self.deactivate_batchnorm(model._modules['layer'+str(layer)][block]._modules['bn2'])
                 try:
                     
                     model._modules['layer'+str(layer)][block]._modules['downsample'][0].weight.data.fill_(1)
                     self.deactivate_batchnorm(model._modules['layer'+str(layer)][block]._modules['downsample'][1])
                     
                 except KeyError:
                
                     pass
    def hook_layer_backward(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]

        self.model._modules['conv1'].register_backward_hook(hook_function)

    def hook_layer_forward(self):
        def hook_function(module, input, output):
            # Gets the conv output of the selected filter (from selected layer)
         
            self.conv_output = output[0, 0]
        # Hook the selected layer
        if self.selected_layer == 0:
            self.model._modules['conv1'].register_forward_hook(hook_function)
        else:
            self.model._modules['layer'+str(self.selected_layer)][self.selected_block].register_forward_hook(hook_function)

    def get_RF(self):
        img_np = np.ones((1,3,224,224))
        img_ = Variable(torch.from_numpy(img_np).float(),requires_grad=True)
        out_cnn = self.model(img_)
        
        #self.model.zero_grad()
        #grad = torch.zeros_like(self.conv_output)
        #print(self.conv_output.shape)
        #grad[self.neuron_idx] = 1
        loss = self.conv_output[self.neuron_idx]
        loss.backward()
        
        grad_np=self.gradients.data.numpy()[0]
        grad_np_sum = np.sum(grad_np,axis=0)
        print(grad_np_sum[grad_np_sum!=0].shape)
        idx_nonzeros=np.where(np.sum(grad_np,axis=0)!=0)

        center = [(np.max(idx)-np.min(idx))//2+np.min(idx) for idx in idx_nonzeros]
        RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]


        return center,RF
