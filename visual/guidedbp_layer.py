"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import sys

from torchvision import transforms

sys.path.append('../')
import cv2
import torch
from torch.nn import ReLU
from skimage.transform import resize
import numpy as np
from utils import *
import vecquantile as vecquantile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_strongest_filters(activation_img, top=1):

    activation_img = activation_img.detach().numpy()
    # Find maximum activation for each filter for a given image
    activation_img = np.nanmax(activation_img, axis=3)
    activation_img = np.nanmax(activation_img, axis=2)

    activation_img = activation_img.sum(0)

    # Make activations 1-based indexing
    #activation_img = np.insert(activation_img, 0, 0.0)

    #  activation_image is now a vector of length equal to number of filters (plus one for one-based indexing)
    #  each entry corresponds to the maximum/summed activation of each filter for a given image

    top_filters = activation_img.argsort()[-top:]
    return list(top_filters)


class GuidedBackprop:
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, input_model, selected_layer, selected_block, to_size=None, filter_idx=None):
        self.model = input_model
        self.selected_layer = selected_layer
        self.selected_block = selected_block
        self.to_size = to_size 
        self.num_maps = 1
        self.conv_output = 0
        self.gradients = None
        self.forward_relu_outputs = []
        self.strongest_filters = None
        self.filter_idx = filter_idx
        # Put model in evaluation mode
        self.model.eval()

        #self.hook_list = []
        self.update_relus()
        self.hook_layer_backward()
        self.hook_layer_forward()

    
    def hook_layer_backward(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
            
        # Register hook to the first layer
        #first_layer = list(self.model.features._modules.items())[0][1]

        self.model._modules['conv1'].register_backward_hook(hook_function)

    def hook_layer_forward(self):
        def hook_function(module, input, output):
            self.conv_output = output[0]
        # Hook the selected layer
        self.model._modules['layer'+str(self.selected_layer)][self.selected_block]._modules['conv2'].register_forward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output 
            
            grad_relu = grad_in[0].clone()
            modified_grad_in =  torch.clamp(grad_relu, min=0.0)
           
            return (modified_grad_in,)

         # hook up the first relu
        
        self.model._modules['relu'].register_backward_hook(relu_backward_hook_function)
        # Loop through layers, hook up ReLUs
        num_blocks = [3,4,6,3]
        for layer in range(1,4+1):
            for block in range(0,num_blocks[layer-1]):
              
                self.model._modules['layer'+str(layer)][block]._modules['relu'].register_backward_hook(relu_backward_hook_function)
                if layer == self.selected_layer and block == self.selected_block:
                    
                    return 
        

    def generate_gradients(self, input_image):

        # Forward pass
        #torch.autograd.set_detect_anomaly(True)
        
        _ = self.model(input_image)
        
        '''
        conv_np = self.conv_output.cpu().data.numpy()
        quantile = 0.01
        quant = vecquantile.QuantileVector(depth=self.conv_output.shape[0], seed=1)
        distr = np.transpose(conv_np, axes=(1, 2, 0)).reshape((-1, self.conv_output.shape[0]))
        quant.add(distr)
        thresholds = quant.readout(1000)[:, int(1000 * (1-quantile)-1)]
        mask_layer = 0
        for i in range(conv_np.shape[0]):
        
            mask = resize(conv_np[i], self.to_size)
            mask = 1.0*(mask > thresholds[i])
            mask_layer += mask
        mask_layer[mask_layer>0] = 1.0
        '''
        #print(mask_layer[mask_layer>0])
        # Zero gradients
        self.model.zero_grad()
        if self.filter_idx is None:
            loss = torch.sum(torch.max(torch.max(self.conv_output,axis=2)[0],axis=1)[0])
        elif self.filter_idx == 'max':
            loss = torch.max(self.conv_output)
        elif self.filter_idx == 'min':
            loss = torch.min(self.conv_output)
        elif self.filter_idx == 'rand':
            loss = self.conv_output.view(-1)[90]

        #loss = torch.max(torch.max(torch.max(self.conv_output,axis=2)[0],axis=1)[0])
        #loss = self.conv_output[0].view(-1)[0]
        #loss = 50*torch.mean(self.conv_output)

        loss.backward(retain_graph=False)

        gradients_as_arr = resize(self.gradients.cpu().data.numpy()[0].transpose(1,2,0),self.to_size)

        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
       # gradients_as_arr = self.gradients.data.numpy()[0]

        #threshold_scale = 0.5
        #mask_all = np.zeros((224,224))
        #for feature_map in self.conv_output:
        #    feature_map = feature_map.data.numpy()
        #    feature_map = feature_map / np.max(feature_map)
       
         #   mask = cv2.resize(feature_map, (224,224))
         #   mask[mask < 0.5*np.max(feature_map)] = 0.0 # binarize the mask
         #   mask[mask > 0.5*np.max(feature_map)] = 1.0
         #   mask_all += mask
        #mask_all[mask_all>1.0] = 1.0


        return gradients_as_arr#,mask_layer

    def pipe_line(self, input_img):
        
        #device = torch.cuda.device(0)
        #model = model.cuda()
        # Guided backprop
        #GBP = GuidedBackprop(model, layer, block, to_size, filter_idx)
        # Get gradients
        input_img.requires_grad = True
        guided_grads = self.generate_gradients(input_img)

        return guided_grads


if __name__ == "__main__":
    inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

    weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'

    model = get_model(weights_dir).cuda()
    #model = models.resnet34(pretrained=True).cuda()
    img_name = 'aachen1.jpg'
    img_path = os.path.join('imgs',img_name)
    img = load_image(img_path)
    # preprocess an image, return a pytorch variable
    input_img = preprocess(img,(224,224))
    ori_img =  inv_normalize(input_img[0]).cpu().numpy().transpose(1,2,0)

    idx = 'rand'
    GBP = GuidedBackprop(model,4,2, (224,224),filter_idx=idx)
    guided_grads_4 = GBP.pipe_line(input_img.cuda())
    guided_grads_4 = np.clip(guided_grads_4,0,guided_grads_4.max())

    GBP = GuidedBackprop(model, 1, 2, (224, 224),filter_idx=idx)
    guided_grads_2 = GBP.pipe_line(input_img.cuda())

    guided_grads_2 = np.clip(guided_grads_2, 0, guided_grads_2.max())

    #grads_output = convert_to_grayscale(guided_grads.transpose(2,0,1))
    grads_output_4 = norm_std(guided_grads_4)
    grads_output_2 = norm_std(guided_grads_2)
    file_name_to_export = 'visual_'+img_name
    to_folder = 'imgs'

    #save_original_images(ori_img, to_folder, 'resized_'+img_name)
    #save_original_images(grads_output, to_folder, file_name_to_export)


    import matplotlib.pyplot as plt
    from PIL import ImageDraw
    from PIL import ImageFont


    def canvas(grads_in, s):
        output = Image.fromarray(grads_in)
        draw = ImageDraw.Draw(output)
        font = ImageFont.truetype("./open-sans/OpenSans-Regular.ttf", 20)
        draw.text((0, 0), s, (255, 255, 255), font=font)
        output = np.asarray(output)

        return output
    grads_2 = canvas(grads_output_2,'conv 1, {}'.format(idx))
    grads_4 = canvas(grads_output_4,'conv 4, {}'.format(idx))

    to_show = np.concatenate([np.uint8(np.ones_like(grads_2) * 255), grads_2, grads_4], axis=1)
    save_original_images(to_show, to_folder, 'patch_rand_' + img_name.split('.')[0])
    '''
    plt.imshow(to_show)
    plt.axis('off')
    plt.show()
    '''
