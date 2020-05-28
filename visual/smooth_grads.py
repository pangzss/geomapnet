"""
Created on Wed Mar 28 10:12:13 2018

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from torch.autograd import Variable
import torch
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SmoothGradients():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, selected_layer,selected_block,param_n, param_sigma_multiplier, mode = 'IBP',to_size=(224,224)):
        self.param_n = param_n
        self.param_sigma_multiplier = param_sigma_multiplier

        if mode == 'IBP':
            from integrated_grads import IntegratedGradients
            self.Backprop = IntegratedGradients(model,selected_layer,selected_block,to_size)
        elif mode == 'GBP':
            from guidedbp_layer import GuidedBackprop
            self.Backprop = GuidedBackprop(model,selected_layer,selected_block,to_size)

    def pipe_line(self, prep_img ):
        """
            Generates smooth gradients of given Backprop type. You can use this with both vanilla
            and guided backprop
        Args:
            Backprop (class): Backprop type
            prep_img (torch Variable): preprocessed image
            target_class (int): target class of imagenet
            param_n (int): Amount of images used to smooth gradient
            param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
        """
        # Generate an empty image/matrix
        smooth_grad = np.zeros(prep_img.size()[1:]).transpose(1,2,0)

        mean = 0
        sigma = self.param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
        for x in range(self.param_n):
            # Generate noise
            noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
            # Add noise to the image
            noisy_img = prep_img + noise
            # Calculate gradients
            grads = self.Backprop.pipe_line(noisy_img)

            #stat = torch.cuda.memory_summary(device='cuda', abbreviated=True)
            #print(stat)
            # Add gradients to smooth_grad
            smooth_grad = smooth_grad + grads
    

        # Average it out
        smooth_grad = smooth_grad / self.param_n
        return smooth_grad


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision.models as models
    inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

    #model = models.resnet34(pretrained=True).cuda()
    weights_dir = '../scripts/logs/stylized_models/AachenDayNight__mapnet_stylized_4_styles_seed0.pth.tar'

    model = get_model(weights_dir).cuda()
    img_name = 'aachen1.jpg'
    img_path = os.path.join('imgs',img_name)
    img = load_image(img_path)
    # preprocess an image, return a pytorch variable
    input_img = preprocess(img)
    ori_img =  inv_normalize(input_img[0]).cpu().numpy().transpose(1,2,0)

    param_n = 50
    param_sigma_multiplier = 3
    SG = SmoothGradients(model,4,2,param_n, param_sigma_multiplier, mode = 'GBP')
    SG_grads = SG.pipe_line(input_img)
    SG_grads = np.clip(SG_grads,0,SG_grads.max())
    grads_output = norm_std(SG_grads)
    grads_output = convert_to_grayscale(grads_output)

    '''
    grads_output = np.uint8(convert_to_grayscale(grads_output)[:,:,None].repeat(3,axis=2)*255)
    #grads_output = ori_img * grads_output.repeat(3,axis=2)
    print(grads_output.shape)
    # newmask is the mask image I manually labelled
    import cv2
    # wherever it is marked white (sure foreground), change mask=1
    # wherever it is marked black (sure background), change mask=0
    up_th = np.percentile(grads_output, 95)
    low_th = np.percentile(grads_output,30)
    mask = np.zeros(grads_output.shape[:2],np.uint8)
    print(mask.shape)
    mask[grads_output[:,:,0] <= low_th] = 0
    mask[grads_output[:,:,0] >= up_th] = 1
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    # = cv2.cvtColor(grads_output[:,:,None, cv2.COLOR_BGR2GRAY)
    mask, bgdModel, fgdModel = cv2.grabCut(grads_output,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    mask = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = ori_img*mask[:,:,np.newaxis]
    '''
    img = ori_img*grads_output[:,:,None]
    file_name_to_export = 'SG_'+img_name.split('.')[0]
    to_folder = 'imgs'

    save_original_images(ori_img, to_folder, 'resized_'+img_name)
    save_original_images(img, to_folder, file_name_to_export)

