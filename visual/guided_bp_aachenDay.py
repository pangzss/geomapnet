import torch
from guided_bp_sanity import *
import pickle
import os
import os.path as osp

def pipe_line(model, img_path, layer, block, to_folder, filter_idx = None, top_idx = None, style=None,task='classification'):
  
    img_file = img_path
    # load an image
    img = load_image(img_file)
    # preprocess an image, return a pytorch variable
    input_img = preprocess(img)
    input_img.requires_grad = True
  
    # Guided backprop
    GBP = GuidedBackprop(model, layer, block,filter_idx = filter_idx)
    # Get gradients
    guided_grads = GBP.generate_gradients(input_img)
    # Save colored gradients

    #save_gradient_images(guided_grads, file_name_to_export, pretrained)
    # Convert to grayscale
    #grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    #save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
    # Positive and negative saliency maps
    #pos_sal, _ = get_positive_negative_saliency(guided_grads)

    file_name_to_export = 'layer_'+str(layer)+'_block_'+str(block)+'_filterNo.'+str(filter_idx)+'_top'+str(top_idx)
    
    save_gradient_images_style(guided_grads, to_folder, file_name_to_export,style,task)
    save_original_images_style(img, to_folder, file_name_to_export+'ori',style,task)
    #save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
    #plt.imshow(grayscale_guided_grads[0],cmap='gray')
    #plt.imshow(pos_sal.transpose(1,2,0))
    #plt.show(block=False)
    #plt.show()
    print('Guided backprop completed. Layer {}, block {}, filter No.{}, top {}'.format(layer, block, filter_idx,top_idx))

if __name__ == '__main__':
    filterMaxima = torch.load('./figs/AachenDay_files/filterMaxima.pt')
    
    # choose top 

    task = 'localization/guidedbp'
    img_paths = \
    img_path = \
    style = 'Four_styles'
    model = get_model(task,pretrained=True)
    num_blocks = [3,4,6,3]
    to_folder = 'AachenDay_files'
    for layer in range(1,4+1):
        for block in range(0,num_blocks[layer-1]):
            # In AachenDay, there are 349 sample images. Every one of them
            # has a strongest filter in every block. Every strongest filter has its
            # own index among all the filters owned by the block.  Here the code aims to
            # pick out top k of those filters and corresponding sample images.
            topK = 8

            maxima_values = filterMaxima['layer'+str(layer)]['block'+str(block)][0]
            maxima_filter_idces = filterMaxima['layer'+str(layer)]['block'+str(block)][1]

            maxima_img_h2l = np.argsort(maxima_values)[::-1][:topK]
            maxima_img_filter_idces = maxima_filter_idces[maxima_img_h2l]

            with open('./figs/AachenDay_files/img_dirs.txt', 'rb') as fb:
                img_dirs = pickle.load(fb)

            imgs_selected = []
            for idx in maxima_img_h2l:
                imgs_selected.append(img_dirs[idx]) 
            
        
            for i,img_dir in enumerate(imgs_selected):

                dataset = 'AachenDay'
                path = osp.join('data', dataset)
                img_path = osp.join(path,img_dir)
                filter_idx = maxima_img_filter_idces[i]

                pipe_line(model, img_path, layer, block, to_folder,filter_idx = filter_idx, top_idx = i, style=style,task = task)

