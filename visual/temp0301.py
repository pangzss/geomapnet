import pickle

with open('./figs/AachenDay_files/maxima_patches/guidedbp/style_0_img_filter_indices_dict.txt','rb') as fp:
        filter_dic = pickle.load(fp)
    
print(filter_dic)