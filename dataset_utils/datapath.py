import os
import numpy as np

_dataset_path_dict={
    'ucm': 'dataset/UCMerced/UCMerced_LandUse/UCMerced_LandUse/Images',
    'patch': 'dataset/patch_pkl_classwise',
    'inter_patch_knts': 'dataset/inter_patch_knts_pkl_classwise',
    'pca4_inter_patch_knts': 'dataset/pca4_inter_patch_knts_pkl_classwise',
    'inter_patch_pkl_SH_knts':'dataset/inter_patch_pkl_SH_knts_classwise'
}


def img_path_generator(dataset='ucm'):
    img_dir = _dataset_path_dict[dataset]
    img_path = []
    img_labels = []
    dicts = os.listdir(img_dir)
    for root, _, files in os.walk(img_dir):
        for name in files:
            img_path.append(os.path.join(root, name))
            label_name = root.split('/')[-1]
            img_labels.append(int(dicts.index(label_name)))
    return np.array(img_path, dtype=object), np.array(img_labels), len(dicts)
