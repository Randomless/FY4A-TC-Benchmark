# re-arrange pkl folder to class-wise sub-folders [0~5]

import glob
import os
import shutil
from tqdm import tqdm


def rename_pkl_to_knts(input_pkl_dir, output_pkl_dir, if_copy=False):
    if not os.path.exists(output_pkl_dir):
        os.makedirs(output_pkl_dir, exist_ok=True)
    pkl_list = glob.glob(input_pkl_dir + "/*.pickle")
    for org_pkl in tqdm(pkl_list):
        speedMS = float(os.path.basename(org_pkl).split("_")[1])
        speedKnts = speedMS * 1.94384449
        new_pkl = os.path.join(
            output_pkl_dir,
            os.path.basename(org_pkl).replace("_" + str(speedMS), "_" + str(speedKnts)),
        )
        if if_copy:
            shutil.copy(org_pkl, new_pkl)
        else:
            shutil.move(org_pkl, new_pkl)

    return


def reArrange_pkl(raw_pkl_path, pkl_path):
    for i in range(6):
        os.makedirs(os.path.join(pkl_path, str(i)), exist_ok=True)
    for i in range(6):
        class_pkl_files = glob.glob(os.path.join(raw_pkl_path, str(i) + "*.pickle"))
        print(f"class {i} has {len(class_pkl_files)} files")
        for file in class_pkl_files:
            shutil.move(file, os.path.join(pkl_path, str(i)))


def reArrange_png(raw_png_path, png_path):
    for i in range(6):
        os.makedirs(os.path.join(png_path, str(i)), exist_ok=True)
    for i in range(6):
        class_pkl_files = glob.glob(os.path.join(raw_png_path, str(i) + "*.png"))
        print(f"class {i} has {len(class_pkl_files)} files")
        for file in class_pkl_files:
            shutil.move(file, os.path.join(png_path, str(i)))


def cnt_pickle_glob_recursive(path):
    pickle_fns = glob.glob(os.path.join(path, "**/*.pickle"), recursive=True)
    return len(pickle_fns)


if __name__ == "__main__":

    ms_pkl_path = "/data4/dataset/typhoon_rs/FIN_v1_all/FIN_v1"
    knts_pkl_path = "/data4/dataset/typhoon_rs/FIN_v1_all/FIN_v1_knts"
    # rename_pkl_to_knts(ms_pkl_path, knts_pkl_path)
    # reArrange_pkl(knts_pkl_path, knts_pkl_path.replace("knts", "knts_classwise"))

    knts_png_path = "/data4/dataset/typhoon_rs/FIN_v1_all/png_channel"
    reArrange_png(knts_png_path, knts_png_path.replace("channel", "channel_classwise"))

    # pca_inter_patch
    # raw_pkl_path = '/data1/tmp_datasetDir/pca_inter_patch_knts_pkl'
    # # classwise_pkl_path = '/data1/tmp_datasetDir/pca_inter_patch_knts_pkl_classwise'
    # reArrange_pkl(raw_pkl_path,raw_pkl_path.replace('pkl','pkl_classwise'))

    # cnt1=cnt_pickle_glob_recursive('inter_patch_knts_pkl_classwise')
    # cnt2=cnt_pickle_glob_recursive('pca_inter_patch_knts_pkl_classwise')
    # print(cnt1-cnt2)
    # # 3023 3022
    # print('done')


# class 0 has 180 files
# class 1 has 836 files
# class 2 has 404 files
# class 3 has 292 files
# class 4 has 131 files
# class 5 has 148 files
