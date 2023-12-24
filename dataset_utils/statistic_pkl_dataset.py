
import glob
import os
import shutil
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt

if __name__=='__main__':

    input_pkl_dir = '/data1/dataset/typhoon_rs/FY4ATyphoon_DatasetTools/inter_patch_knts_pkl'
    pkl_list = glob.glob(input_pkl_dir+'/*.pickle')

    class_cnt=Counter()

    for org_pkl in tqdm(pkl_list):
        # speedMS=float(os.path.basename(org_pkl).split('_')[1])
        # speedKnts=speedMS*1.94384449
        speedKnts=float(os.path.basename(org_pkl).split('_')[1])
        class_name=int(os.path.basename(org_pkl).split('_')[0])
        # count class_name 
        class_cnt[class_name]+=1
    print(class_cnt)
    # cnt to dict
    class_cnt_dict=dict(class_cnt)
    # plot bar with value above the bar

    # plt.show()
    plt.bar(class_cnt_dict.keys(), class_cnt_dict.values())
    
    # plt.show()
    plt.savefig('./pic/class_cnt_bar.png',dpi=300)

    print('done')



        
    



