from os import listdir, path, makedirs
from os.path import isfile, join
from shutil import copyfile
from random import shuffle

testset_ratio = 0.2

base_dir = '/home/share/DesnowNet/dataset/DSN_video/train/'
gt_dir = base_dir + 'gt/'
mask_dir = base_dir + 'mask/'
syn_dir = base_dir + 'synthesized/'

train_dir = 'train/'
test_dir = 'test/'

subdir_list = listdir(gt_dir)

for sub_i in range(len(subdir_list)):
    subdir_path = gt_dir + subdir_list[sub_i]
    files = listdir(subdir_path)
    test_set_number = int(len(files) * testset_ratio)
    cnt = 0
    clip_order = [x for x in range(int(len(files) / 90))]
    shuffle(clip_order)
    print('Processing dir: {:s}\t({:d} files, {:d} in testset)'.format(subdir_path, len(files), test_set_number))
    for file in files:
        syn_file = syn_dir + subdir_list[sub_i] + '/' + file
        mask_file = mask_dir + subdir_list[sub_i] + '/' + file
        gt_file = gt_dir + subdir_list[sub_i] + '/' + file

        if cnt < test_set_number:
            dst = './test/' + subdir_list[sub_i] + '/'
            if not path.exists(dst):
                makedirs(dst)
            for i in ['syn/', 'mask/', 'gt/']:
                if not path.exists(dst + i):
                    makedirs(dst + i)
            copyfile(syn_file, dst + 'syn/' + file)
            copyfile(mask_file, dst + 'mask/' + file)
            copyfile(gt_file, dst + 'gt/' + file)
        else:
            dst = './train/'
            copyfile(syn_file, dst + 'syn/' + file)
            copyfile(mask_file, dst + 'mask/' + file)
            copyfile(gt_file, dst + 'gt/' + file)
        cnt += 1













