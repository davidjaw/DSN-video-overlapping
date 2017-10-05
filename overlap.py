import cv2
from os import listdir
from os.path import isfile, join
import util
import tensorflow as tf
from random import shuffle
import multiprocessing

mask_dir = './synthetic_video/'
base_dir = './wo_snow_vid/'
mask_attr_route = './mask_attr'
batch_frames = 90

out_base_dir = '/home/media/overlap/'
skip = 0

testset_ratio = 0.2


def main():
    # read files from dir
    base_files = [file for file in listdir(base_dir) if isfile(join(base_dir, file))]
    mask_files = [mask_dir + file for file in listdir(mask_dir) if isfile(join(mask_dir, file))]
    mask_attr = util.read_mask_attr(mask_attr_route)
    haze = mask_files[:10]
    mask_files = mask_files[10:]

    with tf.Session() as sess:
        # define overlap operation
        base = tf.placeholder(tf.float32, [batch_frames, None, None, 3])
        mask = tf.placeholder(tf.float32, [batch_frames, None, None, 3])
        r_brightness = tf.reduce_max(base) * tf.random_uniform((), 0.8, 1)
        overlapping = mask * r_brightness + (tf.ones_like(mask, tf.float32) - mask) * base

        tmp = 0
        # read videos
        for video_index in range(len(base_files)):
            if video_index < skip:
                continue
            vid_file_r = base_dir + base_files[video_index]
            base_vid = cv2.VideoCapture(vid_file_r)
            if not base_vid.isOpened():
                print('Error while opening video: %s' % vid_file_r)
                break
            width, height, frame_num = util.get_vid_info(base_vid)
            tmp += int(frame_num / 90)
            print(tmp)

            # random the order of clips into testset and trainset
            # random_set_order = [1 if x > int(frame_num / 90) * testset_ratio else 0 for x in range(int(frame_num / 90))]
            # shuffle(random_set_order)
            # for index in range(0, frame_num, batch_frames):
            #     if index + batch_frames > frame_num:
            #         break
            #     out_type = 'test/' if random_set_order[int(index / 90)] == 0 else 'train/'
            #     # random masks for each batch
            #     frames, resize_size = util.stack_frames(base_vid, batch_frames, width, height)
            #     masks, overlapped_mask_attr = util.random_masks(mask_files, batch_frames, resize_size, mask_attr)
            #
            #     frames /= 255.
            #     masks /= 255.
            #
            #     overlapped = sess.run(overlapping, feed_dict={base: frames, mask: masks})
            #     util.write_img_output(overlapped, overlapped_mask_attr, out_base_dir + out_type + 'syn/', batch_frames,
            #                           filename='mask_{:d}_{:d}'.format(video_index, int(index / batch_frames)))
            #     util.write_img_output(frames, overlapped_mask_attr, out_base_dir + out_type + 'gt/', batch_frames,
            #                           filename='mask_{:d}_{:d}'.format(video_index, int(index / batch_frames)))
            #     util.write_img_output(masks, overlapped_mask_attr, out_base_dir + out_type + 'mask/', batch_frames,
            #                           filename='mask_{:d}_{:d}'.format(video_index, int(index / batch_frames)))
            #     print('mask_{:d}_{:d}.mp4 is created on {:s} set'.format(video_index, int(index / batch_frames), out_type))


if __name__ == '__main__':
    main()
