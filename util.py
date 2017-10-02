import cv2
import numpy as np
import os


def get_vid_info(video_handle):
    length = int(video_handle.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video_handle.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_handle.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height, length


def crop_frames(video_h, batch_size, start_x, start_y, resize_size):
    batch = []
    frame_number = int(video_h.get(cv2.CAP_PROP_FRAME_COUNT))
    r_frame_start = np.random.randint(0, frame_number - batch_size - 1)
    for i in range(r_frame_start):
        _, _ = video_h.read()
    for index in range(batch_size):
        ret, frame = video_h.read()
        if ret:
            frame = frame[start_y:start_y + resize_size[1], start_x:start_x + resize_size[0], :]
            batch.append(frame.astype('float32'))
    batch = np.expand_dims(batch, 0)[0]
    return batch


def stack_frames(video_h, batch_size, width, height):
    resize_factor = (lambda a, b: 640 / b if b > a else 640 / a)(width, height)
    resize_size = (int(resize_factor * width), int(resize_factor * height))
    batch = []
    for index in range(batch_size):
        ret, frame = video_h.read()
        if ret:
            frame = cv2.resize(frame, resize_size)
            batch.append(frame.astype('float32'))
    batch = np.expand_dims(batch, 0)[0]
    return batch, resize_size


def read_mask_attr(mask_attr_route):
    mask_lists = {}
    with open(mask_attr_route) as f:
        read_data = f.read()
        for line in read_data.split('\n'):
            if line == '':
                break
            attrs = line.split('\t')
            mask_number = attrs.pop(0)
            attrs = ''.join(attrs)
            if attrs in mask_lists.keys():
                mask_lists[attrs].append(mask_number)
            else:
                mask_lists[attrs] = []
                mask_lists[attrs].append(mask_number)
    f.close()
    return mask_lists


def write_output(frames, mask_attr, output_dir, batch_size, filename):
    frames *= 255.
    frames = frames.astype('uint8')
    file_dir = output_dir + mask_attr
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    writer = cv2.VideoWriter(filename=file_dir + '/' + filename, fourcc=fourcc, fps=30.0,
                             frameSize=(frames.shape[2], frames.shape[1]), isColor=True)
    for i in range(batch_size):
        writer.write(frames[i, :, :, :])
    writer.release()


def write_img_output(frames, mask_attr, output_dir, batch_size, filename):
    frames *= 255.
    frames = frames.astype('uint8')
    file_dir = output_dir + mask_attr
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    # fourcc = cv2.VideoWriter_fourcc(*'PIM1')
    # writer = cv2.VideoWriter(filename=file_dir + '/' + filename, fourcc=fourcc, fps=30.0,
    #                          frameSize=(frames.shape[2], frames.shape[1]), isColor=True)
    # for i in range(batch_size):
    #     writer.write(frames[i, :, :, :])
    # writer.release()
    for i in range(batch_size):
        cv2.imwrite('{:s}{:s}/{:s}_{:d}.jpg'.format(output_dir, mask_attr, filename, i), frames[i, :, :, :],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def random_masks(mask_lists, batch_size, resize_size, mask_attr):
    r_attr = ''.join([str(x + 1) for x in np.random.randint(3, size=3).tolist()])
    r_coord_w, r_coord_h = [np.random.randint(0, 1000 - resize_size[0]), np.random.randint(0, 1000 - resize_size[1])]

    # check if attr exist
    while r_attr not in mask_attr.keys():
        r_attr = ''.join([str(x + 1) for x in np.random.randint(3, size=3).tolist()])

    # select a random mask from attr list
    mask_number = mask_attr[r_attr][np.random.randint(0, len(mask_attr[r_attr]))]
    mask_vid = cv2.VideoCapture(mask_lists[int(mask_number) - 1])
    crop_mask = crop_frames(mask_vid, batch_size, r_coord_w, r_coord_h, resize_size)

    mask_vid.release()
    return crop_mask, r_attr

