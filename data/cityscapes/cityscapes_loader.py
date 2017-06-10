from __future__ import division
import json
import os
import numpy as np
import scipy.misc
from glob import glob

class cityscapes_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split='train',
                 crop_bottom=True, # Get rid of the car logo
                 sample_gap=2,  # Sample every two frames to match KITTI frame rate
                 img_height=171, 
                 img_width=416,
                 seq_length=5):
        self.dataset_dir = dataset_dir
        self.split = split
        # Crop out the bottom 25% of the image to remove the car logo
        self.crop_bottom = crop_bottom
        self.sample_gap = sample_gap
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        assert seq_length % 2 != 0, 'seq_length must be odd!'
        self.frames = self.collect_frames(split)
        self.num_frames = len(self.frames)
        if split == 'train':
            self.num_train = self.num_frames
        else:
            self.num_test = self.num_frames
        print('Total frames collected: %d' % self.num_frames)
        
    def collect_frames(self, split):
        img_dir = self.dataset_dir + '/leftImg8bit_sequence/' + split + '/'
        city_list = os.listdir(img_dir)
        frames = []
        for city in city_list:
            img_files = glob(img_dir + city + '/*.png')
            for f in img_files:
                frame_id = os.path.basename(f).split('leftImg8bit')[0]
                frames.append(frame_id)
        return frames

    def get_train_example_with_idx(self, tgt_idx):
        tgt_frame_id = self.frames[tgt_idx]
        if not self.is_valid_example(tgt_frame_id):
            return False
        example = self.load_example(self.frames[tgt_idx])
        return example

    def load_intrinsics(self, frame_id, split):
        city, seq, _, _ = frame_id.split('_')
        camera_file = os.path.join(self.dataset_dir, 'camera',
                                   split, city, city + '_' + seq + '_*_camera.json')
        camera_file = glob(camera_file)[0]
        with open(camera_file, 'r') as f: 
            camera = json.load(f)
        fx = camera['intrinsic']['fx']
        fy = camera['intrinsic']['fy']
        u0 = camera['intrinsic']['u0']
        v0 = camera['intrinsic']['v0']
        intrinsics = np.array([[fx, 0, u0],
                               [0, fy, v0],
                               [0,  0,  1]])
        return intrinsics

    def is_valid_example(self, tgt_frame_id):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence', 
                                self.split, city, curr_frame_id + 'leftImg8bit.png')
            if not os.path.exists(curr_image_file):
                return False
        return True

    def load_image_sequence(self, tgt_frame_id, seq_length, crop_bottom):
        city, snippet_id, tgt_local_frame_id, _ = tgt_frame_id.split('_')
        half_offset = int((self.seq_length - 1)/2 * self.sample_gap)
        image_seq = []
        for o in range(-half_offset, half_offset + 1, self.sample_gap):
            curr_local_frame_id = '%.6d' % (int(tgt_local_frame_id) + o)
            curr_frame_id = '%s_%s_%s_' % (city, snippet_id, curr_local_frame_id)
            curr_image_file = os.path.join(self.dataset_dir, 'leftImg8bit_sequence', 
                                self.split, city, curr_frame_id + 'leftImg8bit.png')
            curr_img = scipy.misc.imread(curr_image_file)
            raw_shape = np.copy(curr_img.shape)
            if o == 0:
                zoom_y = self.img_height/raw_shape[0]
                zoom_x = self.img_width/raw_shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            if crop_bottom:
                ymax = int(curr_img.shape[0] * 0.75)
                curr_img = curr_img[:ymax]
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y
    
    def load_example(self, tgt_frame_id, load_gt_pose=False):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(tgt_frame_id, self.seq_length, self.crop_bottom)
        intrinsics = self.load_intrinsics(tgt_frame_id, self.split)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_frame_id.split('_')[0]
        example['file_name'] = tgt_frame_id[:-1]
        return example

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out