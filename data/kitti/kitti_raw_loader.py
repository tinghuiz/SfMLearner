from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc

class kitti_raw_loader(object):
    def __init__(self, 
                 dataset_dir,
                 split,
                 img_height=256,
                 img_width=256,
                 seq_length=5):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        static_frames_file = dir_path + '/static_frames.txt'
        test_scene_file = dir_path + '/test_scenes_' + split + '.txt'
        with open(test_scene_file, 'r') as f:
            test_scenes = f.readlines()
        self.test_scenes = [t[:-1] for t in test_scenes]
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length
        self.cam_ids = ['02', '03']
        self.date_list = ['2011_09_26', '2011_09_28', '2011_09_29', 
                          '2011_09_30', '2011_10_03']
        self.collect_static_frames(static_frames_file)
        self.collect_train_frames()

    def collect_static_frames(self, static_frames_file):
        with open(static_frames_file, 'r') as f:
            frames = f.readlines()
        self.static_frames = []
        for fr in frames:
            if fr == '\n':
                continue
            date, drive, frame_id = fr.split(' ')
            curr_fid = '%.10d' % (np.int(frame_id[:-1]))
            for cid in self.cam_ids:
                self.static_frames.append(drive + ' ' + cid + ' ' + curr_fid)
        
    def collect_train_frames(self):
        all_frames = []
        for date in self.date_list:
            drive_set = os.listdir(self.dataset_dir + date + '/')
            for dr in drive_set:
                drive_dir = os.path.join(self.dataset_dir, date, dr)
                if os.path.isdir(drive_dir):
                    if dr[:-5] in self.test_scenes:
                        continue
                    for cam in self.cam_ids:
                        img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
                        N = len(glob(img_dir + '/*.png'))
                        for n in range(N):
                            frame_id = '%.10d' % n
                            all_frames.append(dr + ' ' + cam + ' ' + frame_id)
                        

        for s in self.static_frames:
            try:
                all_frames.remove(s)
                # print('removed static frame from training: %s' % s)
            except:
                pass

        self.train_frames = all_frames
        self.num_train = len(self.train_frames)

    def is_valid_sample(self, frames, tgt_idx):
        N = len(frames)
        tgt_drive, cid, _ = frames[tgt_idx].split(' ')
        half_offset = int((self.seq_length - 1)/2)
        min_src_idx = tgt_idx - half_offset
        max_src_idx = tgt_idx + half_offset
        if min_src_idx < 0 or max_src_idx >= N:
            return False
        min_src_drive, min_src_cid, _ = frames[min_src_idx].split(' ')
        max_src_drive, max_src_cid, _ = frames[max_src_idx].split(' ')
        if tgt_drive == min_src_drive and tgt_drive == max_src_drive and cid == min_src_cid and cid == max_src_cid:
            return True
        return False

    def get_train_example_with_idx(self, tgt_idx):
        if not self.is_valid_sample(self.train_frames, tgt_idx):
            return False
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_image_sequence(self, frames, tgt_idx, seq_length):
        half_offset = int((seq_length - 1)/2)
        image_seq = []
        for o in range(-half_offset, half_offset + 1):
            curr_idx = tgt_idx + o
            curr_drive, curr_cid, curr_frame_id = frames[curr_idx].split(' ')
            curr_img = self.load_image_raw(curr_drive, curr_cid, curr_frame_id)
            if o == 0:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx, self.seq_length)
        tgt_drive, tgt_cid, tgt_frame_id = frames[tgt_idx].split(' ')
        intrinsics = self.load_intrinsics_raw(tgt_drive, tgt_cid, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive + '_' + tgt_cid + '/'
        example['file_name'] = tgt_frame_id
        return example

    def load_image_raw(self, drive, cid, frame_id):
        date = drive[:10]
        img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cid, 'data', frame_id + '.png')
        img = scipy.misc.imread(img_file)
        return img

    def load_intrinsics_raw(self, drive, cid, frame_id):
        date = drive[:10]
        calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')

        filedata = self.read_raw_calib_file(calib_file)
        P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
        intrinsics = P_rect[:3, :3]
        return intrinsics

    def read_raw_calib_file(self,filepath):
        # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                        data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                        pass
        return data

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out


