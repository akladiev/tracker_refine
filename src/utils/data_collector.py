from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
import torch
import h5py


class RefineDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = h5py.File(str(Path(data_path) / 'refine_data.hdf5'), 'r')

    def __len__(self):
        return len(self.data['shift_gt'])

    def __del__(self):
        self.data.close()

    def __getitem__(self, idx):
        shift_init = self.data['shift_init'][idx]
        if shift_init != 0:
            # Randomly reset init frame to the previous frame to augment data
            # by adding more close pairs
            shift_init = -1 if random.random() < 0.5 else shift_init

        shift_prev = self.data['shift_prev'][idx]
        shift_gt = self.data['shift_gt'][idx]

        current_template = self.data['z_current'][idx]
        init_template = self.data['z_ground_truth'][idx + shift_init]
        previous_template = self.data['z_refined'][idx + shift_prev]
        ground_truth_template = self.data['z_ground_truth'][idx + shift_gt]

        net_input = np.concatenate((init_template, previous_template, current_template), axis=0)
        sample = {
            'input': torch.Tensor(net_input),
            'target': torch.Tensor(ground_truth_template)
        }
        return sample


class DataCollector:
    def __init__(self, save_path, data_shape):
        self.start_frame = 0
        self.save_path = Path(save_path)
        self.data = h5py.File(str(Path(save_path) / 'refine_data.hdf5'), 'w')
        self.save_path.mkdir(exist_ok=True, parents=True)

        # features of detection from the current frame
        self.data.create_dataset('z_current', shape=data_shape)
        # features of template refined during tracking (for stages > 2)
        self.data.create_dataset('z_refined', shape=data_shape)
        # features of ground-truth template (prediction for the next frame)
        self.data.create_dataset('z_ground_truth', shape=data_shape)
        # shift in index to get init frame
        self.data.create_dataset('shift_init', shape=(data_shape[0],), dtype=np.int64)
        # shift in index to get previous frame (0 for start/reset frame, -1 for others)
        self.data.create_dataset('shift_prev', shape=(data_shape[0],), dtype=np.int64)
        # shift in index to get ground truth frame (0 for the last frame, 1 for the others)
        self.data.create_dataset('shift_gt', shape=(data_shape[0],), dtype=np.int64)

    def __del__(self):
        self.data.close()

    def add(self, idx, cur, pre, gt, i_pre, i_gt, i_init):
        self.data['z_current'][idx] = cur.cpu().data
        self.data['z_refined'][idx] = pre.cpu().data
        self.data['z_ground_truth'][idx] = gt.cpu().data

        self.data['shift_prev'][idx] = i_pre
        self.data['shift_gt'][idx] = i_gt
        self.data['shift_init'][idx] = i_init

    def get_pre_gt_shifts(self, frame_idx, num_frames):
        end_frame = num_frames - 1
        prev_shift = 0 if frame_idx <= self.start_frame else -1  # no prev frame for first frame
        gt_shift = 0 if frame_idx >= end_frame else 1  # no next ground truth for last frame
        return prev_shift, gt_shift

    def add_init(self, frame_idx, num_frames, init_feat):
        prev_shift, gt_shift = self.get_pre_gt_shifts(frame_idx, num_frames)
        self.add(idx=frame_idx, cur=init_feat, pre=init_feat,
                 gt=init_feat, i_pre=prev_shift, i_gt=gt_shift, i_init=0)

    def add_tracking(self, frame_idx, num_frames, init_shift, cur_feat, pre_feat, gt_feat):
        prev_shift, gt_shift = self.get_pre_gt_shifts(frame_idx, num_frames)
        self.add(idx=frame_idx, cur=cur_feat, pre=pre_feat, gt=gt_feat,
                 i_pre=prev_shift, i_gt=gt_shift, i_init=-init_shift)

    def save(self):
        self.data.flush()
