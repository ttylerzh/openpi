#!/usr/bin/env python
"""
Surgery数据格式转换脚本
将原始lerobot格式转换为简化的lerobot格式
"""
import os
import cv2
import tqdm
import torch
import pickle
import logging
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

os.environ["HF_LEROBOT_HOME"] = os.path.expanduser("~/zzh/openpi/data")
cv2.setNumThreads(20)
logging.basicConfig(level=logging.INFO)

FEATURES = {
    "observation.images.left": {
        "dtype": "video",
        "shape": (3, 675, 1200),
        "names": ["channels", "height", "width"]
    },
    "observation.images.right": {
        "dtype": "video",
        "shape": (3, 675, 1200),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (20,),
        "names": ["action"],
    },
}




class SurgicalDatasetConverter:
    def __init__(self, repo_id: str, 
                fps: int, 
                root: str, 
                features: dict):
        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=root,
            image_writer_processes=0,
            image_writer_threads=20
        )
        self.image_size = (1200, 675)
        print(f"数据集创建成功: {self.dataset}")
        print(f"特征: {list(self.dataset.features.keys())}")
        print(f"数据集根目录: {self.dataset.root}")

    def cam_data_process(self, frame):
        clip = [10, 1070, 10, 1910]
        frame = frame[clip[0]:clip[1], clip[2]:clip[3], :] # H W C
        frame = cv2.cvtColor(cv2.resize(frame, self.image_size, interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB) # image_size 要填 W,H
        frame = np.transpose(frame, (2, 0, 1)).astype(np.float32)/255 # H W C -> C H W
        return frame

    def read_one_episode_data(self, episode_dir: Path, modify: bool = False):
        for file in os.listdir(episode_dir):
            if file.endswith('.pkl'):
                file_path = os.path.join(episode_dir,file)
                try:
                    with open(file_path, 'rb') as f:
                        meta_info = pickle.load(f)
                except Exception as e:
                    raise ValueError(f'Error loading {file_path}: {str(e)}')
            if file.endswith('.avi'):
                if 'left' in file:
                    left_video_path = os.path.join(episode_dir,file)     
                if 'right' in file: 
                    right_video_path = os.path.join(episode_dir,file)
        try:
            cap_left = cv2.VideoCapture(left_video_path)
            cap_right = cv2.VideoCapture(right_video_path)
        except Exception as e:
            raise ValueError(f'Error loading video files: {str(e)} in {episode_dir}')

        frame_count_left = int(cap_left.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count_right = int(cap_right.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = min(frame_count_left,frame_count_right,len(meta_info)) - 1
        frames = []
        try:
            # 1. save images
            for frame_idx in tqdm.tqdm(range(frame_count),desc='read one episode data'):
                frame = {}
                frame['task'] = 'surgery'
                img_left = self.cam_data_process(cap_left.read()[1])
                img_right = self.cam_data_process(cap_right.read()[1])
                frame["observation.images.left"] = img_left
                frame["observation.images.right"] = img_right
                
                # 2. save control info
                target_psm, filter_pos, mtm_ratio, _, actual_psm = meta_info[frame_idx].values()
                if modify:
                    L_modify_matrix = np.array([[ 1.0000,  0.0041,  0.0052,  0.0013],
                                                [-0.0040,  0.9998, -0.0176, -0.0005],
                                                [-0.0053,  0.0176,  0.9998, -0.0043],
                                                [ 0.0000,  0.0000,  0.0000,  1.0000]])
                    
                    temp = np.eye(4)
                    temp[:3,:4] = target_psm[0].reshape(3,4)
                    modified = L_modify_matrix @ temp
                    target_psm[0] = modified[:3,:4].reshape(12,)

                    R_modify_matrix = np.array([[ 0.9991,  0.0196,  0.0382,  0.0029],
                                                [-0.0212,  0.9989,  0.0418, -0.0086],
                                                [-0.0374, -0.0426,  0.9984, -0.0006],
                                                [ 0.0000,  0.0000,  0.0000,  1.0000]]) 
                    temp = np.eye(4)
                    temp[:3,:4] = target_psm[2].reshape(3,4)
                    modified = R_modify_matrix @ temp
                    target_psm[2] = modified[:3,:4].reshape(12,)
                # 2.1 save observation state - 转换数据类型和形状
                state_joint_angle = filter_pos.astype(np.float32)
                
                next_target_psm, next_filter_pos, next_mtm_ratio, _, next_actual_psm = meta_info[frame_idx + 1].values() 
                if modify:
                    L_modify_matrix = np.array([[ 1.0000,  0.0041,  0.0052,  0.0013],
                                                [-0.0040,  0.9998, -0.0176, -0.0005],
                                                [-0.0053,  0.0176,  0.9998, -0.0043],
                                                [ 0.0000,  0.0000,  0.0000,  1.0000]])
                    
                    temp = np.eye(4)
                    temp[:3,:4] = next_target_psm[0].reshape(3,4)
                    modified = L_modify_matrix @ temp
                    next_target_psm[0] = modified[:3,:4].reshape(12,)

                    R_modify_matrix = np.array([[ 0.9991,  0.0196,  0.0382,  0.0029],
                                                [-0.0212,  0.9989,  0.0418, -0.0086],
                                                [-0.0374, -0.0426,  0.9984, -0.0006],
                                                [ 0.0000,  0.0000,  0.0000,  1.0000]]) 
                    temp = np.eye(4)
                    temp[:3,:4] = next_target_psm[2].reshape(3,4)
                    modified = R_modify_matrix @ temp
                    next_target_psm[2] = modified[:3,:4].reshape(12,)
                
                action_target_matrix = next_target_psm.astype(np.float32).reshape(4, 12)
                action_tip_ratio = next_mtm_ratio.astype(np.float32)

                current_target_psm = target_psm.reshape(4,3,4)
                next_target_psm = next_target_psm.reshape(4,3,4)
                relative_pose = np.zeros((4, 6))
                for i in range(4):
                    if np.allclose(current_target_psm[i, :3, :3], 0):
                        continue
                    # matrix_current = np.concatenate([current_target_psm[i], np.array([[0, 0, 0, 1]])], axis=0)
                    # matrix_next = np.concatenate([next_target_psm[i], np.array([[0, 0, 0, 1]])], axis=0)
                    # delta_pose = np.linalg.inv(matrix_current) @ matrix_next
                    # relative_pose[i] = np.concatenate([delta_pose[:3, 3], R.from_matrix(delta_pose[:3, :3]).as_euler('xyz', degrees=False)], axis=0)
                    rot_current = R.from_matrix(current_target_psm[i, :3, :3]).as_euler('xyz', degrees=False)
                    rot_next = R.from_matrix(next_target_psm[i, :3, :3]).as_euler('xyz', degrees=False)
                    relative_pose[i, :3] = next_target_psm[i, :3, 3] - current_target_psm[i, :3, 3]
                    relative_pose[i, 3:] = rot_next - rot_current
                mask = relative_pose[:, 3:6] > np.pi
                relative_pose[:, 3:6][mask] -= 2*np.pi
                mask = relative_pose[:, 3:6] < -np.pi
                relative_pose[:, 3:6][mask] += 2*np.pi
                action_relative_pose = relative_pose.astype(np.float32)

                frame['observation.state'] = torch.concat([torch.from_numpy(state_joint_angle[0,5:12]), torch.from_numpy(state_joint_angle[2,5:12])], dim=0)

                target_matrx = action_target_matrix.reshape(4,3,4)

                L_xyz = target_matrx[0, :, 3]        # numpy array, shape: (3,)
                L_ortho6d = target_matrx[0, :, :2]   # numpy array, shape: (3, 2)
                L_gripper = action_tip_ratio[0]      # numpy scalar
                R_xyz = target_matrx[2, :, 3]        # numpy array, shape: (3,)
                R_ortho6d = target_matrx[2, :, :2]   # numpy array, shape: (3, 2)
                R_gripper = action_tip_ratio[2]      # numpy scalar

                frame['action'] = torch.concat([
                    torch.from_numpy(L_xyz),                     # (3,)
                    torch.from_numpy(L_ortho6d.flatten()),       # (6,)
                    torch.tensor([L_gripper], dtype=torch.float32),  # (1,)
                    torch.from_numpy(R_xyz),                     # (3,)
                    torch.from_numpy(R_ortho6d.flatten()),       # (6,)
                    torch.tensor([R_gripper], dtype=torch.float32),  # (1,)
                ], dim=0)  # 总shape: (20,)
                
                frames.append(frame)
                
        except Exception as e:
            raise ValueError(f'Error processing data in {episode_dir}: {str(e)}')
        
        cap_left.release()
        cap_right.release()
        return frames


    def Run(self, data_dir: str):
        eposide_dirs = os.listdir(data_dir)
        eposide_dirs = [os.path.join(data_dir, dir) for dir in eposide_dirs]
        for index, episode_dir in tqdm.tqdm(enumerate(eposide_dirs),desc='处理Episode'):
            print(f"处理第{index}个Episode: {episode_dir}")
            try:
                
                episode_data = self.read_one_episode_data(episode_dir)
                for frame in tqdm.tqdm(episode_data,desc='Add Frame'):
                    self.dataset.add_frame(frame)
                
                self.dataset.save_episode()

                print(f"当前的帧数: {self.dataset.num_frames}")
                print(f"当前的episodes数: {self.dataset.num_episodes}")
                            
            except Exception as e:
                print(f"处理episode {index} 时出错: {str(e)}")
                print(f"跳过episode: {episode_dir}")
                continue


# 使用示例
if __name__ == "__main__":
    writer = SurgicalDatasetConverter(
        repo_id="surgery_80",
        fps=30,
        root="./data/surgery_80",
        features=FEATURES,
        )

    writer.Run('/home/dell/Downloads/video')