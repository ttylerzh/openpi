import os
os.environ["HF_LEROBOT_HOME"] = os.path.expanduser("~/zzh/openpi/data")
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from datasets import load_dataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDatasetMetadata
from controller import RealmanController

np.set_printoptions(precision=4, suppress=True)


def make_train_example(repo_id, episode_index=0):
    root = HF_LEROBOT_HOME / repo_id
    meta = LeRobotDatasetMetadata(repo_id=repo_id,root=root,)

    episode_index = 0
    parquet_rel = meta.get_data_file_path(episode_index)
    parquet_path = root / parquet_rel

    hf_ds = load_dataset("parquet",data_files=str(parquet_path),split="train",)
    states = np.array((hf_ds[:]['state']))
    actions = np.array((hf_ds[:]['action']))
    # image_high = np.array(hf_ds[:]['top_image'])
    # image_left = np.array(hf_ds[:]['lwrist_image'])
    # image_right = np.array(hf_ds[:]['rwrist_image'])

    return states, actions


def plot_comparison(arr1: np.ndarray, arr2: np.ndarray, idx_label: str,
                    n_cols: int = 7, out_dir: str = "./assets/figures"):
    assert arr1.shape == arr2.shape, "两个数组必须形状相同,arr1.shape={}, arr2.shape={}".format(arr1.shape, arr2.shape)
    num_dims = arr1.shape[1]
    n_rows = math.ceil(num_dims / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 5 * n_rows),
        squeeze=False,
        dpi=150
    )
    axes = axes.flatten()

    for i in range(num_dims):
        ax = axes[i]
        ax.plot(arr1[:, i], label='pred_actions', marker='o', markersize=4, linewidth=1.5)
        ax.plot(arr2[:, i], label='real_actions', marker='x', markersize=4, linewidth=1.5)
        ax.set_title(f'Dim {i+1}', fontsize=12)
        ax.set_xlabel('Sample idx', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=8)

    for j in range(num_dims, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()

    save_path = f"{out_dir}/comparison_{idx_label}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"✅ Saved comparison plot to {save_path}")

def train_action(repo_id, episode_index=0):
    left_arm = RealmanController("192.168.1.18", 2)
    right_arm = RealmanController("192.168.1.19")

    left_arm.move_to_init()
    right_arm.move_to_init()
    states, actions = make_train_example(repo_id, episode_index)
    for action in states:
        # print(f"state: {left_arm.get_joint_states_radian()}, {right_arm.get_joint_states_radian()}")
        print(f"action: {action}")
        left_arm.move_radian(action[:6])
        if(action[6] < 0.8):
            left_arm.gripper_close()
        else:
            left_arm.gripper_open()
        right_arm.move_radian(action[7:13])
        if(action[13] < 0.8):
            right_arm.gripper_close()
        else:
            right_arm.gripper_open()
        time.sleep(0.02)


if __name__ == "__main__":
    train_action(repo_id='realman/transfer_250910', episode_index=0)