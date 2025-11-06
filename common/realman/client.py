import cv2
import time
import numpy as np
from openpi_client import websocket_client_policy
from common.realsense_tools import RealSense_Camera
from common.realman.controller import RealmanController

np.set_printoptions(precision=6, suppress=True)


def main(save_image=False):
    cam_top = RealSense_Camera(serial_number='235422301854', width=640, height=480,target_width=224, target_height=224, resize_padding=True)
    cam_lwrist = RealSense_Camera(serial_number='135122072084', width=640, height=480,target_width=224, target_height=224, resize_padding=True)
    cam_rwrist = RealSense_Camera(serial_number='233622079333', width=640, height=480,target_width=224, target_height=224, resize_padding=True)

    left_arm = RealmanController("192.168.1.18", 2)
    right_arm = RealmanController("192.168.1.19")
    left_arm.gripper_open()
    left_arm.move_to_init()
    right_arm.gripper_open()
    right_arm.move_to_init()
    
    time.sleep(2)
    # exit()
    client = websocket_client_policy.WebsocketClientPolicy(host="192.168.50.186", port=8000)

    if save_image:
        import glob, os
        for f in glob.glob("assets/figures/*.png"):
            os.remove(f)

    idx = 0
    while True:
        start_time = time.time()
        obs = {
            "observation/top_image": np.array(cam_top.get_frame()),
            "observation/lwrist_image": np.array(cam_lwrist.get_frame()),
            "observation/rwrist_image": np.array(cam_rwrist.get_frame()),
            "observation/state": np.concatenate([left_arm.get_joint_states_radian(), np.array([left_arm.get_gripper_state()]), 
                                                 right_arm.get_joint_states_radian(), np.array([right_arm.get_gripper_state()])], axis=0),
            "prompt": "transfer the bottle to the box"
        }
        print(f"Time taken to get observation: {time.time() - start_time} seconds")
        
        if save_image:
            cv2.imwrite(f"assets/figures/top_{idx}.png", cv2.cvtColor(obs['observation/top_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"assets/figures/lwrist_{idx}.png", cv2.cvtColor(obs['observation/lwrist_image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"assets/figures/rwrist_{idx}.png", cv2.cvtColor(obs['observation/rwrist_image'], cv2.COLOR_RGB2BGR))
        # breakpoint()
        time_start = time.time()
        action_chunk = client.infer(obs)['actions']
        print(f"Time taken to get action from server: {time.time() - time_start} seconds")

        for action in action_chunk:
            # print(f"state: {left_arm.get_joint_states_radian()}, {right_arm.get_joint_states_radian()}")
            # print(action)
            # breakpoint()
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
        idx += 1


if __name__ == "__main__":
    main(save_image=True)