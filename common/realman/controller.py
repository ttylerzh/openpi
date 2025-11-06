import time
import math
from Robotic_Arm.rm_robot_interface import *

class RealmanController:
    def __init__(self, robot_ip="192.168.1.18", thread_mode=None):
        try:
            if thread_mode is None:
                self.robot = RoboticArm()
            else:
                mode=rm_thread_mode_e(thread_mode)
                self.robot = RoboticArm(mode)
            self.robot.rm_create_robot_arm(robot_ip, 8080)
            time.sleep(0.5)

        except Exception as e:
            return None

    def get_joint_states_angle(self):
        tag, angles = self.robot.rm_get_joint_degree()
        return angles
    
    def get_joint_states_radian(self):
        tag, angles = self.robot.rm_get_joint_degree()
        return [math.radians(a) for a in angles]
    
    def get_eef_states_euler(self):
        tag, angles = self.robot.rm_get_joint_degree()
        pose = self.robot.rm_algo_forward_kinematics(angles, 1)
        return pose
    
    def get_eef_states_quaternion(self):
        tag, angles = self.robot.rm_get_joint_degree()
        pose = self.robot.rm_algo_forward_kinematics(angles, 0)
        return pose

    def get_gripper_states_raw(self):
        tag, state = self.robot.rm_get_gripper_state()
        return state['actpos']
    
    def get_gripper_state(self):
        tag, state = self.robot.rm_get_gripper_state()
        actpos = state['actpos']
        actpos = max(0, min(1000, actpos)) # 归一化前确保范围在0-1000
        return actpos / 1000.0
    
    def move_angle(self, target_q):
        self.robot.rm_movej_canfd(target_q, False)
        time.sleep(0.02)

    def move_radian(self, target_q):
        target_q = [math.degrees(q) for q in target_q]
        self.robot.rm_movej_canfd(target_q, False)
        time.sleep(0.02)

    def tcp_move(self, target_tcp):
        self.robot.rm_movep_canfd(target_tcp, False) # 根据传入的list长度自适应6/7：欧拉角/四元数
        time.sleep(0.02)

    def move_to_zero(self):
        self.move_angle([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def move_to_init(self):
        self.move_angle([0.0, 50.0, -140.0, 0.0, -90.0, -90.0])

    def gripper_move_raw(self, width):
        state = self.robot.rm_set_gripper_position(width, block=False, timeout=5)
        if state != 0:
           raise Exception("Move failed")
        time.sleep(0.03)

    def gripper_move(self, width):
        assert 0.0 <= width <= 1.0, "Width must be between 0.0 and 1.0"
        state = self.robot.rm_set_gripper_position(int(width*1000), block=False, timeout=5)
        if state != 0:
           raise Exception("Move failed")
        time.sleep(0.03)

    def gripper_open(self):
        width=self.get_gripper_states_raw()
        if(width <= 800):
            state = self.robot.rm_set_gripper_release(speed=500, block=False, timeout=5)
            if state != 0:
                raise Exception("Open failed")
            time.sleep(0.1)

    def gripper_close(self):
        width=self.get_gripper_states_raw()
        if(width >= 800):
            state = self.robot.rm_set_gripper_pick_on(speed=500, force=200, block=False, timeout=5)
            if state != 0:
                raise Exception("Close failed")
            time.sleep(0.1)


if __name__ == "__main__":
    left_arm = RealmanController("192.168.1.18", 2)
    right_arm = RealmanController("192.168.1.19")
    # left_arm.move_to_zero()
    # time.sleep(3)
    left_arm.move_to_init()
    left_arm.gripper_close()
    time.sleep(0.1)
    left_arm.gripper_open()
    right_arm.gripper_close()
    time.sleep(0.1)
    right_arm.gripper_open()