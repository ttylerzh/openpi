import cv2
import time
import numpy as np
import pyrealsense2 as rs


'''
    top: Intel RealSense D455 序列号: 235422301854
    lwrist: Intel RealSense D435I 序列号: 135122072084
    rwrist: Intel RealSense D435I 序列号: 233622079333
'''

def list_realsense_devices():
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        devices.append({'serial': serial, 'name': name})
    for i, d in enumerate(devices):
        print(f"设备{i}: {d['name']} 序列号: {d['serial']}")

class RealSense_Camera:
    def __init__(self, width=640, height=480, fps=30, target_width=0, target_height=0, resize_padding=False, serial_number=None):
        self.width = width
        self.height = height
        self.fps = fps
        self.target_width = target_width if target_width > 0 else width
        self.target_height = target_height if target_height > 0 else height
        self.resize_padding = resize_padding
        self.serial_number = serial_number

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.serial_number:
            self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        self.profile = None
        self.open()

    def open(self):
        try:
            self.profile = self.pipeline.start(self.config)
            time.sleep(0.5)
            return True
        except Exception as e:
            print(f"无法打开RealSense相机: {e}")
            return False

    def close(self):
        try:
            self.pipeline.stop()
            print("RealSense相机已关闭")
        except Exception as e:
            print(f"关闭RealSense相机时出错: {e}")

    def is_opened(self):
        return self.profile is not None

    def _process_frame(self, frame):
        if frame.shape[1] == self.target_width and frame.shape[0] == self.target_height:
            return frame
        if self.resize_padding:
            original_height, original_width = frame.shape[:2]
            ratio = min(self.target_width / original_width, self.target_height / original_height)
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            padded = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
            x_offset = (self.target_width - new_width) // 2
            y_offset = (self.target_height - new_height) // 2
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            return padded
        else:
            return cv2.resize(frame, (self.target_width, self.target_height))

    def get_frame(self):
        if not self.is_opened():
            print("RealSense相机未打开")
            return None
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("未获取到彩色帧")
                return None
            frame = np.asanyarray(color_frame.get_data())
            processed_frame = self._process_frame(frame)
            return processed_frame
        except Exception as e:
            print(f"获取帧时出错: {e}")
            return None

    def show(self, window_name="RealSense Camera", show_fps=True):
        if not self.is_opened():
            print("RealSense相机未打开")
            return False
        print(f"显示窗口: {window_name}")
        print("按 'q' 或 'ESC' 退出显示")
        fps_counter = 0
        start_time = time.time()
        current_fps = 0
        try:
            while True:
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = 30 / elapsed_time if elapsed_time > 0 else 0
                    start_time = time.time()
                if show_fps and current_fps > 0:
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(frame, f"Policy: {'Padding' if self.resize_padding else 'Stretch'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break
            return True
        except KeyboardInterrupt:
            print("用户中断")
            return True
        except Exception as e:
            print(f"显示时出错: {e}")
            return False
        finally:
            try:
                cv2.destroyWindow(window_name)
            except:
                pass

    def get_info(self):
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'target_width': self.target_width,
            'target_height': self.target_height,
            'resize_padding': self.resize_padding,
            'serial_number': self.serial_number,
            'is_opened': self.is_opened()
        }

if __name__ == "__main__":
    # list_realsense_devices()

    serial = '233622079333'
    cam = RealSense_Camera(width=640, height=480, target_width=224, target_height=224, resize_padding=True, serial_number=serial)
    cam.show()