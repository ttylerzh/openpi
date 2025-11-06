import cv2
import time
import numpy as np
from typing import Union, Optional

class RGB_Camera:
    def __init__(self, 
                 video_id: Union[int, str] = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 target_width: int = 0,
                 target_height: int = 0,
                 video_fps: int = 0,
                 format: str = 'MJPG',
                 resize_padding: bool = False):

        self.video_id = video_id
        self.original_width = width
        self.original_height = height
        self.original_fps = fps
        
        self.target_width = target_width if target_width > 0 else width
        self.target_height = target_height if target_height > 0 else height
        self.target_fps = video_fps if video_fps > 0 else fps
        self.format = format
        self.resize_padding = resize_padding

        self.cap = None
        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0
        
        self.open()
        

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if (frame.shape[1] == self.target_width and 
            frame.shape[0] == self.target_height):
            return frame
            
        if self.resize_padding:
            result = self._padding_resize(frame)
        else:
            result = self._stretch_resize(frame)
        
        return result
    
    def _stretch_resize(self, frame: np.ndarray) -> np.ndarray:
        if (frame.shape[1] == self.target_width and 
            frame.shape[0] == self.target_height):
            return frame
        return cv2.resize(frame, (self.target_width, self.target_height))
    
    def _padding_resize(self, frame: np.ndarray) -> np.ndarray:
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
    
    def __del__(self):
        """对象被销毁时自动释放相机"""
        try:
            self.close()
        except Exception as e:
            print(f"释放相机 {self.video_id} 时出错: {e}")

    def open(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.video_id)

            if not self.cap.isOpened():
                print(f"无法打开相机: {self.video_id}")
                return False

            if self.format:
                fourcc = cv2.VideoWriter_fourcc(*self.format)
                self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

            if self.original_width > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.original_width)
            if self.original_height > 0:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.original_height)
            if self.original_fps > 0:
                self.cap.set(cv2.CAP_PROP_FPS, self.original_fps)
            
            time.sleep(0.2)
            
            # 丢弃缓冲区旧帧
            for _ in range(30):
                self.cap.read()

            self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            return True
        
        except Exception as e:
            print(f"打开相机时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

    def close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            print("相机已关闭")

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def get_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened():
            print("相机未打开")
            return None
        for _ in range(3):
            self.cap.read()
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("无法读取帧")
                return None

            if frame.size == 0:
                print("帧数据为空")
                return None
            processed_frame = cv2.cvtColor(self._process_frame(frame), cv2.COLOR_BGR2RGB)
            return processed_frame
            
        except Exception as e:
            print(f"获取帧时出错: {e}")
            return None

    def show(self, window_name: str = "Camera", show_fps: bool = True) -> bool:
        if not self.is_opened():
            print("相机未打开")
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
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 255, 0), 2)
                
                cv2.putText(frame, f"Size: {frame.shape[1]}x{frame.shape[0]}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 1)
                
                cv2.putText(frame, f"Policy: {'Padding' if self.resize_padding else 'Stretch'}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 1)
                
                cv2.imshow(window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
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

    def get_info(self) -> dict:
        return {
            'video_id': self.video_id,
            'original_width': self.original_width,
            'original_height': self.original_height,
            'original_fps': self.original_fps,
            'actual_width': self.actual_width,
            'actual_height': self.actual_height,
            'actual_fps': self.actual_fps,
            'target_width': self.target_width,
            'target_height': self.target_height,
            'target_fps': self.target_fps,
            'resize_padding': self.resize_padding,
            'is_opened': self.is_opened()
        }


if __name__ == "__main__":
    cam = RGB_Camera(video_id='/dev/cam_lwrist', width=640, height=480,
                     target_width=224, target_height=224, resize_padding=True)
    print("相机信息:", cam.get_info())
    # cam.show()
    frame = cam.get_frame()
    # cv2.imshow("Captured Frame", frame)
    # cv2.waitKey(0)
    print(type(frame), frame.shape)