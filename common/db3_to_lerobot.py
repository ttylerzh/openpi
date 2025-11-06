import os
os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
os.environ["HF_LEROBOT_HOME"] = os.path.expanduser("~/zzh/openpi/data")
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Generator, Tuple, Optional
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


FEATURES = {
    "top_image": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "lwrist_image": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "rwrist_image": {
        "dtype": "video",
        "shape": (224, 224, 3),
        "names": ["height", "width", "channel"],
    },
    "state": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": ["action"],
    },
}

MAP = {
    'top_image': {
        'topic': '/cam_top/cam_top/color/image_raw',
        'type': 'rgb'
    },
    'lwrist_image': {
        'topic': '/cam_lwrist/cam_lwrist/color/image_raw',
        'type': 'rgb'
    },
    'rwrist_image': {
        'topic': '/cam_rwrist/cam_rwrist/color/image_raw',
        'type': 'rgb'
    },
    'state': { 
        'topic': ['/left_arm/joint_states', '/right_arm/joint_states'],
        'type': 'state' 
    }, 
}

SOURCES = [
    {
        "path": "~/disk/data_ros2/transfer_250910",
        "task": "transfer the bottle to the box",
        "episodes": 30
    },
]


class ConvertConfig:
    def __init__(
        self,
        source_dirs: List[Dict[str, Any]],
        repo_id: str,
        robot_type: str,
        features: Dict[str, Dict[str, Any]],
        fps: int = 30,
        topic_map: Dict[str, Dict[str, Any]] = None,
        add_action: Optional[str] = None,
        gripper_normalize: bool = False,
        gripper_min: float = 0.0,
        gripper_max: float = 1.0,
        image_threads: int = 64,
        image_processes: int = 32,
        merge_joints: bool = True
    ):
        self.source_dirs = source_dirs
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.features = features
        self.fps = fps
        self.topic_map = topic_map or {}
        self.add_action = add_action
        self.gripper_normalize = gripper_normalize
        self.gripper_min = gripper_min
        self.gripper_max = gripper_max
        self.image_threads = image_threads
        self.image_processes = image_processes
        self.merge_joints = merge_joints


class ROS2BagToLeRobotConverter:
    def __init__(self, config: ConvertConfig):
        self.config = config
        self.bridge = CvBridge()
        self.repo_id = config.repo_id
        self.robot_type = config.robot_type
        self.features = config.features
        self.fps = config.fps
        self.merge_joints = config.merge_joints
        self.gripper_normalize = config.gripper_normalize
        self.gripper_min = config.gripper_min
        self.gripper_max = config.gripper_max
        self.add_action = config.add_action
        
        self.state_feature_name = self._detect_state_feature_name()
        self.action_feature_name = config.add_action if config.add_action else None

        if self.gripper_normalize and self.gripper_min >= self.gripper_max:
            raise ValueError("gripper_min must be less than gripper_max for normalization")

        self.image_topic_map = {}
        for feature_name, topic_info in config.topic_map.items():
            if topic_info.get('type') == 'rgb':
                self.image_topic_map[feature_name] = topic_info['topic']
        
        self.image_shapes = {
            name: config["shape"] for name, config in self.features.items()
            if config.get("dtype") == "video"
        }
        
        # Create LeRobot dataset
        self.dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            robot_type=self.robot_type,
            fps=self.fps,
            features=self.features,
            image_writer_threads=config.image_threads,
            image_writer_processes=config.image_processes,
        )

    def _detect_state_feature_name(self) -> str:
        for name, config in self.features.items():
            if config.get('dtype') == 'float32' and len(config.get('shape', [])) == 1:
                return name
        raise ValueError("No valid state feature found in features dictionary.")

    def get_topic_types(self, db3_path: str) -> Dict[str, str]:
        """Automatically infer topics and types from db3 file"""
        storage_options = StorageOptions(uri=db3_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)
        topics = reader.get_all_topics_and_types()

        return {topic.name: topic.type for topic in topics}

    def read_rosbag2(self, bag_path: str, topic_type_map: Dict[str, str]) -> Dict[str, List[Tuple[int, Any]]]:
        buffers = {}
        all_topics = set()
        for feature_info in self.config.topic_map.values():
            topic = feature_info['topic']
            if isinstance(topic, list):
                all_topics.update(topic)
            else:
                all_topics.add(topic)
        
        for topic in all_topics:
            buffers[topic] = []

        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic not in buffers:
                continue

            msg_type = topic_type_map.get(topic)
            if msg_type == 'sensor_msgs/msg/Image':
                msg = deserialize_message(data, Image)
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                buffers[topic].append((timestamp, img))

            elif msg_type == 'sensor_msgs/msg/JointState':
                msg = deserialize_message(data, JointState)
                arr = np.array(msg.position, dtype=np.float32)
                buffers[topic].append((timestamp, arr))

        return buffers

    def gripper_normalize_state(self, joint_arr: np.ndarray) -> np.ndarray:
        if not self.gripper_normalize or len(joint_arr) == 0:
            return joint_arr

        normalized = joint_arr.copy()
        gripper_value = normalized[-1]
        normalized_gripper = (gripper_value - self.gripper_min) / (self.gripper_max - self.gripper_min)
        normalized_gripper = np.clip(normalized_gripper, 0.0, 1.0)
        normalized[-1] = normalized_gripper
        
        return normalized

    def downsample_and_align(self, buffers: Dict[str, List[Tuple[int, Any]]]) -> Generator[Tuple[Dict[str, np.ndarray], np.ndarray], None, None]:
        """Align timeline and downsample, yielding frames one by one"""
        # Validate data exists
        for topic, buf in buffers.items():
            if not buf:
                print(f"[WARNING] Topic '{topic}' has no data!")

        image_topics = [t for t in buffers if isinstance(buffers[t][0][1], np.ndarray) and buffers[t][0][1].ndim == 3]

        state_topic_info = self.config.topic_map.get(self.state_feature_name, {})
        joint_topics = state_topic_info.get('topic', [])
        if isinstance(joint_topics, str):
            joint_topics = [joint_topics]

        if not image_topics or not joint_topics:
            raise RuntimeError(f"Missing image or joint topics: images={image_topics}, joints={joint_topics}")

        ts = {k: np.array([t for t, _ in v], dtype=np.int64) for k, v in buffers.items() if v}
        if not ts:
            raise RuntimeError("No timestamp data found.")

        t0 = max(ts[k][0] for k in ts)
        t1 = min(ts[k][-1] for k in ts)
        dt = int(1e9 / self.fps)  # nanosecond interval
        N = int((t1 - t0) // dt)
        timeline = t0 + np.arange(N, dtype=np.int64) * dt
        print(f"[INFO] Aligning timeline: {N} frames, time range [{t0}, {t1}], interval = {dt}ns")

        for t in timeline:
            frame_data = {}

            for topic in image_topics:
                idx = np.abs(ts[topic] - t).argmin()
                frame_data[topic] = buffers[topic][idx][1]

            joint_data = []
            for topic in joint_topics:
                if topic not in buffers or not buffers[topic]:
                    raise RuntimeError(f"Missing required joint topic: {topic}")
                idx = np.abs(ts[topic] - t).argmin()
                joint_arr = buffers[topic][idx][1]
                if self.gripper_normalize:
                    joint_arr = self.gripper_normalize_state(joint_arr)
                joint_data.append(joint_arr)

            if self.merge_joints:
                state = np.concatenate(joint_data)
            else:
                state = joint_data[0]

            yield frame_data, state

    def _padding_resize(self, frame: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Resize image with padding to fit target shape (height, width).
        """
        target_height, target_width = target_shape[:2]
        original_height, original_width = frame.shape[:2]

        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)

        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        padded = np.zeros((target_height, target_width, 3), dtype=resized.dtype)

        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        return padded

    def convert_episode(self, db3_path: str, task: str) -> int:
        print(f"🔍 Starting conversion: {db3_path}")
        start_time = time.time()

        topic_type_map = self.get_topic_types(db3_path)
        buffers = self.read_rosbag2(db3_path, topic_type_map)

        frame_generator = self.downsample_and_align(buffers)
        frame_count = 0
        prev_frame = None

        for frame_data, current_state in frame_generator:
            frame = {}

            for lr_key, ros_topic in self.image_topic_map.items():
                if ros_topic in frame_data:
                    img = frame_data[ros_topic]
                    target_shape = self.image_shapes.get(lr_key)

                    if target_shape is not None:
                        target_h, target_w = target_shape[0], target_shape[1]
                        if img.shape[0] != target_h or img.shape[1] != target_w:
                            img = self._padding_resize(img, (target_h, target_w))

                    frame[lr_key] = img
                else:
                    available = list(frame_data.keys())
                    raise ValueError(
                        f"Image topic not found: {ros_topic}\n"
                        f"Available topics: {available}\n"
                        f"Check image_topic_map configuration"
                    )

            frame[self.state_feature_name] = current_state
            frame['task'] = task

            if prev_frame is not None:
                if self.add_action and self.action_feature_name:
                    prev_frame[self.action_feature_name] = current_state
                self.dataset.add_frame(prev_frame)
                frame_count += 1

            prev_frame = frame

        if prev_frame is not None:
            # For the last frame, action is the same as state if action is enabled
            if self.add_action and self.action_feature_name:
                prev_frame[self.action_feature_name] = prev_frame[self.state_feature_name]
            self.dataset.add_frame(prev_frame)
            frame_count += 1

        self.dataset.save_episode()

        duration = time.time() - start_time
        print(f"✅ Conversion complete: {frame_count} frames | Time: {duration:.2f}s | Path: {db3_path}\n")
        return frame_count

    def convert_batch(self):
        """Convert all source directories specified in the config"""
        total_frames = 0
        
        for source in self.config.source_dirs:
            path = source["path"]
            task = source["task"]
            episodes = source.get("episodes", 30)
            
            path = Path(os.path.expanduser(path))
            print(f"🚀 Starting batch conversion: {path} | Task: '{task}' | Episodes: {episodes}")
            
            for ep in range(episodes):
                db3_path = path / f'episode_{ep}'
                if not db3_path.exists():
                    print(f"⚠️ Skipping missing episode: {db3_path}")
                    continue
                    
                try:
                    frames = self.convert_episode(str(db3_path), task)
                    total_frames += frames
                except Exception as e:
                    print(f"❌ Conversion failed {db3_path}: {str(e)}")
                    import traceback
                    traceback.print_exc()
        
        print(f"\n🎉 Conversion complete! Total frames: {total_frames}")
        print(f"📁 LeRobot dataset saved to: {HF_LEROBOT_HOME / self.repo_id}")
        return total_frames


if __name__ == '__main__':
    config = ConvertConfig(
        source_dirs=SOURCES,
        repo_id="realman/transfer_250910",
        robot_type='realman',
        features=FEATURES,
        fps=30,
        topic_map=MAP,
        add_action='action',
        gripper_normalize=False
    )
    
    converter = ROS2BagToLeRobotConverter(config)
    converter.convert_batch()