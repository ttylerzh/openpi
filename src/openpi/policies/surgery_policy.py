import einops
import dataclasses
import numpy as np
from openpi import transforms
from openpi.models import model as _model

'''inputs中是模型输入的字段格式字段，data读取的字段是config中映射得到的字段名'''

def _parse_image(image) -> np.ndarray:
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class Surgery_Inputs_2V(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType 

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation.state"], self.action_dim)
        top_left_image = _parse_image(data["observation.images.top_left"])
        top_right_image = _parse_image(data["observation.images.top_right"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_left_image,
                "left_wrist_0_rgb": np.zeros_like(top_left_image),
                "right_wrist_0_rgb": top_right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Surgery_Inputs_4V(transforms.DataTransformFn):
    action_dim: int
    model_type: _model.ModelType 

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["observation.state"], self.action_dim)
        top_image = _parse_image(data["observation.images.top_left"])
        left_image = _parse_image(data["observation.images.lwrist"])
        right_image = _parse_image(data["observation.images.rwrist"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": top_image,
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class Surgery_Outputs_abs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :20])}


@dataclasses.dataclass(frozen=True)
class Surgery_Outputs_re(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :14])}
