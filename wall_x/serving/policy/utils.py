from typing import Dict, List, Any, Optional
import logging
import numpy as np
from wall_x.data.utils import KEY_MAPPINGS, preprocesser_call
from qwen_vl_utils.vision_process import smart_resize
import torch
from PIL import Image
from transformers import BatchFeature
from collections import OrderedDict

logger = logging.getLogger(__name__)


def prepare_batch(
    obs: Dict,
    processor,
    camera_key: List[str],
    agent_pos_dim,
    action_dim,
    pred_horizon,
    fixed_action_dim,
    max_length,
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    predict_mode: str = "fast",
    device: str = "cuda",
) -> BatchFeature:
    """Prepare observation into model input format.

    Args:
        obs: Dictionary containing:
            - 'camera_key_0' : image 0
            - 'camera_key_1' : image 1
            ...
            - 'prompt': Text prompt
            - 'state': Robot state/proprioception
            - 'dataset_names': Dataset names

    Returns:
        BatchFeature object ready for model input
    """
    # Handle images - can be single image, list of images, or dict of images
    images = []
    images = [obs[key] for key in camera_key]
    # Convert numpy arrays to PIL Images
    processed_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Debug: Log the shape and dtype
            logger.debug(f"Image shape: {img.shape}, dtype: {img.dtype}")

            # Handle unexpected dimensions - squeeze if needed
            if img.ndim > 3:
                logger.warning(
                    f"Image has {img.ndim} dimensions, squeezing extra dimensions"
                )
                img = np.squeeze(img)

            # Verify shape is valid for PIL
            if img.ndim == 2:
                # Grayscale image
                pass
            elif img.ndim == 3:
                # Check if channel dimension is first or last
                if img.shape[0] == 3 or img.shape[0] == 1:
                    # Channels first, transpose to channels last
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[2] == 3 or img.shape[2] == 1:
                    # Already channels last
                    pass
                else:
                    raise ValueError(
                        f"Unexpected image shape: {img.shape}. Expected (H, W, C) or (C, H, W)"
                    )
            else:
                raise ValueError(
                    f"Invalid image dimensions: {img.ndim}. Expected 2 or 3 dimensions, got shape {img.shape}"
                )

            # Convert to PIL Image
            if img.dtype == np.uint8:
                img = Image.fromarray(img)
            else:
                img = Image.fromarray((img * 255).astype(np.uint8))
        processed_images.append(img)

    # Apply smart resize to images
    resized_images = process_images(
        processed_images, image_factor, min_pixels, max_pixels
    )

    # Handle text prompt - format with vision tokens
    instruction = obs["prompt"]
    formatted_text = format_text_with_vision_tokens(
        instruction, camera_key, predict_mode, pred_horizon
    )

    # Use processor to prepare inputs
    inputs = preprocesser_call(
        processor=processor,
        text=[formatted_text],
        images=[resized_images],
        videos=None,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    moe_token_types = inputs.input_ids == action_token_id
    inputs["moe_token_types"] = moe_token_types

    # Handle robot state/proprioception if available
    if "state" in obs:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        # Add batch dimension if needed
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        # Pad to 20 dimensions if needed (same as training)
        if state.shape[-1] < 20:
            padding = torch.zeros(state.shape[0], state.shape[1], 20 - state.shape[-1])
            state = torch.cat([state, padding], dim=-1)

        # Create mask for valid dimensions
        agent_pos_mask = torch.ones_like(state)
        if state.shape[-1] > agent_pos_dim:
            agent_pos_mask[:, :, agent_pos_dim:] = 0

        inputs["proprioception"] = state
        inputs["agent_pos_mask"] = agent_pos_mask

    # Add dataset name (required by model)
    inputs["dataset_names"] = obs["dataset_names"]

    # Move all tensors to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    dof_mask = torch.ones([state.shape[0], pred_horizon, fixed_action_dim])
    dof_mask[:, :, action_dim:] = 0

    inputs["dof_mask"] = dof_mask

    # Convert to BatchFeature to maintain consistency with training pipeline
    return BatchFeature(data=dict(inputs)).to(device)

@torch.no_grad()
def do_normalize(data, normalizer, dataset_name, fixed_dim=20):
    origin_dim = data.shape[-1]
    batch_size = data.shape[0]
    import os
    assert origin_dim <= fixed_dim, f"Data dimension {origin_dim} is greater than fixed dimension {fixed_dim}"
    # if isinstance(data, np.ndarray):
    #     data = torch.from_numpy(data).float()
    # elif not isinstance(data, torch.Tensor):
    #     data = torch.tensor(data, dtype=torch.float32)

    # Add batch dimension if needed
    if data.dim() == 1:
        data = data.unsqueeze(0)
    if data.dim() == 2:
        data = data.unsqueeze(1)  # [batch, 1, state_dim]

    # Pad to 20 dimensions if needed (same as training)
    if data.shape[-1] < fixed_dim:
        padding = torch.zeros(data.shape[0], data.shape[1], fixed_dim - data.shape[-1]).to(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        data = torch.cat([data, padding], dim=-1).to(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    # Create mask for valid dimensions
    # device = next(normalizer.parameters()).device

    mask = torch.ones_like(data).to(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    mask[:, :, origin_dim:] = 0
    
    data = normalizer.normalize_data(data, [dataset_name] * batch_size)

    return data, mask

def process_images(
    images: List[Image.Image], image_factor: int, min_pixels: int, max_pixels: int
) -> List[Image.Image]:
    """Process images with smart resize following the data loading pattern.

    Args:
        images: List of PIL Images

    Returns:
        List of resized PIL Images
    """
    resized_images = []
    for img_pil in images:
        current_width, current_height = img_pil.size

        # Apply smart scaling (Qwen logic)
        resized_height, resized_width = smart_resize(
            current_height,
            current_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        resized_img = img_pil.resize((resized_width, resized_height))
        resized_images.append(resized_img)

    return resized_images

def preprocess(dataset_name, data, image_factor, min_pixels, max_pixels, target_size, generate_subtask_ratio, pred_horizon, priority_order, cam_mapping, model_type):
    key_mapping = KEY_MAPPINGS.get(dataset_name, None)
    assert key_mapping is not None, f"{dataset_name} not found in KEY_MAPPINGS"
    camera_key = key_mapping["camera"].keys()

    h, w, resize_h, resize_w = None, None , None, None
    for key in camera_key:
        current_obs = data[key].clone().permute(1, 2, 0)
        image_inputs, h, w, resize_h, resize_w = vision_preprocess(current_obs, image_factor, min_pixels, max_pixels, target_size)
        image_inputs.append(image_inputs)
    
    agent_pos = data[key_mapping["state"]]
    action = data[key_mapping["action"]]
    frame_index = data["frame_index"]
    instruction_info = {"instruction": data["task"]}
    processed_text = process_text(instruction_info, frame_index, pred_horizon, h, w, resize_h, resize_w, model_type, priority_order, cam_mapping, generate_subtask_ratio)
    result = {
            "image_inputs": image_inputs,
            "text": processed_text,
            "action": action,
            "agent_pos": agent_pos,
            "frame_index": frame_index,
        }
    return result


def vision_preprocess(current_obs: torch.Tensor, image_factor: int, 
                      min_pixels: int, max_pixels: int, 
                      target_size):
    processed_frames = []
    img_pil = Image.fromarray((current_obs * 255).to(torch.uint8).cpu().numpy())
    orig_width, orig_height = img_pil.size
    # 2. Apply resolution constraints (if config is not -1)
    if target_size != -1:
        # Maintain aspect ratio logic
        if orig_width > orig_height:  # Landscape image
            new_width = target_size
            new_height = int(target_size * orig_height / orig_width)
        else:  # Portrait image
            new_height = target_size
            new_width = int(target_size * orig_width / orig_height)
        img_pil = img_pil.resize((new_width, new_height))

    # 3. Apply smart scaling (qwen logic)
    current_width, current_height = img_pil.size
    resized_height, resized_width = smart_resize(
        current_height,
        current_width,  
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    resized_img = img_pil.resize((resized_width, resized_height))
    processed_frames.append(resized_img)

    return processed_frames, orig_height, orig_width, resized_height, resized_width

def vision_preprocess_batched(current_obs: torch.Tensor, image_factor: int, 
                      min_pixels: int, max_pixels: int, 
                      target_size=-1):
    from qwen_vl_utils.vision_process import smart_resize

    # current_obs: [B, C, H, W]
    if current_obs.dtype != torch.uint8:
        current_obs = (current_obs * 255).to(torch.uint8)
    
    orig_height, orig_width = current_obs.shape[-2:]
    
    # 2. Apply resolution constraints (if config is not -1)
    if target_size != -1:
        # Maintain aspect ratio logic
        if orig_width > orig_height:  # Landscape image
            new_width = target_size
            new_height = int(target_size * orig_height / orig_width)
        else:  # Portrait image
            new_height = target_size
            new_width = int(target_size * orig_width / orig_height)
        
        # torch resize: input [C, H, W], output [C, new_H, new_W]
        current_obs = torch.nn.functional.interpolate(
            current_obs.float(),  
            size=(new_height, new_width),
            mode='bicubic',
            align_corners=False
        )

    # 3. Apply smart scaling (qwen logic)
    current_height, current_width = current_obs.shape[-2:]
    resized_height, resized_width = smart_resize(
        current_height,
        current_width,  
        factor=image_factor,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    
    current_obs = torch.nn.functional.interpolate(
        current_obs.float(),
        size=(resized_height, resized_width),
        mode='bicubic',
        align_corners=False
    )
    return current_obs, orig_height, orig_width, resized_height, resized_width



def process_text(instruction_info: Dict[str, Any], 
                 frame_index: int,
                 pred_horizon: int,
                 orig_height: int,
                 orig_width: int,
                 resized_height: int,
                 resized_width: int,
                 model_type: str,
                 priority_order: Optional[OrderedDict] = None, 
                 cam_mapping: Optional[Dict[str, str]] = None, 
                 generate_subtask_ratio: float = 0.0) -> str:
    from wall_x.data.utils import get_wallx_normal_text, process_grounding_points
    complete_text, generate_subtask = get_wallx_normal_text(
        instruction_info,
        pred_horizon,
        frame_index,
        priority_order,
        cam_mapping,
        generate_subtask_ratio=generate_subtask_ratio,
    )
    text = process_grounding_points(
            complete_text, orig_height, orig_width, resized_height, resized_width, model_type
    )
    
    return text


def data_preprocess(batch, max_length, normalizer_action, normalizer_propri, dataset_names, processor, action_tokenizer=None):
    additional_inputs = {}
    for key in batch[0].keys():
        if key == "agent_pos":
            agent_pos = torch.stack([item["agent_pos"] for item in batch])
            if agent_pos.dim() == 2:
                agent_pos = agent_pos.unsqueeze(1)
            agent_pos_mask = (~torch.isnan(agent_pos)).float()
            agent_pos.nan_to_num_(nan=0.0)
            if agent_pos.shape[-1] != 20:
                agent_pos = torch.cat(
                    [
                        agent_pos,
                        torch.zeros(
                            agent_pos.shape[0],
                            agent_pos.shape[1],
                            20 - agent_pos.shape[-1],
                        ),
                    ],
                    dim=-1,
                )
                agent_pos_mask = torch.cat(
                    [
                        agent_pos_mask,
                        torch.zeros(
                            agent_pos_mask.shape[0],
                            agent_pos_mask.shape[1],
                            20 - agent_pos_mask.shape[-1],
                        ),
                    ],
                    dim=-1,
                )
            agent_pos = normalizer_propri.normalize_data(agent_pos, dataset_names, agent_pos_mask)
            additional_inputs["proprioception"] = agent_pos
            additional_inputs["agent_pos_mask"] = agent_pos_mask
        elif key == "action":
            action = torch.stack([item["action"] for item in batch])
            if action.dim() == 2:
                action = action.unsqueeze(1)
            dof_mask = (~torch.isnan(action)).float()
            action.nan_to_num_(nan=0.0)
            if action.shape[-1] != 20:
                action = torch.cat(
                    [
                        action,
                        torch.zeros(
                            action.shape[0], action.shape[1], 20 - action.shape[-1]
                        ),
                    ],
                    dim=-1,
                )
                dof_mask = torch.cat(
                    [
                        dof_mask,
                        torch.zeros(
                            dof_mask.shape[0],
                            dof_mask.shape[1],
                            20 - dof_mask.shape[-1],
                        ),
                    ],
                    dim=-1,
                )
            action = normalizer_action.normalize_data(action, dataset_names, dof_mask)
            additional_inputs["action_chunk"] = action
            additional_inputs["dof_mask"] = dof_mask
        elif key == "image_inputs":
            additional_inputs["image_inputs"] = [
                item["image_inputs"] for item in batch
            ]
        elif key == "text":
            additional_inputs["text"] = [item["text"] for item in batch]
        elif key == "frame_index":
            additional_inputs["frame_index"] = torch.stack(
                [item["frame_index"] for item in batch]
            )
        else:
            raise NotImplementedError(
                f"{key} input not implemented in preprocesser"
            )

        from wall_x.data.utils import replace_action_token
        additional_inputs["text"] = replace_action_token(
            additional_inputs["text"],
            additional_inputs["action_chunk"],
            action_tokenizer,
            dataset_names,
            additional_inputs["dof_mask"],
        )

        inputs = preprocesser_call(
            processor=processor,
            text=additional_inputs.pop("text"),
            images=additional_inputs.pop("image_inputs"),
            videos=None,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )

        action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")

        # Gating token types
        additional_inputs["moe_token_types"] = inputs.input_ids == action_token_id

        inputs.update(additional_inputs)

        inputs["dataset_names"] = dataset_names
        
        return inputs

def format_text_with_vision_tokens(
    instruction: str,
    camera_key: List[str],
    predict_mode: str = "fast",
    pred_horizon: int = 32,
) -> str:
    """Format text prompt with vision tokens for the model.

    Args:
        instruction: Task instruction text
        camera_key: List of camera names

    Returns:
        Formatted text with special tokens
    """
    # Special tokens for formatting
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    # action_fast_symbol = "<|action_fast|>"

    # Camera name mapping
    camera_name_mapping = {
        "front_view": "front view",
        "face_view": "front view",
        "left_wrist_view": "left wrist view",
        "right_wrist_view": "right wrist view",
        "top_view": "top view",
        "wall_view": "wall view",
    }
    pred_horizon = 32

    # System prologue
    prologue = (
        f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    )

    # User request with observation
    user_request = f"{role_start_symbol}user\nObservation:"
    if camera_key:
        for cam_name in camera_key:
            view_name = camera_name_mapping.get(cam_name, cam_name)
            user_request += f" {view_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    text_prompt = (
        f"\nPredict the next action in robot action.\nProprioception: {propri_symbol}\n"
    )
    user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
    assistant_output = f"{role_start_symbol}assistant\n"
    if predict_mode == "diffusion":
        assistant_output += f"{action_symbol * pred_horizon}"
    complete_text = prologue + user_message + assistant_output

    return complete_text
