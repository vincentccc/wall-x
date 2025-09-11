# Training Configuration Guide

This document explains the key configuration parameters that can be modified for Wall-X training.

## Quick Start Checklist
1. **Update run.sh**: Set `code_dir` and `config_path` to your actual paths
2. **Configure GPUs**: Set `CUDA_VISIBLE_DEVICES` for your available GPUs
3. **Update config paths**: Replace all `/path/to/` placeholders in `config_qact.yml` with actual paths
4. **Configure robot**: Set `dof_config` and `agent_pos_config` for your robot
5. **Set dataset**: Choose appropriate `repo_id` for your dataset
6. **Adjust batch size**: Set `batch_size_per_gpu` based on GPU memory
7. **Run training**: Execute `bash ./workspace/lerobot_example/run.sh`

## Enable FAST tokenizer
To fine-tune using the FAST tokenizer, please download the repository and update the `action_tokenizer_path`. Make sure to set `use_fast_tokenizer` to `true`:
```bash
git clone https://huggingface.co/physical-intelligence/fast
```

## Required Paths (Must Modify)
```yaml
pretrained_wallx_path: "/path/to/wallx_model/"      # Path to pretrained wallx model
save_path: "/path/to/workspace/"                    # Path to save training outputs
use_fast_tokenizer: False                           # True: train FAST, False: train Flow
action_tokenizer_path: "/path/to/fast/"             # Must set if use_fast_tokenizer is True
```

## Training Parameters (Commonly Modified)

### Learning Rate Settings
- `learning_rate`: Initial learning rate (default: 0.00009)
- `min_lr`: Minimum learning rate for scheduler (default: 0.00005)
- `num_warmup_steps`: Number of warmup steps (default: 100)

### Batch Size and Memory
- `batch_size_per_gpu`: Batch size per GPU - adjust based on GPU memory
- `gradient_accumulation_steps`: Gradient accumulation steps
- `num_training_steps`: Total training steps
- `num_epoch`: Number of training epochs

## Robot Configuration (Modify for Your Robot)

### DOF Configuration
Modify `dof_config` to match your robot's action space:
- Add/remove action keys based on your robot's capabilities
- Ensure DOF numbers match your robot's action dimensions

### Agent Position Configuration
Keep `agent_pos_config` consistent with `dof_config`.

### Action Keys
- `obs_action_keys`: Actions used as observation context
- `predict_action_keys`: Actions to predict/control

## Data Configuration

### Dataset
- `repo_id`: LeRobot dataset identifier
- `train_test_split`: Training/validation split ratio (default: 0.95)
- `action_horizon`: Number of future actions to predict (default: 32)

### Image Settings
- `resolution`: Image resolution for different camera views
- `download_videos`: Whether to download video files (true/false)

## Resume Training (Optional)
- `resume.ckpt`: Path to checkpoint for resuming training
- `resume.load_ckpt_only`: Only load model weights, not optimizer state

## Performance Settings (Optional)
- `profile`: Enable PyTorch profiling (true/false)
- `padding_side`: Token padding side (left/right)
