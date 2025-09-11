import os
import yaml
import torch
import matplotlib.pyplot as plt
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.data.load_lerobot_dataset import load_test_dataset, get_data_configs


model_path = "path/to/model"
action_tokenizer_path = "path/to/action_tokenizer"
save_dir = "path/to/plot"
model = Qwen2_5_VLMoEForAction.from_pretrained(
    model_path, action_tokenizer_path=action_tokenizer_path
)
model.eval()
model = model.to("cuda")
model = model.bfloat16()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["data"]["model_type"] = config.get("model_type")

    return config


# get test dataloader
path = "path/to/config"
config = load_config(path)
dataload_config = get_data_configs(config["data"])
lerobot_config = dataload_config.get("lerobot_config", {})
dataset = load_test_dataset(config, lerobot_config, seed=42)
dataloader = dataset.get_dataloader()

total_frames = len(dataloader)

pred_horizon = 32
action_dim = 14
gt_traj = torch.zeros((total_frames, action_dim))
pred_traj = torch.zeros((total_frames, action_dim))

for idx, batch in enumerate(dataloader):
    if idx % pred_horizon == 0 and idx + pred_horizon < total_frames:
        batch = batch.to("cuda")
        with torch.no_grad():
            outputs = model(
                **batch,
                action_dim=action_dim,
                pred_horizon=pred_horizon,
                mode="predict",
                predict_mode="fast",
            )
        pred_traj[idx : idx + pred_horizon] = outputs["predict_action"].detach().cpu()

        # Denormalize ground truth actions
        gt_action_chunk = batch["action_chunk"][:, :, :action_dim]
        dof_mask = batch["dof_mask"].to(gt_action_chunk.dtype)
        denormalized_gt = model.action_preprocessor.normalizer_action.unnormalize_data(
            gt_action_chunk, ["x2_normal"], dof_mask
        )
        gt_traj[idx : idx + pred_horizon] = denormalized_gt.detach().cpu()


gt_traj_np = gt_traj.numpy()
pred_traj_np = pred_traj.numpy()

timesteps = gt_traj.shape[0]

fig, axs = plt.subplots(action_dim, 1, figsize=(15, 5 * action_dim), sharex=True)
fig.suptitle("Action Comparison for lerobot", fontsize=16)

for i in range(action_dim):
    axs[i].plot(range(timesteps), gt_traj_np[:, i], label="Ground Truth")
    axs[i].plot(range(timesteps), pred_traj_np[:, i], label="Prediction")
    axs[i].set_ylabel(f"Action Dim {i+1}")
    axs[i].legend()
    axs[i].grid(True)

axs[-1].set_xlabel("Timestep")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, "lerobot_comparison.png"))
plt.close()
