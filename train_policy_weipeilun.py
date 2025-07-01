"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from grutopia.core.lerobot_dataset.lerobot_dataset import LeRobotDataset
from grutopia.core.config import SimulatorConfig
from grutopia.core.constants import *
from grutopia_extension.models.simple_policy import SimplePolicy

file_path = './GRUtopia/demo/configs/h1_locomotion.yaml'
sim_config = SimulatorConfig(file_path)

repo_id = "sim_robot"
root = 'GRUtopia/output'

# Create a directory to store the training checkpoint.
training_steps = 50000
output_directory = Path(f"GRUtopia/output/sim_robot_model_{training_steps}")
output_directory.mkdir(parents=True, exist_ok=True)

# Number of offline training steps (we'll only do offline training for this example.)
# Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
device = torch.device("cuda")
log_freq = 250

# Set up the dataset.
physics_fps = int(1 / eval(sim_config.config.simulator.physics_dt))
robot_fps = int(physics_fps / 4)
history_feature_len = 5
delta_timestamps = {
    OBSERVATION_TRACKING_COMMAND_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    OBSERVATION_GRAVITY_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    OBSERVATION_BASE_LIN_VEL_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    OBSERVATION_BASE_ANG_VEL_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    OBSERVATION_JOINT_POS_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    OBSERVATION_JOINT_VEL_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    OBSERVATION_RELATIVE_BASE_HEIGHT_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len)])),
    JOINT_ACTION_KEY: list(reversed([-i / robot_fps for i in range(history_feature_len + 1)])),
}
dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)

policy = SimplePolicy(path=None)
policy.train(device)

optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)

# Create dataloader for offline training.
dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=20,
    batch_size=16,
    shuffle=True,
    pin_memory=device != torch.device("cpu"),
    drop_last=True,
)

# Run training loop.
step = 0
done = False
while not done:
    for batch in dataloader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        feature_list = []
        feature_list.append(batch['observation.tracking_command'].squeeze())
        feature_list.append(batch['observation.gravity'].squeeze())
        feature_list.append(batch['observation.base_lin_vel'].squeeze())
        feature_list.append(batch['observation.base_ang_vel'].squeeze())
        feature_list.append(batch['observation.joint_pos'].squeeze())
        feature_list.append(batch['observation.joint_vel'].squeeze())
        feature_list.append(batch['observation.relative_base_height'].squeeze(1))
        feature_list.append(batch['action'][:, 0:5, :])
        feature_tensor = torch.concatenate(feature_list, dim=-1)
        feature_reshape = feature_tensor.reshape((-1, feature_tensor.shape[1] * feature_tensor.shape[2]))
        feature_input = feature_reshape.to(device, non_blocking=True)

        feature_target = batch['action'][:, -1, :].to(device, non_blocking=True)

        output_dict = policy.forward(feature_input, target=feature_target)
        loss = output_dict["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            print(f"step: {step} loss: {loss.item():.3f}")
        step += 1
        if step >= training_steps:
            done = True
            break

# Save a policy checkpoint.
policy.save_pretrained(output_directory)
