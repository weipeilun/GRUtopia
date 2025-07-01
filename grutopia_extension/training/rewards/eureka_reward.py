import torch
import numpy as np
from grutopia.core.constants import (
    OBSERVATION_TRACKING_COMMAND_KEY,
    OBSERVATION_BASE_LIN_VEL_KEY,
    OBSERVATION_BASE_ANG_VEL_KEY,
    OBSERVATION_GRAVITY_KEY,
    OBSERVATION_RELATIVE_BASE_HEIGHT_KEY,
    OBSERVATION_LEFT_ANKLE_COLLISION_KEY,
    OBSERVATION_RIGHT_ANKLE_COLLISION_KEY
)
from grutopia_extension.training.utils import features as features_util


class EurekaReward():
    def __init__(self, env):
        self.env = env

        self.target_gravity_vec = torch.tensor([0, 0, -1])

        # Constants
        self.target_distance_between_ankle = 0.4  # desired distance between ankles
        self.target_step_height = 0.2  # desired mark time ankle height
        self.target_z_position = 0.98  # desired height of the torso
        self.orientation_penalty_factor = 0.1  # increased to add more significance to orientation
        self.small_value_scale = 0.1
        self.step_height_tolerance = 0.2  # Tolerance for step height variation
        self.step_width_tolerance = 0.4  # Tolerance for step width variation
        self.ground_contact_threshold = 0.01  # Threshold for considering foot on ground

        # step frequency
        env_timestamp = env._runner._world.current_time
        self.target_step_interval = 0.8
        self.step_interval_exceed_threshold = self.target_step_interval * 1.5
        self.last_left_foot_on_ground_time = torch.ones(self.env.episode_length_buf.shape, dtype=torch.float, device='cpu') * env_timestamp
        self.last_right_foot_on_ground_time = torch.ones(self.env.episode_length_buf.shape, dtype=torch.float, device='cpu') * env_timestamp
        self.left_foot_step_interval = torch.zeros(self.env.episode_length_buf.shape, dtype=torch.float, device='cpu')
        self.right_foot_step_interval = torch.zeros(self.env.episode_length_buf.shape, dtype=torch.float, device='cpu')
        self.step_interval_min_threshold = self.env._runner._world._initial_physics_dt * self.env.config.env_set.robot_per_inference_frames * 3

    def load_env(self, env):
        self.env = env

    def reset(self, env_id):
        env_timestamp = self.env._runner._world.current_time
        self.last_left_foot_on_ground_time[env_id] = env_timestamp
        self.last_right_foot_on_ground_time[env_id] = env_timestamp
        self.left_foot_step_interval[env_id] = 0.
        self.right_foot_step_interval[env_id] = 0.

    def compute_reward(self, obs_dict):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.

        obs_dict_by_key = env.merge_observation_dict_by_key(obs_dict)
        obs_dict_by_key = features_util.observation_dict_list_to_tensor(obs_dict_by_key)
        # Calculate different components of the reward
        obs_tracking_command = obs_dict_by_key[OBSERVATION_TRACKING_COMMAND_KEY]
        obs_base_lin_val = obs_dict_by_key[OBSERVATION_BASE_LIN_VEL_KEY]
        obs_base_ang_val = obs_dict_by_key[OBSERVATION_BASE_ANG_VEL_KEY]
        obs_gravity = obs_dict_by_key[OBSERVATION_GRAVITY_KEY]
        obs_base_height = obs_dict_by_key[OBSERVATION_RELATIVE_BASE_HEIGHT_KEY]
        # obs_left_ankle_world_pos = obs_dict_by_key[OBSERVATION_LEFT_ANKLE_WORLD_POS_KEY]
        # obs_right_ankle_world_pos = obs_dict_by_key[OBSERVATION_RIGHT_ANKLE_WORLD_POS_KEY]
        # obs_left_ankle_pos = obs_dict_by_key[OBSERVATION_LEFT_ANKLE_POS_KEY]
        # obs_right_ankle_pos = obs_dict_by_key[OBSERVATION_RIGHT_ANKLE_POS_KEY]
        obs_left_ankle_collision = obs_dict_by_key[OBSERVATION_LEFT_ANKLE_COLLISION_KEY].squeeze(1)
        obs_right_ankle_collision = obs_dict_by_key[OBSERVATION_RIGHT_ANKLE_COLLISION_KEY].squeeze(1)

        env_timestamp = env._runner._world.current_time

        # Linear velocity x reward component
        lin_velocity_error_x = torch.abs(obs_base_lin_val[:, 0] - obs_tracking_command[:, 0])
        lin_velocity_reward_x = torch.exp(-lin_velocity_error_x**2 / 10.0) * 2.0  # Scale reduced

        # Linear velocity y reward component
        lin_velocity_error_y = torch.abs(obs_base_lin_val[:, 1] - obs_tracking_command[:, 1])
        lin_velocity_reward_y = torch.exp(-lin_velocity_error_y**2 / 10.0) * 2.0  # Scale reduced

        # Angular velocity reward component
        # set base angular velocity to 1 / 4 round per second
        ang_velocity_error = torch.abs(obs_base_ang_val[:, 0] * 2 / torch.pi - obs_tracking_command[:, 2])
        ang_velocity_reward = torch.exp(-ang_velocity_error**2 / 20.0) * 2.0  # Scale reduced

        # # Ankles should always be in a fixed width.
        # distance_between_ankle_error = torch.abs(torch.abs(obs_left_ankle_pos[:, 1] - obs_right_ankle_pos[:, 1]) - self.target_distance_between_ankle)
        # distance_between_ankle_reward = torch.exp(-distance_between_ankle_error**2 / 0.2) * 1  # Scale reduced
        # #
        # # Add a small step.
        # step_error = torch.abs(torch.abs(obs_left_ankle_world_pos[:, 2] - obs_right_ankle_world_pos[:, 2]) - self.target_step_height)
        # step_reward = torch.exp(-step_error**2 / 0.1) * 1.  # Scale reduced

        # # target step feet height.
        # left_step_foot_height_error = torch.abs(obs_left_ankle_world_pos[:, 2] - self.target_step_height)
        # left_step_foot_height_reward = torch.exp(-left_step_foot_height_error ** 2 / 0.2) * 1.  # Scale reduced
        # right_step_foot_height_error = torch.abs(obs_right_ankle_world_pos[:, 2] - self.target_step_height)
        # right_step_foot_height_reward = torch.exp(-right_step_foot_height_error ** 2 / 0.2) * 1.  # Scale reduced

        # # Always has foot on ground.
        # both_feet_off_ground = torch.logical_and(torch.logical_not(obs_left_ankle_collision), torch.logical_not(obs_right_ankle_collision))
        # has_foot_on_ground_reward = torch.exp(-both_feet_off_ground.to(torch.float)) * 2.  # Scale reduced

        # Add step frequency.
        left_foot_on_ground_interval = env_timestamp - self.last_left_foot_on_ground_time
        is_left_foot_finish_step = torch.logical_and(obs_left_ankle_collision, left_foot_on_ground_interval >= self.step_interval_min_threshold)
        self.left_foot_step_interval = torch.where(is_left_foot_finish_step, left_foot_on_ground_interval, self.left_foot_step_interval)
        self.last_left_foot_on_ground_time = torch.where(obs_left_ankle_collision, env_timestamp, self.last_left_foot_on_ground_time)
        left_foot_step_interval = torch.where(left_foot_on_ground_interval > self.step_interval_exceed_threshold, left_foot_on_ground_interval, self.left_foot_step_interval)

        right_foot_on_ground_interval = env_timestamp - self.last_right_foot_on_ground_time
        is_right_foot_finish_step = torch.logical_and(obs_right_ankle_collision, right_foot_on_ground_interval > self.step_interval_min_threshold)
        self.right_foot_step_interval = torch.where(is_right_foot_finish_step, right_foot_on_ground_interval, self.right_foot_step_interval)
        self.last_right_foot_on_ground_time = torch.where(obs_right_ankle_collision, env_timestamp, self.last_right_foot_on_ground_time)
        right_foot_step_interval = torch.where(right_foot_on_ground_interval > self.step_interval_exceed_threshold, right_foot_on_ground_interval, self.right_foot_step_interval)

        step_frequency_error_left = torch.abs(left_foot_step_interval - self.target_step_interval)
        step_frequency_reward_left = torch.exp(-step_frequency_error_left**2 / 1.) * 2.  # Scale reduced
        step_frequency_error_right = torch.abs(right_foot_step_interval - self.target_step_interval)
        step_frequency_reward_right = torch.exp(-step_frequency_error_right**2 / 1.) * 2.  # Scale reduced
        step_frequency_reward = (step_frequency_reward_left + step_frequency_reward_right) / 2.0

        # Height reward component
        height_error = torch.abs(obs_base_height.squeeze(1) - self.target_z_position)
        height_reward = torch.exp(-height_error**2 / 0.2) * 2  # Scale reduced

        # Orientation reward component
        orientation_error = torch.norm(obs_gravity - self.target_gravity_vec, dim=-1)
        orientation_reward = torch.exp(-np.power(orientation_error, 1 / 3)) * 2  # Scale reduced

        # # Smoothness reward component (penalize large action differences)
        # action_diff = torch.sum(torch.square(env.actions - env.last_actions), dim=-1# smoothness_penalty = torch.exp(-action_diff / 1.0) * 2.0  # Keep a higher scale for emphasis

        # # Joint limit avoidance reward (penalize when dofs are near their limits)
        # dof_pos_normalized = (env.dof_pos - env.dof_pos_limits[:, 0]) / (env.dof_pos_limits[:, 1] - env.dof_pos_limits[:, 0])
        # dof_limit_penalty = self.small_value_scale * torch.sum(torch.exp(-0.5 * (dof_pos_normalized * (1 - dof_pos_normalized))), dim=-1) * 0.8  # Scale reduced

        # # Modify step height reward
        # left_step_height = obs_left_ankle_world_pos[:, 2]
        # right_step_height = obs_right_ankle_world_pos[:, 2]
        # step_height_diff = torch.abs(left_step_height - right_step_height)
        # step_height_stability_reward = torch.exp(-step_height_diff**2 / (2 * self.step_height_tolerance**2)) * 2.0

        # # Add step width stability reward
        # step_width = torch.abs(obs_left_ankle_world_pos[:, 1] - obs_right_ankle_world_pos[:, 1])
        # step_width_error = torch.abs(step_width - self.target_distance_between_ankle)
        # step_width_stability_reward = torch.exp(-step_width_error**2 / (2 * self.step_width_tolerance**2)) * 2.0

        # # Improve ground contact reward
        # left_foot_contact = obs_left_ankle_collision.float()
        # right_foot_contact = obs_right_ankle_collision.float()
        # alternating_contact_reward = torch.abs(left_foot_contact - right_foot_contact) * 2.0
        # Remove undefined rewards from total_reward calculation
        total_reward = (
            lin_velocity_reward_x + lin_velocity_reward_y + ang_velocity_reward +
            step_frequency_reward + height_reward + orientation_reward
        )

        # Update individual_rewards dictionary to include only defined rewards
        individual_rewards = {
            "rewards.lin_velocity_reward_x": lin_velocity_reward_x,
            "rewards.lin_velocity_reward_y": lin_velocity_reward_y,
            "rewards.ang_velocity_reward": ang_velocity_reward,
            "rewards.step_frequency_reward": step_frequency_reward,
            "rewards.height_reward": height_reward,
            "rewards.orientation_reward": orientation_reward,
        }

        return total_reward, individual_rewards

    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

    def gravity_to_quat(self, gravity):
        # Convert gravity vector to quaternion
        gravity_normalized = gravity / torch.norm(gravity, dim=-1, keepdim=True)

        # Calculate the rotation axis and angle
        up_vector = torch.tensor([0.0, 0.0, -1.0], device=gravity.device).expand_as(gravity_normalized)
        rotation_axis = torch.cross(up_vector, gravity_normalized)
        rotation_axis_norm = torch.norm(rotation_axis, dim=-1, keepdim=True)

        # Handle the case where gravity is already pointing up
        rotation_axis = torch.where(rotation_axis_norm > 1e-6, rotation_axis / rotation_axis_norm, torch.tensor([1.0, 0.0, 0.0], device=gravity.device))

        cos_angle = torch.clamp(torch.einsum('bi,bi->b', up_vector, gravity_normalized), -1.0, 1.0)
        angle = torch.acos(cos_angle)

        # Convert axis-angle to quaternion
        qx = rotation_axis[:, 0] * torch.sin(angle / 2)
        qy = rotation_axis[:, 1] * torch.sin(angle / 2)
        qz = rotation_axis[:, 2] * torch.sin(angle / 2)
        qw = torch.cos(angle / 2)

        return torch.stack((qx, qy, qz, qw), dim=-1)
    
    def quat_conjugate(self, q):
        # Compute the conjugate of a quaternion
        # q = [x, y, z, w]
        return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)

    def quat_multiply(self, q1, q2):
        # Multiply two quaternions
        # q1, q2 = [x1, y1, z1, w1], [x2, y2, z2, w2]
        x1, y1, z1, w1 = torch.unbind(q1, dim=-1)
        x2, y2, z2, w2 = torch.unbind(q2, dim=-1)
        
        return torch.stack([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        ], dim=-1)
