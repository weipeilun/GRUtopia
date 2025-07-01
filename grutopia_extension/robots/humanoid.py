import os
import cv2
import time
from typing import Dict

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationSubset
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.robots.robot import Robot as IsaacRobot
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction, ArticulationActions
from omni.isaac.sensor import Camera
from pxr import PhysxSchema, UsdPhysics

import grutopia.core.util.string as string_utils
import grutopia.core.util.math as math_utils
from grutopia.core.constants import (
    OBSERVATION_IMAGE_LEFT_KEY,
    OBSERVATION_IMAGE_RIGHT_KEY,
    OBSERVATION_GRAVITY_KEY,
    OBSERVATION_BASE_LIN_VEL_KEY,
    OBSERVATION_BASE_ANG_VEL_KEY,
    OBSERVATION_JOINT_POS_KEY,
    OBSERVATION_JOINT_VEL_KEY,
    OBSERVATION_RELATIVE_BASE_HEIGHT_KEY,
    OBSERVATION_LEFT_ANKLE_WORLD_POS_KEY,
    OBSERVATION_RIGHT_ANKLE_WORLD_POS_KEY,
    OBSERVATION_LEFT_ANKLE_POS_KEY,
    OBSERVATION_RIGHT_ANKLE_POS_KEY,
    OBSERVATION_LEFT_ANKLE_COLLISION_KEY,
    OBSERVATION_RIGHT_ANKLE_COLLISION_KEY,
    OBSERVATION_RIGHT_ANKLE_POS_KEY,
    OBSERVATION_LEFT_ANKLE_COLLISION_KEY,
    OBSERVATION_RIGHT_ANKLE_COLLISION_KEY,
)
from grutopia.actuators import ActuatorBase, ActuatorBaseCfg, DCMotorCfg
from grutopia.core.config.robot import RobotUserConfig as Config
from grutopia.core.robot.robot import BaseRobot
from grutopia.core.robot.robot_model import RobotModel
from grutopia.core.util import log


class Humanoid(IsaacRobot):

    actuators_cfg = {
        'base_legs':
        DCMotorCfg(
            joint_names_expr=['.*'],
            effort_limit={
                '.*hip.*': 200,
                '.*knee.*': 300,
                '.*ankle.*': 40,
                'torso_joint': 200,
                '.*shoulder_pitch.*': 40,
                '.*shoulder_roll.*': 40,
                '.*shoulder_yaw.*': 18,
                '.*elbow.*': 18
            },
            saturation_effort=400.0,
            velocity_limit=1000.0,
            stiffness={
                '.*hip.*': 200,
                '.*knee.*': 300,
                '.*ankle.*': 40,
                'torso_joint': 300,
                '.*shoulder.*': 100,
                '.*elbow.*': 100
            },
            damping={
                '.*hip.*': 5,
                '.*knee.*': 6,
                '.*ankle.*': 2,
                'torso_joint': 6,
                '.*shoulder.*': 2,
                '.*elbow.*': 2
            },
            friction=0.0,
            armature=0.0,
        ),
    }

    def __init__(self,
                 prim_path: str,
                 usd_path: str,
                 name: str,
                 position: np.ndarray = None,
                 orientation: np.ndarray = None,
                 scale: np.ndarray = None):
        add_reference_to_stage(prim_path=prim_path, usd_path=os.path.abspath(usd_path))
        super().__init__(prim_path=prim_path, name=name, position=position, orientation=orientation, scale=scale)
        self.actuators: Dict[str, ActuatorBase]

    def set_gains(self, gains):
        """[summary]

        Args:
            kps (Optional[np.ndarray], optional): [description]. Defaults to None.
            kds (Optional[np.ndarray], optional): [description]. Defaults to None.

        Raises:
            Exception: [description]
        """
        num_leg_joints = 19
        kps = np.array([0.] * num_leg_joints)
        kds = np.array([0.] * num_leg_joints)

        if kps is not None:
            kps = self._articulation_view._backend_utils.expand_dims(kps, 0)
        if kds is not None:
            kds = self._articulation_view._backend_utils.expand_dims(kds, 0)
        self._articulation_view.set_gains(kps=kps, kds=kds, save_to_usd=False)
        # VERY important!!! additional physics parameter
        self._articulation_view.set_solver_position_iteration_counts(
            self._articulation_view._backend_utils.expand_dims(4, 0))
        self._articulation_view.set_solver_velocity_iteration_counts(
            self._articulation_view._backend_utils.expand_dims(0, 0))
        self._articulation_view.set_enabled_self_collisions(self._articulation_view._backend_utils.expand_dims(True, 0))

    def _process_actuators_cfg(self):
        self.actuators = dict.fromkeys(Humanoid.actuators_cfg.keys())
        for actuator_name, actuator_cfg in Humanoid.actuators_cfg.items():
            # type annotation for type checkersc
            actuator_cfg: ActuatorBaseCfg
            # create actuator group
            joint_ids, joint_names = self.find_joints(actuator_cfg.joint_names_expr)
            stiffness, damping = self._articulation_view.get_gains()
            actuator: ActuatorBase = actuator_cfg.class_type(
                cfg=actuator_cfg,
                joint_names=joint_names,
                joint_ids=joint_ids,
                num_envs=1,
                device='cpu',
                stiffness=self._articulation_view.get_gains()[0][0],
                damping=self._articulation_view.get_gains()[1][0],
                armature=torch.tensor(self._articulation_view.get_armatures()),
                friction=torch.tensor(self._articulation_view.get_friction_coefficients()),
                effort_limit=torch.tensor(self._articulation_view._physics_view.get_dof_max_forces()),
                velocity_limit=torch.tensor(self._articulation_view._physics_view.get_dof_max_velocities()),
            )
            # log information on actuator groups
            self.actuators[actuator_name] = actuator

    def apply_actuator_model(self, control_action: ArticulationAction, controller_name: str,
                             joint_set: ArticulationSubset):
        name = 'base_legs'
        actuator = self.actuators[name]

        if isinstance(control_action.joint_positions, torch.Tensor):
            control_joint_pos = control_action.joint_positions.clone().detach()
        else:
            control_joint_pos = torch.tensor(control_action.joint_positions, dtype=torch.float32)
        control_actions = ArticulationActions(
            joint_positions=control_joint_pos,
            joint_velocities=torch.zeros_like(control_joint_pos),
            joint_efforts=torch.zeros_like(control_joint_pos),
            joint_indices=actuator.joint_indices,
        )

        joint_pos = torch.tensor(self.get_joint_positions(), dtype=torch.float32)
        joint_vel = torch.tensor(self.get_joint_velocities(), dtype=torch.float32)
        control_actions = actuator.compute(
            control_actions,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
        )
        if control_actions.joint_positions is not None:
            joint_set.set_joint_positions(control_actions.joint_positions)
        if control_actions.joint_velocities is not None:
            joint_set.set_joint_velocities(control_actions.joint_velocities)
        if control_actions.joint_efforts is not None:
            self._articulation_view._physics_view.set_dof_actuation_forces(control_actions.joint_efforts,
                                                                           torch.tensor([0]))

    def find_joints(self, name_keys, joint_subset=None):
        """Find joints in the articulation based on the name keys.

        Please see the :func:`omni.isaac.orbit.utils.string.resolve_matching_names` function for more information
        on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the joint names.
            joint_subset: A subset of joints to search for. Defaults to None, which means all joints
                in the articulation are searched.

        Returns:
            A tuple of lists containing the joint indices and names.
        """
        if joint_subset is None:
            joint_subset = self._articulation_view.dof_names
        # find joints
        return string_utils.resolve_matching_names(name_keys, joint_subset)


@BaseRobot.register('HumanoidRobot')
class HumanoidRobot(BaseRobot):

    def __init__(self, config: Config, robot_model: RobotModel, scene: Scene):
        super().__init__(config, robot_model, scene)
        self._sensor_config = robot_model.sensors
        self._gains = robot_model.gains
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'humanoid {config.name}: position    : ' + str(self._start_position))
        log.debug(f'humanoid {config.name}: orientation : ' + str(self._start_orientation))

        self.usd_path = robot_model.usd_path
        if self.usd_path.startswith('/Isaac'):
            self.usd_path = get_assets_root_path() + self.usd_path

        log.debug(f'humanoid {config.name}: self.usd_path         : ' + str(self.usd_path))
        log.debug(f'humanoid {config.name}: config.prim_path : ' + str(config.prim_path))
        self.isaac_robot = Humanoid(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=self.usd_path,
        )

        if robot_model.joint_names is not None:
            self.joint_subset = ArticulationSubset(self.isaac_robot, robot_model.joint_names)

        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
            self.isaac_robot.set_local_scale(self._robot_scale)
        # self.isaac_robot_world_velocity = self.isaac_robot.get_world_velocity()
        self.isaac_robot_world_pose = self.isaac_robot.get_world_pose()

        self._robot_ik_base = None

        self._robot_base = RigidPrim(prim_path=config.prim_path + '/pelvis', name=config.name + '_base')
        self._robot_right_ankle = RigidPrim(prim_path=config.prim_path + '/right_ankle_link', name=config.name + 'right_ankle')
        self._robot_left_ankle = RigidPrim(prim_path=config.prim_path + '/left_ankle_link', name=config.name + 'left_ankle')

        # detect collision with ground
        left_ankle_prim = scene.stage.GetPrimAtPath(config.prim_path + '/left_ankle_link')
        UsdPhysics.CollisionAPI.Apply(left_ankle_prim)
        PhysxSchema.PhysxTriggerAPI.Apply(left_ankle_prim)
        self.left_ankle_collision_api = PhysxSchema.PhysxTriggerStateAPI.Apply(left_ankle_prim)
        right_ankle_prim = scene.stage.GetPrimAtPath(config.prim_path + '/right_ankle_link')
        UsdPhysics.CollisionAPI.Apply(right_ankle_prim)
        PhysxSchema.PhysxTriggerAPI.Apply(right_ankle_prim)
        self.right_ankle_collision_api = PhysxSchema.PhysxTriggerStateAPI.Apply(right_ankle_prim)
        self.ground_contact_prim_path = '/World/defaultGroundPlane/GroundPlane/CollisionPlane'

        self.sensor_to_key_dict = {
            'left_eye_camera': OBSERVATION_IMAGE_LEFT_KEY,
            'right_eye_camera': OBSERVATION_IMAGE_RIGHT_KEY,
        }
        self.sky_camera_key = 'sky_camera'

        self.default_frame_dict = dict()

    def post_reset(self):
        super().post_reset()
        self.isaac_robot._process_actuators_cfg()
        if self._gains is not None:
            self.isaac_robot.set_gains(self._gains)

    def reset(self):
        # self.isaac_robot = Humanoid(
        #     prim_path=self.user_config.prim_path,
        #     name=self.user_config.name,
        #     position=self._start_position,
        #     orientation=self._start_orientation,
        #     usd_path=self.usd_path,
        # )
        # if self.user_config.scale is not None:
        #     self._robot_scale = np.array(self.user_config.scale)
        #     self.isaac_robot.set_local_scale(self._robot_scale)
        self.isaac_robot._articulation_view.post_reset()
        self.post_reset()
        self.isaac_robot.post_reset()
        self.isaac_robot.set_world_velocity(np.zeros(6, dtype=np.float32))
        # self.isaac_robot.set_world_velocity(self.isaac_robot_world_velocity)
        self.isaac_robot.set_world_pose(*self.isaac_robot_world_pose)
        # self.joint_subset.set_joint_positions(np.zeros(len(self.joint_subset.joint_names), dtype=np.float32))

    def get_ankle_height(self):
        return np.min([self._robot_right_ankle.get_world_pose()[0][2], self._robot_left_ankle.get_world_pose()[0][2]])

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_base(self) -> RigidPrim:
        return self._robot_base

    def get_robot_ik_base(self):
        return self._robot_ik_base

    def get_world_pose(self):
        return self._robot_base.get_world_pose()

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warn(f'unknown controller {controller_name} in action')
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            self.isaac_robot.apply_actuator_model(control, controller_name, self.joint_subset)

    def is_contact_with_target(self, contact_list, target_prim_path):
        for contact_path in contact_list:
            if contact_path.pathString == target_prim_path:
                return True
        return False

    def get_obs(self, training=False):
        # Get obs for policy.
        base_pose_w = self._robot_base.get_world_pose()
        base_quat_w = torch.tensor(base_pose_w[1], device='cpu', dtype=torch.float).reshape(1, -1)
        base_lin_vel_w = torch.tensor(self._robot_base.get_linear_velocity(), device='cpu', dtype=torch.float).reshape(1, -1)
        base_ang_vel_w = torch.tensor(self._robot_base.get_angular_velocity()[:], device='cpu', dtype=torch.float).reshape(1, -1)
        base_lin_vel = np.array(math_utils.quat_rotate_inverse(base_quat_w, base_lin_vel_w).reshape(-1))
        base_ang_vel = np.array(math_utils.quat_rotate_inverse(base_quat_w, base_ang_vel_w).reshape(-1))
        base_ang_vel = base_ang_vel * np.pi / 180.0
        
        projected_gravity = torch.tensor([[0., 0., -1.]], device='cpu', dtype=torch.float)
        projected_gravity = np.array(math_utils.quat_rotate_inverse(base_quat_w, projected_gravity).reshape(-1))
        joint_pos = self.joint_subset.get_joint_positions(
        ) if self.joint_subset is not None else self.isaac_robot.get_joint_positions()
        joint_vel = self.joint_subset.get_joint_velocities(
        ) if self.joint_subset is not None else self.isaac_robot.get_joint_velocities()

        base_height = base_pose_w[0][2]
        ankle_height = self.get_ankle_height()
        relative_base_height = np.array([base_height - ankle_height])

        obs = dict()
        obs[OBSERVATION_GRAVITY_KEY] = projected_gravity
        obs[OBSERVATION_BASE_LIN_VEL_KEY] = base_lin_vel
        obs[OBSERVATION_BASE_ANG_VEL_KEY] = base_ang_vel
        obs[OBSERVATION_JOINT_POS_KEY] = joint_pos
        obs[OBSERVATION_JOINT_VEL_KEY] = joint_vel
        obs[OBSERVATION_RELATIVE_BASE_HEIGHT_KEY] = relative_base_height

        if training:
            # ankle positions
            left_ankle_world_pos = torch.tensor(self._robot_left_ankle.get_world_pose()[0], device='cpu', dtype=torch.float)
            left_ankle_pos = np.array(math_utils.quat_rotate(base_quat_w, left_ankle_world_pos).squeeze(0))
            right_ankle_world_pos = torch.tensor(self._robot_right_ankle.get_world_pose()[0], device='cpu', dtype=torch.float)
            right_ankle_pos = np.array(math_utils.quat_rotate(base_quat_w, right_ankle_world_pos).squeeze(0))
            obs[OBSERVATION_LEFT_ANKLE_WORLD_POS_KEY] = left_ankle_world_pos
            obs[OBSERVATION_RIGHT_ANKLE_WORLD_POS_KEY] = right_ankle_world_pos
            obs[OBSERVATION_LEFT_ANKLE_POS_KEY] = left_ankle_pos
            obs[OBSERVATION_RIGHT_ANKLE_POS_KEY] = right_ankle_pos

            # detect foot collision with ground
            left_collision_list = self.left_ankle_collision_api.GetTriggeredCollisionsRel().GetTargets()
            left_foot_collision_with_ground = torch.tensor([self.is_contact_with_target(left_collision_list, self.ground_contact_prim_path)], device='cpu', dtype=torch.bool)
            right_collision_list = self.right_ankle_collision_api.GetTriggeredCollisionsRel().GetTargets()
            right_foot_collision_with_ground = torch.tensor([self.is_contact_with_target(right_collision_list, self.ground_contact_prim_path)], device='cpu', dtype=torch.bool)
            obs[OBSERVATION_LEFT_ANKLE_COLLISION_KEY] = left_foot_collision_with_ground
            obs[OBSERVATION_RIGHT_ANKLE_COLLISION_KEY] = right_foot_collision_with_ground
        
        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs[c_obs_name] = controller_obs.get_obs(training=training)
        for sensor_name, sensor_obs in self.sensors.items():
            if self.user_config.return_eye_image and sensor_name in self.sensor_to_key_dict:
                obs[self.sensor_to_key_dict[sensor_name]] = self.get_camera_frame_default(sensor_name, sensor_obs)
        return obs

    def get_video_frame(self):
        for sensor_name, sensor_obs in self.sensors.items():
            if sensor_name == self.sky_camera_key:
                return self.get_camera_frame_default(sensor_name, sensor_obs)
        return None
    
    def get_camera_frame_default(self, sensor_name, sensor_obs):
        if sensor_name not in self.default_frame_dict:
            if sensor_obs.config.resolution_x is not None and sensor_obs.config.resolution_y is not None:
                resolution = (sensor_obs.config.resolution_y, sensor_obs.config.resolution_x, 3)
            else:
                resolution = (128, 128, 3)
            self.default_frame_dict[sensor_name] = np.zeros((*resolution, 3), dtype=np.uint8)
            
        sensor_data = sensor_obs.get_data()
        return sensor_data['rgba'][:, :, :3] if 'rgba' in sensor_data and sensor_data['rgba'].dtype == np.uint8 else self.default_frame_dict[sensor_name]
