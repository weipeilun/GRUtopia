import os
import typing

import yaml
from omni.isaac.core.scenes import Scene

from grutopia.core.config import TaskUserConfig
from grutopia.core.robot.robot import create_robots
from grutopia.core.robot.robot_model import RobotModels

# ROBOT_TYPES = {}

ROBOT_MODELS_PATH = os.path.join(
    os.path.split(os.path.realpath(__file__))[0], '../../../grutopia_extension/robots', 'robot_models.yaml')
# print(ROBOT_MODELS_PATH)

with open(ROBOT_MODELS_PATH, 'r') as f:
    models = yaml.load(f.read(), Loader=yaml.FullLoader)
    # print(models)
    robot_models = RobotModels(**models)


def init_robots(the_config: TaskUserConfig, scene: Scene) -> typing.Dict:
    if the_config.robot_models_file_path is not None:
        with open(the_config.robot_models_file_path, 'r') as f:
            user_models = yaml.load(f.read(), Loader=yaml.FullLoader)
            # print(models)
            user_robot_models = RobotModels(**user_models)
        return create_robots(the_config, user_robot_models, scene)
    else:
        return create_robots(the_config, robot_models, scene)
