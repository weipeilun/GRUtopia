# -*- coding: utf-8-*-

import yaml
import logging
from system.constants import (
    CONFIG_KEY_VALID_CAMERAS,
    CONFIG_KEY_OBSERVATION_KEY_LIST,
    CONFIG_KEY_TASK_CONFIG_FILE_PATH,
    CONFIG_KEY_ROBOT_CONFIG_FILE_PATH,
    CONFIG_KEY_ROBOT_OUTPUT_CAMERAS,
)
from grutopia.core.config import SimulatorConfig
from grutopia.core.constants import DEFAULT_CAMERA_RESOLUTION
from system.utils.path_util import get_relative_path


try:
    with open(get_relative_path('system/config/processor.yml'), "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise FileNotFoundError('processor config not found in %s' % get_relative_path('processor/config/processor.yml'))

try:
    with open(get_relative_path('system/config/gpu.yml'), "r") as f:
        gpu_config = yaml.safe_load(f)
except Exception as e:
    raise FileNotFoundError('camera config not found in %s' % get_relative_path('processor/config/gpu.yml'))


def parse_camera_config():
    camera_list = list()
    observation_key_list = list()

    if 'isaac_sim' in config:
        isaac_sim_config = config['isaac_sim']
        
        if CONFIG_KEY_TASK_CONFIG_FILE_PATH in isaac_sim_config:
            try:
                camera_idx_dict = dict()
                for idx, (camera_name, observation_key) in enumerate(isaac_sim_config[CONFIG_KEY_ROBOT_OUTPUT_CAMERAS].items()):
                    camera_idx_dict[camera_name] = idx
                    observation_key_list.append(observation_key)
                
                # Read robots configuration
                if CONFIG_KEY_ROBOT_CONFIG_FILE_PATH in isaac_sim_config:
                    try:
                        with open(get_relative_path(isaac_sim_config[CONFIG_KEY_ROBOT_CONFIG_FILE_PATH]), "r") as f:
                            robot_config = yaml.safe_load(f)
                    except Exception as e:
                        logging.error(f"Error reading robot config file: {e}")
                        exit()
                else:
                    logging.error(f"`{CONFIG_KEY_ROBOT_CONFIG_FILE_PATH}` not in robot config file: {isaac_sim_config[CONFIG_KEY_TASK_CONFIG_FILE_PATH]}")
                    exit()
                # Generate a robot dict with robot type as key
                robot_type_dict = dict()
                for robot in robot_config['robots']:
                    camera_resolution_list = [None] * len(camera_idx_dict)
                    for sensor in robot['sensors']:
                        sensor_name = sensor.get('name', '')
                        if sensor['type'] == 'Camera' and sensor['name'] in camera_idx_dict:
                            if 'resolution_x' in sensor and 'resolution_y' in sensor:
                                resolution_x = sensor['resolution_x']
                                resolution_y = sensor['resolution_y']
                                camera_resolution_list[camera_idx_dict[sensor_name]] = (resolution_y, resolution_x, 3)
                            else:
                                camera_resolution_list[camera_idx_dict[sensor_name]] = DEFAULT_CAMERA_RESOLUTION
                    if all(resolution is not None for resolution in camera_resolution_list):
                        robot_type_dict[robot['type']] = camera_resolution_list

                # Get all robots config
                task_config = SimulatorConfig(isaac_sim_config[CONFIG_KEY_TASK_CONFIG_FILE_PATH])
                for task in task_config.config.tasks:
                    task_name = task.name
                    for robot in task.robots:
                        robot_name = robot.name
                        robot_type = robot.type
                        if robot_type in robot_type_dict:
                            robot_config_info = robot_type_dict[robot_type]
                            camera_list.append((task_name, robot_name, robot_config_info))
                        else:
                            logging.warning(f"Robot '{robot_name}' not found in robot config")
            except Exception as e:
                logging.error(f"Error reading task config file: {e}")
                exit()

    return camera_list, observation_key_list
    

def check_cameras():
    camera_list, camera_idx_dict = parse_camera_config()

    # Check if all camera configurations in camera_list have the same shape
    if camera_list:
        first_shape = camera_list[0][2]  # Get the shape of the first camera configuration
        if not all(camera[2] == first_shape for camera in camera_list):
            logging.error("Not all camera configurations have the same shape")
            exit()
    else:
        logging.warning("No camera configurations found in camera_list")
        exit()
    return camera_list, camera_idx_dict

config[CONFIG_KEY_VALID_CAMERAS], config[CONFIG_KEY_OBSERVATION_KEY_LIST] = check_cameras()
