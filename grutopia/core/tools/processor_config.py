# -*- coding: utf-8-*-

import cv2
import yaml
import logging
from grutopia.core.constants import (
    CONFIG_KEY_CAMERA,
    CONFIG_KEY_FREQUENCY,
    CONFIG_KEY_VISUALIZATION_HEIGHT,
    CONFIG_KEY_VISUALIZATION_WIDTH
)
from system.utils.path_util import get_relative_path


try:
    with open(get_relative_path('config/processor.yml'), "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    raise FileNotFoundError('processor config not found in %s' % get_relative_path('processor/config/processor.yml'))

try:
    with open(get_relative_path('processor/config/gpu.yml'), "r") as f:
        gpu_config = yaml.safe_load(f)
except Exception as e:
    raise FileNotFoundError('camera config not found in %s' % get_relative_path('processor/config/gpu.yml'))


def __check_camera_by_group(valid_camera_list, camera_role_path_dict, cameras_to_check, config_camera_positions, config_camera_attributes, camera_info_dict, camera_type_rgb):
    for camera_role in cameras_to_check:
        if camera_role in config_camera_positions:
            camera_port_info = config_camera_positions[camera_role]
            if 'camera' not in camera_port_info or 'port_number' not in camera_port_info:
                raise AttributeError(f'camera.yml error: `camera` or `port_number` not in config:{camera_port_info}')
            camera_type = camera_port_info['camera']
            port_number = camera_port_info['port_number']
            bus = camera_port_info['bus'] if 'bus' in camera_port_info else ''
            if camera_type not in config_camera_attributes:
                raise AttributeError(f'camera.yml error: camera type `{camera_type}` not in attribute:{config_camera_attributes}')
            camera_attributes_dict = config_camera_attributes[camera_type]
            if 'idVendor' not in camera_attributes_dict or 'idProduct' not in camera_attributes_dict:
                raise AttributeError(f'camera.yml error: `idVendor` or `idProduct` not in camera attributes:{camera_attributes_dict}')
            vendor_id = camera_attributes_dict['idVendor']
            product_id = camera_attributes_dict['idProduct']
            type = camera_attributes_dict['type']

            if (vendor_id, product_id, port_number) not in camera_info_dict:
                raise AttributeError(f'camera vendor_id={vendor_id}, product_id={product_id} not found on port={port_number}')

            camera_file_path = camera_info_dict[(vendor_id, product_id, port_number)]
            if type in camera_type_rgb:
                cap = cv2.VideoCapture(camera_file_path)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FPS, config[CONFIG_KEY_CAMERA][CONFIG_KEY_FREQUENCY])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config[CONFIG_KEY_CAMERA][CONFIG_KEY_VISUALIZATION_HEIGHT])
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config[CONFIG_KEY_CAMERA][CONFIG_KEY_VISUALIZATION_WIDTH])

                    if round(cap.get(cv2.CAP_PROP_FPS)) != config[CONFIG_KEY_CAMERA][CONFIG_KEY_FREQUENCY] or \
                            round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != config[CONFIG_KEY_CAMERA][
                        CONFIG_KEY_VISUALIZATION_HEIGHT] or \
                            round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != config[CONFIG_KEY_CAMERA][
                        CONFIG_KEY_VISUALIZATION_WIDTH]:
                        logging.warning(
                            f'Failed to config camera id={camera_role}. Camera resolution({cap.get(cv2.CAP_PROP_FRAME_WIDTH)}*{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}) or fps({cap.get(cv2.CAP_PROP_FPS)}) not consistent with config resolution({config[CONFIG_KEY_CAMERA][CONFIG_KEY_VISUALIZATION_WIDTH]}*{config[CONFIG_KEY_CAMERA][CONFIG_KEY_VISUALIZATION_HEIGHT]}), fps({config[CONFIG_KEY_CAMERA][CONFIG_KEY_FREQUENCY]})')
                    else:
                        valid_camera_list.append(camera_role)
                cap.release()
            else:
                valid_camera_list.append(camera_role)
            camera_role_path_dict[camera_role] = (camera_file_path, bus)

def check_cameras():
    valid_cameras = []
    valid_wide_cameras = []
    valid_temperature_cameras = []
    camera_role_path_dict = dict()

    camera_info_dict = read_all_camera_info()

    if CONFIG_KEY_CAMERA_POSITIONS in camera_config and CONFIG_KEY_CAMERA_ATTRIBUTES in camera_config:
        config_camera_positions = camera_config[CONFIG_KEY_CAMERA_POSITIONS]
        config_camera_attributes = camera_config[CONFIG_KEY_CAMERA_ATTRIBUTES]

        camera_type_rgb = {'camera', 'wide_camera'}

        if config_camera_positions is not None and len(config_camera_positions) > 0 and config_camera_attributes is not None and len(config_camera_attributes) > 0:
            __check_camera_by_group(valid_cameras, camera_role_path_dict, DMS_OMS_CAMERA_LIST_FREEZE if getattr(sys, 'frozen', False) else DMS_OMS_CAMERA_LIST, config_camera_positions, config_camera_attributes, camera_info_dict, camera_type_rgb)
            __check_camera_by_group(valid_wide_cameras, camera_role_path_dict, WIDE_CAMERA_LIST, config_camera_positions, config_camera_attributes, camera_info_dict, camera_type_rgb)
            __check_camera_by_group(valid_temperature_cameras, camera_role_path_dict, TEMPERATURE_CAMERA_LIST, config_camera_positions, config_camera_attributes, camera_info_dict, camera_type_rgb)

    return valid_cameras, valid_wide_cameras, valid_temperature_cameras, camera_role_path_dict
config[CONFIG_KEY_VALID_CAMERAS], config[CONFIG_KEY_VALID_WIDE_CAMERAS], config[CONFIG_KEY_VALID_IR_CAMERAS], camera_config[CONFIG_KEY_CAMERA_PORT_ID] = check_cameras()
