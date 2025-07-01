import numpy as np
import torch

from grutopia.core.constants import *


def observation_dict_to_feature_array(obs_dict: dict):
    feature_list = []
    for observation_key in FEATURE_OBSERVATION_KEY_LIST:
        feature_list.append(obs_dict[observation_key])
    return np.concatenate(feature_list, axis=-1, dtype=np.float32)


def merge_observation_dict_by_key(obs_dict, robot_list, valid_action_name_dict):
    obs_dict_by_key = {key: list() for key in ALL_OBSERVATION_KEY_LIST}
    for task_name, robot_name, _, robot_action in robot_list:
        robot_obs = obs_dict[task_name][robot_name]

        robot_obs_dict = robot_obs.copy()
        robot_obs_dict[OBSERVATION_TRACKING_COMMAND_KEY] = robot_action
        for action_name, obs_key in valid_action_name_dict.items():
            if action_name in robot_obs:
                robot_obs_dict[obs_key] = robot_obs[action_name]

        for obs_key, data_list in obs_dict_by_key.items():
            # allow un-exist key here
            if obs_key in robot_obs_dict:
                obs_dict_by_key[obs_key].append(robot_obs_dict[obs_key])

    result_dict = dict()
    for obs_key, data_list in obs_dict_by_key.items():
        if len(data_list) > 0:
            result_dict[obs_key] = np.asarray(data_list)
    return result_dict


def observation_dict_list_to_tensor(obs_dict):
    result_dict = dict()
    for obs_key, data_array in obs_dict.items():
        result_dict[obs_key] = torch.tensor(data_array)
    return result_dict
