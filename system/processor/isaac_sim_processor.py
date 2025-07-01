# encoding: utf-8
from abc import abstractmethod
from queue import Queue
import numpy as np
import logging
import time

from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container

from system.processor.abstract_processor import AbstractProcessor
from system.constants import CONFIG_KEY_ISAAC_SIM, CONFIG_KEY_TASK_CONFIG_FILE_PATH, CONFIG_KEY_ROBOT_CONFIG_FILE_PATH, CONFIG_KEY_VALID_CAMERAS, CONFIG_KEY_OBSERVATION_KEY_LIST
from system.tools.processor_config import config


class IsaacSimProcessor(AbstractProcessor):

    _PID = CONFIG_KEY_ISAAC_SIM

    def init(self):
        processor_config = config[self._PID]
        self.valid_camera_list = config[CONFIG_KEY_VALID_CAMERAS]
        self.observation_key_list = config[CONFIG_KEY_OBSERVATION_KEY_LIST]
        
        sim_config = SimulatorConfig(processor_config[CONFIG_KEY_TASK_CONFIG_FILE_PATH])
        robot_config = processor_config[CONFIG_KEY_ROBOT_CONFIG_FILE_PATH]
        
        headless = False
        webrtc = False

        if is_in_container():
            headless = True
            webrtc = True

        self.env = BaseEnv(sim_config, headless=headless, webrtc=webrtc, robot_models_file_path=robot_config, teleoperate=True)

        self.actions = [{'h1': {'random_move': []}}]
    
    def stop(self):
        self.env.close()
        self.env.simulation_app.close()
        return super().stop()
    
        # simplify run_loop for better performance
    def run_loop(self):
        while self._running:
            if self.interrupt_check():
                logging.debug("processor %s detect interrupt" % self._PID)
                break

            observations = self.env.step(actions=self.actions)
            
            all_robot_image_list = list()
            for task_name, robot_name, _ in self.valid_camera_list:
                robot_observation = observations[task_name][robot_name]
                robot_image_list = [robot_observation[camera_names] for camera_names in self.observation_key_list]
                all_robot_image_list.append(robot_image_list)
            self.push((True, np.asarray(all_robot_image_list)))
