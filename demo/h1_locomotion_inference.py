from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container

file_path = './GRUtopia/demo/configs/h1_locomotion_inference.yaml'
sim_config = SimulatorConfig(file_path)

headless = False
webrtc = False

if is_in_container():
    headless = True
    webrtc = True

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

import numpy as np
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles

from grutopia.core.util import log

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0

actions = {'h1': {'random_move': []}}

while env.simulation_app.is_running():
    i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)

    if i % 10000 == 0:
        print(i)

env.close()
env.simulation_app.close()
