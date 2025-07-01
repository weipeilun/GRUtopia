from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container
import os
import sys

file_path = './GRUtopia/demo/configs/h1_house.yaml'
sim_config = SimulatorConfig(file_path)

# sys_path = os.path.split(os.path.abspath(sim_config.config.tasks[0].scene_asset_path))[0]
# print(sys_path)
# sys.path.append(sys_path)

headless = False
webrtc = False

if is_in_container():
    headless = True
    webrtc = True

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

i = 0

actions = {'h1': {'move_with_keyboard': []}}

while env.simulation_app.is_running():
    i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)

    if i % 10000 == 0:
        print(i)

env.simulation_app.close()
