from grutopia.core.config import SimulatorConfig
from grutopia.core.env import TrainEnv
from grutopia.core.util.container import is_in_container
import os
import wandb
import yaml


no_wandb = False
if not no_wandb:
    wandb_config_filename = os.path.join(os.path.expanduser('~'), '.wandb/config.yaml')
    wandb_config_dict = yaml.load(open(wandb_config_filename, 'r'), yaml.FullLoader)
    wandb.login(key=wandb_config_dict['key'])


file_path = './GRUtopia/demo/configs/h1_locomotion_train.yaml'
sim_config = SimulatorConfig(file_path)

headless = False
webrtc = False

if is_in_container():
    headless = True
    webrtc = True

actions = {
    'h1_0': {'train_ppo': [1, 0, 0]},
    'h1_1': {'train_ppo': [-1, 0, 0]},
    'h1_2': {'train_ppo': [0, 1, 0]},
    'h1_3': {'train_ppo': [0, -1, 0]},
    'h1_4': {'train_ppo': [0, 0, 1]},
    'h1_5': {'train_ppo': [0, 0, -1]},
    'h1_6': {'train_ppo': [0, 0, 0]},
}

env = TrainEnv(sim_config, actions, headless=headless, webrtc=webrtc, no_wandb=False)
env.learn()

env.close()
env.simulation_app.close()
