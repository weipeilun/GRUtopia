from grutopia.core.config import SimulatorConfig
from grutopia.core.env import BaseEnv
from grutopia.core.util.container import is_in_container

file_path = './GRUtopia/demo/configs/h1_locomotion_data_gen.yaml'
sim_config = SimulatorConfig(file_path)

headless = False
webrtc = False

if is_in_container():
    headless = True
    webrtc = True

actions = {
    'h1_0': {'move_by_speed': [1, 0, 0]},
    'h1_1': {'move_by_speed': [-1, 0, 0]},
    'h1_2': {'move_by_speed': [0, 1, 0]},
    'h1_3': {'move_by_speed': [0, -1, 0]},
    'h1_4': {'move_by_speed': [0, 0, 1]},
    'h1_5': {'move_by_speed': [0, 0, -1]},
    'h1_6': {'move_by_speed': [0, 0, 0]},
}
# actions = {
#     'h1_0': {'move_by_speed': [0, 0, -1]},
#     'h1_1': {'move_by_speed': [0, 0, -1]},
#     'h1_2': {'move_by_speed': [0, 0, -1]},
#     'h1_3': {'move_by_speed': [0, 0, -1]},
#     'h1_4': {'move_by_speed': [0, 0, -1]},
#     'h1_5': {'move_by_speed': [0, 0, -1]},
#     'h1_6': {'move_by_speed': [0, 0, -1]},
# }
stop_actions = {
    'h1_0': {'move_by_speed': [0, 0, 0]},
    'h1_1': {'move_by_speed': [0, 0, 0]},
    'h1_2': {'move_by_speed': [0, 0, 0]},
    'h1_3': {'move_by_speed': [0, 0, 0]},
    'h1_4': {'move_by_speed': [0, 0, 0]},
    'h1_5': {'move_by_speed': [0, 0, 0]},
    'h1_6': {'move_by_speed': [0, 0, 0]},
}

env = BaseEnv(sim_config, headless=headless, webrtc=webrtc)

task_name = env.config.tasks[0].name
robot_name = env.config.tasks[0].robots[0].name

sim_fps = int(1 / eval(sim_config.config.simulator.physics_dt))

path = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (3.0, 4.0, 0.0)]
i = 0

# actions = {'h1': {'move_with_keyboard': []}}

movement_start_time = 1.
movement_end_time = 5.
env_stop_timestamp = 5.5

while env.simulation_app.is_running():
    current_timestamp = i / sim_fps
    apply_movement = stop_actions
    if movement_start_time < current_timestamp < movement_end_time:
        apply_movement = actions

    i += 1
    env_actions = []
    env_actions.append(actions)
    obs = env.step(actions=env_actions)

    if i % sim_fps == 0:
        print(i)

    if current_timestamp >= env_stop_timestamp:
        break

env.close()
env.simulation_app.close()
