import queue
import threading
from typing import Any, Dict, List
import multiprocessing as mp
from multiprocessing import shared_memory, Queue
import numpy as np
import wandb
import torch
import time
import os
import copy
from collections import deque
from collections import defaultdict
from ml_logger import logger
# from params_proto.neo_proto import PrefixProto

from grutopia.core.config import SimulatorConfig
from grutopia.core.util import log
from system.tools import interrupt
from grutopia.core.util import camera as camera_util
from grutopia_extension.training.utils import features as features_util
from grutopia.core.constants import OBSERVATION_IMAGE_LEFT_KEY, OBSERVATION_IMAGE_RIGHT_KEY, JOINT_ACTION_KEY, OBSERVATION_DICT_KEY_ORIGINAL_DICT, OBSERVATION_DICT_KEY_TENSOR, OBSERVATION_RELATIVE_BASE_HEIGHT_KEY


MINI_GYM_ROOT_DIR = 'GRUtopia/output/sim_robot_rl_model'


class BaseEnv:
    """
    Env base class. All tasks should inherit from this class(or subclass).
    ----------------------------------------------------------------------
    """

    def __init__(self, config: SimulatorConfig, headless: bool = True, webrtc: bool = False, native: bool = False, robot_models_file_path: str = None, teleoperate: bool = False) -> None:
        self._simulation_config = None
        self._render = None
        # Setup Multitask Env Parameters
        self.env_map = {}
        self.obs_map = {}

        self.config = config.config
        self.env_num = config.env_num
        self._column_length = int(np.sqrt(self.env_num))
        self.modify_configs(robot_models_file_path, teleoperate)

        # Init Isaac Sim
        from isaacsim import SimulationApp
        self.headless = headless
        self._simulation_app = SimulationApp({'headless': self.headless, 'anti_aliasing': 0})

        if webrtc:
            from omni.isaac.core.utils.extensions import enable_extension  # noqa

            self._simulation_app.set_setting('/app/window/drawMouse', True)
            self._simulation_app.set_setting('/app/livestream/proto', 'ws')
            self._simulation_app.set_setting('/app/livestream/websocket/framerate_limit', 60)
            self._simulation_app.set_setting('/ngx/enabled', False)
            enable_extension('omni.services.streamclient.webrtc')

        elif native:
            from omni.isaac.core.utils.extensions import enable_extension  # noqa

            self._simulation_app.set_setting("/app/window/drawMouse", True)
            self._simulation_app.set_setting("/app/livestream/proto", "ws")
            self._simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
            self._simulation_app.set_setting("/ngx/enabled", False)
            enable_extension("omni.kit.livestream.native")

        from grutopia.core import datahub  # noqa E402.
        from grutopia.core.runner import SimulatorRunner  # noqa E402.

        self._runner = SimulatorRunner(config=config)
        # self._simulation_config = sim_config

        log.debug(self.config.tasks)
        # create tasks
        self._runner.add_tasks(self.config.tasks)
        
        # teleoperate server
        if config.config.env_set.teleoperate_server:
            assert config.config.env_set.return_eye_image, ValueError('return_eye_image must be True when teleoperate_server is True')
            # Set up shared memory for left and right eye images
            left_eye_shape = None
            right_eye_shape = None
            for task_config in self._runner.current_tasks.values():
                for robot_config in task_config.robots.values():
                    left_eye_shape = camera_util.get_camera_resolution(robot_config.sensors['left_eye_camera'].config)
                    right_eye_shape = camera_util.get_camera_resolution(robot_config.sensors['right_eye_camera'].config)
                    break
            dtype = np.uint8

            left_shm = shared_memory.SharedMemory(create=True, size=np.prod(left_eye_shape) * dtype().itemsize)
            right_shm = shared_memory.SharedMemory(create=True, size=np.prod(right_eye_shape) * dtype().itemsize)
            self.teleoperate_image_trigger_queue = Queue()
            
            self.left_eye_image_array = np.ndarray(left_eye_shape, dtype=dtype, buffer=left_shm.buf)
            self.right_eye_image_array = np.ndarray(right_eye_shape, dtype=dtype, buffer=right_shm.buf)
        return

    def modify_configs(self, robot_models_file_path: str, teleoperate: bool):
        self.config.env_set.teleoperate_server = teleoperate
        # force return eye image for teleoperate
        if teleoperate:
            self.config.env_set.return_eye_image = True
            
        robot_per_inference_frames = self.config.env_set.robot_per_inference_frames
        return_eye_image = self.config.env_set.return_eye_image
        for task in self.config.tasks:
            for robot in task.robots:
                robot.per_inference_frames = robot_per_inference_frames
                robot.return_eye_image = return_eye_image
            task.robot_models_file_path = robot_models_file_path

    @property
    def runner(self):
        return self._runner

    @property
    def is_render(self):
        return self._render

    def get_dt(self):
        return self._runner.dt

    def step(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        run step with given action(with isaac step)

        Args:
            actions (List[Dict[str, Any]]): action(with isaac step)

        Returns:
            List[Dict[str, Any]]: observations(with isaac step)
        """
        if len(actions) != len(self.config.tasks):
            raise AssertionError('len of action list is not equal to len of task list')
        _actions = []
        for action_idx, action in enumerate(actions):
            _action = {}
            for k, v in action.items():
                _action[f'{k}_{action_idx}'] = v
            _actions.append(_action)
        action_after_reshape = {
            self.config.tasks[action_idx].name: action
            for action_idx, action in enumerate(_actions)
        }

        # log.debug(action_after_reshape)
        self._runner.step(action_after_reshape)
        observations = self.get_observations()
        return observations

    def reset(self, envs: List[int] = None):
        """
        reset the environment(use isaac word reset)

        Args:
            envs (List[int]): env need to be reset(default for reset all envs)
        """
        if envs is not None:
            if len(envs) == 0:
                return
            log.debug(f'============= reset: {envs} ==============')
            # int -> name
            self._runner.reset([self.config.tasks[e].name for e in envs])
            return self.get_observations(), {}
        self._runner.reset()
        return self.get_observations(), {}

    def get_observations(self) -> List[Dict[str, Any]]:
        """
        Get observations from Isaac environment
        Returns:
            List[Dict[str, Any]]: observations
        """
        _obs = self._runner.get_obs()
        return _obs

    def render(self, mode='human'):
        return

    def close(self):
        """close the environment"""
        self._runner.close()
        self._simulation_app.close()
        return

    @property
    def simulation_config(self):
        """config of simulation environment"""
        return self._simulation_config

    @property
    def simulation_app(self):
        """simulation app instance"""
        return self._simulation_app

# class RunnerArgs(PrefixProto, cli=False):
class RunnerArgs:
    # runner
    algorithm_class_name = 'PPO'
    num_steps_per_env = 60  # per iteration
    max_iterations = 10000  # number of policy updates

    # logging
    save_interval = 500  # check for potential saves every this many iterations
    save_video_interval = 100
    log_freq = 10

    # load and resume
    resume = False
    load_run = -1  # -1 = last run
    checkpoint = -1  # -1 = last saved model
    resume_path = None  # updated from load_run and chkpt


class TrainEnv(BaseEnv):
    """
    Train env class.
    ----------------------------------------------------------------------
    """
    class DataCaches:
        def __init__(self, curriculum_bins):
            from grutopia_extension.training.ppo.metrics_caches import DistCache, SlotCache

            self.slot_cache = SlotCache(curriculum_bins)
            self.dist_cache = DistCache()

    def __init__(self, config: SimulatorConfig, actions: dict, headless: bool = True, webrtc: bool = False, native: bool = False, no_wandb=True) -> None:
        run_name = 'h1_locomotion_training'
        name_prefix = 'grutopia'
        wandb_group = 'Peter Wei'
        wandb.init(
            project="robot",
            name=f"{name_prefix}-{run_name}",
            group=wandb_group,
            config={
                "actions": actions,
            },
            mode="disabled" if no_wandb else "online",
        )

        from grutopia_extension.training.ppo.ppo import PPO
        from grutopia_extension.training.ppo.actor_critic import ActorCritic

        observation_dim = config.config.env_set.observation_dim
        privileged_obs_shape = config.config.env_set.privileged_obs_shape
        num_actions = config.config.env_set.num_actions
        self.device = config.config.env_set.device

        self.num_envs = len(actions)
        self.num_obs = observation_dim
        self.num_privileged_obs = privileged_obs_shape
        self.num_steps_per_env = RunnerArgs.num_steps_per_env
        self.max_episode_length = np.ceil(config.config.env_set.episode_length_s / eval(config.config.simulator.physics_dt) / config.config.env_set.robot_per_inference_frames)

        self.obs_history_length = config.config.env_set.num_observation_history
        self.num_obs_history = self.obs_history_length * self.num_obs

        actor_critic = ActorCritic(observation_dim,
                                   privileged_obs_shape,
                                   self.num_obs_history,
                                   num_actions,
                                   ).to(self.device)
        self.alg = PPO(actor_critic, device=self.device)

        # init storage and model
        self.alg.init_storage(self.num_envs, self.num_steps_per_env, [observation_dim],
                         [privileged_obs_shape], [self.num_obs_history], [num_actions])

        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # todo: currently ignore privileged observation, set requires_grad=False
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                              dtype=torch.float, requires_grad=False)

        # history
        self.obs_history = torch.zeros(self.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.device, requires_grad=False)

        self.actions = actions
        self.valid_action_name_dict = {'train_ppo': JOINT_ACTION_KEY}

        valid_actions_list = {robot_name: (action_name, action_list) for robot_name, action in self.actions.items() for action_name, action_list in action.items() if action_name in self.valid_action_name_dict}
        self.robot_list = [(task.name, robot.name, *valid_actions_list[robot.prim_path.split('/')[-1]]) for task in config.config.tasks for robot in task.robots]

        self.caches = self.DataCaches(self.num_envs)

        self.video_frame_dict = None
        if config.config.env_set.record_video:
            self.video_frame_dict = defaultdict(lambda: [])
            self.last_recording_it = 0
            self.video_info_queue = queue.Queue()
            threading.Thread(target=self.log_video_thread, args=(self.video_info_queue, ), daemon=True).start()

        super().__init__(config=config, headless=headless, webrtc=webrtc, native=native)

        self._prepare_reward_function(self._runner.get_obs(training=True))

    def log_video_thread(self, video_info_queue):
        while True:
            if interrupt.interrupt_callback():
                break

            try:
                filename, video_buffer = video_info_queue.get(block=True, timeout=0.1)
                logger.save_video(video_buffer, filename,
                                  fps=int(self._runner.physics_fps / self.config.env_set.robot_per_inference_frames))
            except queue.Empty:
                continue

    def _prepare_reward_function(self, original_obs_dict):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # reward containers
        reward_container_name = self.config.env_set.reward_container_name
        from grutopia_extension.training.rewards.original_reward import OriginalReward
        from grutopia_extension.training.rewards.eureka_reward import EurekaReward
        if reward_container_name == "OriginalReward":
            self.reward_container = OriginalReward(self)
        elif reward_container_name == "EurekaReward":
            self.reward_container = EurekaReward(self)
        else:
            raise NameError(f"Unknown reward container: {reward_container_name}")

        if reward_container_name == "OriginalReward":
            # remove zero scales + multiply non-zero ones by dt
            for key in list(self.reward_scales.keys()):
                scale = self.reward_scales[key]
                if scale == 0:
                    self.reward_scales.pop(key)
                else:
                    self.reward_scales[key] *= self.dt
            # prepare list of functions
            self.reward_functions = []
            self.reward_names = []
            for name, scale in self.reward_scales.items():
                if name == "termination":
                    continue
                if not hasattr(self.reward_container, '_reward_' + name):
                    print(f"Warning: reward {'_reward_' + name} has nonzero coefficient but was not found!")
                else:
                    self.reward_names.append(name)
                    self.reward_functions.append(getattr(self.reward_container, '_reward_' + name))
        else:
            _, reward_components = self.reward_container.compute_reward(original_obs_dict)
            self.reward_names = list(reward_components.keys())

        # reward episode sums
        self.episode_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_names}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                 requires_grad=False)
        self.episode_sums["success"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                    requires_grad=False)
        self.episode_sums_eval = {
            name: -1 * torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in self.reward_names}
        self.episode_sums_eval["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                      requires_grad=False)
        self.episode_sums_eval["success"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                                            requires_grad=False)
        self.command_sums = {
            name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            for name in
            list(self.reward_names) + ["lin_vel_raw", "ang_vel_raw", "lin_vel_residual", "ang_vel_residual",
                                               "ep_timesteps"]}

    def get_privileged_observations(self, horizon=0):
        if horizon == 0:
            return self.privileged_obs_buf
        else:
            env_timesteps_remaining_until_rand = int(self.cfg.domain_rand.rand_interval) - self.episode_length_buf % int(self.cfg.domain_rand.rand_interval)
            switched_env_ids = torch.arange(self.num_envs, device=self.device)[env_timesteps_remaining_until_rand>=horizon]
            privileged_obs_buf = self.privileged_obs_buf
            privileged_obs_buf[switched_env_ids] = self.next_privileged_obs_buf[switched_env_ids]
            return privileged_obs_buf

    def merge_observation_dict_by_key(self, obs_dict):
        return features_util.merge_observation_dict_by_key(obs_dict, self.robot_list, self.valid_action_name_dict)

    def get_observations(self, original_obs=None, training=False):
        if original_obs is None:
            original_obs = self._runner.get_obs(training=training)
        obs_dict_by_key = self.merge_observation_dict_by_key(original_obs)
        obs_list = features_util.observation_dict_to_feature_array(obs_dict_by_key)
        obs_tensor = torch.tensor(obs_list, device=self.device)
        privileged_obs = self.get_privileged_observations()
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs:], obs_tensor), dim=-1)
        return {OBSERVATION_DICT_KEY_ORIGINAL_DICT: original_obs, OBSERVATION_DICT_KEY_TENSOR: obs_tensor, 'privileged_obs': privileged_obs, 'obs_history': self.obs_history}

    def learn(self, num_learning_iterations=RunnerArgs.max_iterations, init_at_random_ep_len=False, eval_freq=100, eval_expert=False):
        # initialize writer
        assert logger.prefix, "you will overwrite the entire instrument server"

        logger.start('start', 'epoch', 'episode', 'run', 'step')
        task_dir = os.path.join(MINI_GYM_ROOT_DIR, 'tmp')
        model_dir = os.path.join(task_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)

        if self.config.env_set.record_video:
            video_dir = os.path.join(task_dir, 'videos')
            os.makedirs(video_dir, exist_ok=True)

        # wandb.watch(self.alg.actor_critic, log="all", log_freq=RunnerArgs.log_freq)

        if init_at_random_ep_len:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf,
                                                             high=int(self.max_episode_length))

        # split train and test envs
        num_train_envs = self.num_envs
        self.num_train_envs = num_train_envs
        self.num_eval_envs = self.num_envs - self.num_train_envs

        obs_dict = self.get_observations()
        obs, privileged_obs, obs_history = obs_dict[OBSERVATION_DICT_KEY_TENSOR], obs_dict["privileged_obs"], obs_dict["obs_history"]
        obs, privileged_obs, obs_history = obs.to(self.device), privileged_obs.to(self.device), obs_history.to(
            self.device)
        self.alg.actor_critic.train()

        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        rewbuffer_eval = deque(maxlen=100)
        lenbuffer_eval = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        if hasattr(self, "curriculum"):
            self.caches.__init__(curriculum_bins=len(self.curriculum))

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    t0 = time.time()
                    actions_train = self.alg.act(obs[:num_train_envs], privileged_obs[:num_train_envs],
                                                 obs_history[:num_train_envs])
                    if eval_expert:
                        actions_eval = self.alg.actor_critic.act_teacher(obs[num_train_envs:],
                                                                         privileged_obs[num_train_envs:])
                    else:
                        actions_eval = self.alg.actor_critic.act_student(obs[num_train_envs:],
                                                                         obs_history[num_train_envs:])
                    t1 = time.time()
                    # privileged information is concatenated to the observation, and observation history is stored in info
                    ret = self.step(torch.cat((actions_train, actions_eval), dim=0), self.video_frame_dict)
                    t2 = time.time()
                    obs_dict, rewards, dones, infos = ret
                    obs, privileged_obs, obs_history, env_bins = obs_dict[OBSERVATION_DICT_KEY_TENSOR], obs_dict["privileged_obs"], obs_dict[
                        "obs_history"], infos["env_bins"]

                    obs, privileged_obs, obs_history, rewards, dones, env_bins = obs.to(self.device), privileged_obs.to(
                        self.device), obs_history.to(self.device), rewards.to(self.device), dones.to(self.device), env_bins.to(self.device)
                    t3 = time.time()
                    self.alg.process_env_step(rewards[:num_train_envs], dones[:num_train_envs], env_bins[:num_train_envs], infos)
                    t4 = time.time()

                    wandb_log_dict = dict()
                    for info_key, info_value_tensor in infos.items():
                        if info_key.startswith('rewards'):
                            info_value_array = info_value_tensor.detach().cpu().numpy()
                            for info_value_idx, info_value in enumerate(info_value_array):
                                wandb_key1 = f"{info_key.replace('rewards.', '')}/robot{info_value_idx}"
                                wandb_log_dict[wandb_key1] = info_value
                                wandb_key2 = f"robot{info_value_idx}/{info_key.replace('rewards.', '')}"
                                wandb_log_dict[wandb_key2] = info_value
                    # if len(wandb_log_dict) > 0:
                    #     if not self.wandb_defined:
                    #         for wandb_log_key in wandb_log_dict.keys():
                    #             wandb.define_metric(wandb_log_key)
                    #         self.wandb_defined = True
                        wandb.log(wandb_log_dict, step=it)
                    t5 = time.time()

                    if 'train/episode' in infos:
                        with logger.Prefix(metrics="train/episode"):
                            logger.store_metrics(**infos['train/episode'])
                        wandb.log(infos['train/episode'], step=it)

                    if 'eval/episode' in infos:
                        with logger.Prefix(metrics="eval/episode"):
                            logger.store_metrics(**infos['eval/episode'])
                        wandb.log(infos['eval/episode'], step=it)

                    if 'curriculum' in infos:
                        curr_bins_train = infos['curriculum']['reset_train_env_bins']
                        curr_bins_eval = infos['curriculum']['reset_eval_env_bins']

                        self.caches.slot_cache.log(curr_bins_train, **{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('slot/train')
                        })
                        self.caches.slot_cache.log(curr_bins_eval, **{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('slot/eval')
                        })
                        self.caches.dist_cache.log(**{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('dist/train')
                        })
                        self.caches.dist_cache.log(**{
                            k.split("/", 1)[-1]: v for k, v in infos['curriculum'].items()
                            if k.startswith('dist/eval')
                        })

                        cur_reward_sum += rewards
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)

                        new_ids_train = new_ids[new_ids < num_train_envs]
                        rewbuffer.extend(cur_reward_sum[new_ids_train].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids_train].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_train] = 0
                        cur_episode_length[new_ids_train] = 0

                        new_ids_eval = new_ids[new_ids >= num_train_envs]
                        rewbuffer_eval.extend(cur_reward_sum[new_ids_eval].cpu().numpy().tolist())
                        lenbuffer_eval.extend(cur_episode_length[new_ids_eval].cpu().numpy().tolist())
                        cur_reward_sum[new_ids_eval] = 0
                        cur_episode_length[new_ids_eval] = 0
                    t5 = time.time()
                    # print(f't0={t1-t0}, t1={t2-t1}, t2={t3-t2}, t3={t4-t3}, t4={t5-t4}')

                # Learning step
                self.alg.compute_returns(obs[:num_train_envs], privileged_obs[:num_train_envs])

                if num_train_envs < self.num_envs and it % eval_freq == 0:
                    self.env.reset_evaluation_envs()

                    logger.save_pkl({"iteration": it,
                                     **self.caches.slot_cache.get_summary(),
                                     **self.caches.dist_cache.get_summary()},
                                    path=f"curriculum/info.pkl", append=True)

            mean_value_loss, mean_surrogate_loss, mean_adaptation_module_loss = self.alg.update(self.caches)

            logger.store_metrics(
                time_elapsed=logger.since('start'),
                time_iter=logger.split('epoch'),
                adaptation_loss=mean_adaptation_module_loss,
                mean_value_loss=mean_value_loss,
                mean_surrogate_loss=mean_surrogate_loss
            )
            wandb.log({
                "adaptation_loss": mean_adaptation_module_loss,
                "mean_value_loss": mean_value_loss,
                "mean_surrogate_loss": mean_surrogate_loss,
            }, step=it)

            if self.config.env_set.record_video:
                self.log_video(it, video_dir)

            self.tot_timesteps += self.num_steps_per_env * self.num_envs
            # if logger.every(RunnerArgs.log_freq, "iteration", start_on=1):
            #     # if it % Config.log_freq == 0:
            #     logger.log_metrics_summary(key_values={"timesteps": self.tot_timesteps, "iterations": it})
            #     logger.job_running()

            wandb.log({"timesteps": self.tot_timesteps, "iterations": it}, step=it)

            if it % RunnerArgs.save_interval == 0:
                with logger.Sync():
                    logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
                    logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", f"checkpoints/ac_weights_last.pt")
                    # wandb.save(f"checkpoints/ac_weights_{it:06d}.pt")
                    # wandb.save(f"checkpoints/ac_weights_last.pt")

                    os.makedirs(model_dir, exist_ok=True)

                    adaptation_module_path = f'{model_dir}/adaptation_module_{it}.jit'
                    adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
                    traced_script_adaptation_module = torch.jit.script(adaptation_module)
                    traced_script_adaptation_module.save(adaptation_module_path)

                    body_path = f'{model_dir}/body_{it}.jit'
                    body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
                    traced_script_body_module = torch.jit.script(body_model)
                    traced_script_body_module.save(body_path)

                    env_factor_encoder_path = f'{model_dir}/env_factor_encoder_{it}.jit'
                    env_factor_encoder_model = copy.deepcopy(self.alg.actor_critic.env_factor_encoder).to('cpu')
                    env_factor_encoder_module = torch.jit.script(env_factor_encoder_model)
                    env_factor_encoder_module.save(env_factor_encoder_path)

                    logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/")
                    logger.upload_file(file_path=body_path, target_path=f"checkpoints/")
                    logger.upload_file(file_path=env_factor_encoder_path, target_path=f"checkpoints/")

                    # wandb.save(f"{model_dir}/adaptation_module_latest_{it}.jit")
                    # wandb.save(f"{model_dir}/body_latest_{it}.jit")
                    # wandb.save(f"{model_dir}/env_factor_encoder_{it}.jit")

            self.current_learning_iteration += num_learning_iterations

        with logger.Sync():
            logger.torch_save(self.alg.actor_critic.state_dict(), f"checkpoints/ac_weights_{it:06d}.pt")
            logger.duplicate(f"checkpoints/ac_weights_{it:06d}.pt", "checkpoints/ac_weights_last.pt")
            # wandb.save(f"checkpoints/ac_weights_{it:06d}.pt")
            # wandb.save(f"checkpoints/ac_weights_last.pt")

            os.makedirs(model_dir, exist_ok=True)

            adaptation_module_path = f'{model_dir}/adaptation_module_latest.jit'
            adaptation_module = copy.deepcopy(self.alg.actor_critic.adaptation_module).to('cpu')
            traced_script_adaptation_module = torch.jit.script(adaptation_module)
            traced_script_adaptation_module.save(adaptation_module_path)

            body_path = f'{model_dir}/body_latest.jit'
            body_model = copy.deepcopy(self.alg.actor_critic.actor_body).to('cpu')
            traced_script_body_module = torch.jit.script(body_model)
            traced_script_body_module.save(body_path)

            env_factor_encoder_path = f'{model_dir}/env_factor_encoder_latest.jit'
            env_factor_encoder_model = copy.deepcopy(self.alg.actor_critic.env_factor_encoder).to('cpu')
            traced_script_env_factor_encoder_module = torch.jit.script(env_factor_encoder_model)
            traced_script_env_factor_encoder_module.save(env_factor_encoder_path)

            logger.upload_file(file_path=adaptation_module_path, target_path=f"checkpoints/")
            logger.upload_file(file_path=body_path, target_path=f"checkpoints/")
            logger.upload_file(file_path=env_factor_encoder_path, target_path=f"checkpoints/")

            # wandb.save(f"{model_dir}/adaptation_module_latest.jit")
            # wandb.save(f"{model_dir}/body_latest.jit")
            # wandb.save(f"{model_dir}/env_factor_encoder_latest.jit")

    def log_video(self, it, video_dir):
        if it - self.last_recording_it >= self.config.env_set.save_video_interval:
            self.last_recording_it = it

            for robot_name, frames in self.video_frame_dict.items():
                if len(frames) > 0:
                    self.video_info_queue.put((f"{video_dir}/{robot_name}_{it:05d}.mp4", frames.copy()))
                    frames.clear()

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_expert_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_expert

    def step(self, actions: torch.Tensor, video_frame_dict: dict = None) -> List[Dict[str, Any]]:
        """
        run step with given action(with isaac step)

        Args:
            actions (List[Dict[str, Any]]): action(with isaac step)

        Returns:
            List[Dict[str, Any]]: observations(with isaac step)
        """
        if len(actions) != self.num_envs:
            raise AssertionError('len of action list is not equal to len of task list')

        t0 = time.time()
        actions = actions.detach().cpu().numpy()
        action_dict = defaultdict(lambda: defaultdict(dict))
        for action_idx, (action, (task_name, robot_name, action_name, _)) in enumerate(zip(actions, self.robot_list)):
            action_dict[task_name][robot_name][action_name] = action
        t1 = time.time()

        # log.debug(action_after_reshape)
        # attention here: use the last observation of an action batch to generate training data
        for _ in range(self.config.env_set.robot_per_inference_frames):
            self._runner.step(action_dict, return_obs=False)
        t2 = time.time()
        if video_frame_dict is not None:
            self._runner.get_video_frame(video_frame_dict)

        t3 = time.time()
        obs_dict = self.get_observations(training=True)
        t4 = time.time()
        rewards, reward_components = self.reward_container.compute_reward(obs_dict[OBSERVATION_DICT_KEY_ORIGINAL_DICT])
        terminations = self.check_termination(obs_dict[OBSERVATION_DICT_KEY_ORIGINAL_DICT])
        t5 = time.time()

        # todo:: check envbins
        reward_components["env_bins"] = torch.arange(0, rewards.shape[0], dtype=torch.int, device='cpu')

        env_ids = terminations.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        # print(f't0={t1-t0}, t1={t2-t1}, t2={t3-t2}, t3={t4-t3}, t4={t5-t4}')

        return obs_dict, rewards, terminations, reward_components

    def check_termination(self, obs_dict):
        """ Check if environments need to be reset
        """
        # use body height for terminal
        obs_dict_by_key = self.merge_observation_dict_by_key(obs_dict)
        obs_dict_by_key = features_util.observation_dict_list_to_tensor(obs_dict_by_key)
        terminations_tensor = torch.any(obs_dict_by_key[OBSERVATION_RELATIVE_BASE_HEIGHT_KEY] < 0.3, dim=1)
        return terminations_tensor

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return

        # # update curriculum
        # if self.cfg.terrain.curriculum:
        #     self._update_terrain_curriculum(env_ids)

        # reset robot states
        task_robot_list = []
        for env_id in env_ids:
            task_robot_list.append(self.robot_list[env_id][:2])
        self._runner.reset(task_robot_list)

        # reset buffers
        self.episode_length_buf[env_ids] = 0

        self.time_out_buf[env_ids] = 0

        self.privileged_obs_buf[env_ids, :] = 0
        self.obs_history[env_ids, :] = 0

        # reset reward container
        self.reward_container.reset(env_ids)