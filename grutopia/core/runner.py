import logging
import time
from collections import defaultdict
from typing import List
import torch
import shutil
import concurrent
import json
import tqdm
import numpy as np

# import numpy as np
from omni.isaac.core import World
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.utils.stage import add_reference_to_stage  # noqa F401
from omni.physx.scripts import utils
# from pxr import Usd  # noqa
from pathlib import Path
from PIL import Image
from pxr import Usd, UsdGeom, Gf, UsdLux

# Init
from grutopia.core.config import SimulatorConfig, TaskUserConfig
from grutopia.core.register import import_all_modules_for_register
from grutopia.core.scene import delete_prim_in_stage  # noqa F401
from grutopia.core.scene import create_object, create_scene  # noqa F401
from grutopia.core.task.task import BaseTask, create_task
from grutopia.core.util import log
from grutopia.core.constants import *
from grutopia.core.lerobot_dataset import video_utils
from grutopia.core.lerobot_dataset import compute_stats
from grutopia.core.lerobot_dataset import utils as dataset_utils
from grutopia.core.lerobot_dataset.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from grutopia.core.lerobot_dataset.push_dataset_to_hub import utils as push_dataset_utils
from grutopia.core.lerobot_dataset.push_dataset_to_hub import isaacsim_hdf5_format
from grutopia.core.lerobot_dataset.scripts import push_dataset_to_hub as push_dataset_scripts
from grutopia.npc import NPC


def save_image(img_array, key, frame_index, episode_index, videos_dir):
    img = Image.fromarray(img_array.astype(np.uint8))
    path = videos_dir / f"{key}_episode_{episode_index:06d}" / f"frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


class SimulatorRunner:

    def __init__(self, config: SimulatorConfig):
        import_all_modules_for_register()

        self._simulator_config = config.config
        physics_dt = self._simulator_config.simulator.physics_dt if self._simulator_config.simulator.physics_dt is not None else None
        rendering_dt = self._simulator_config.simulator.rendering_dt if self._simulator_config.simulator.rendering_dt is not None else None
        physics_dt = eval(physics_dt) if isinstance(physics_dt, str) else physics_dt
        rendering_dt = eval(rendering_dt) if isinstance(rendering_dt, str) else rendering_dt
        self.dt = physics_dt
        self.physics_fps = int(1 / self.dt)
        log.debug(f'Simulator physics fps: {self.physics_fps}')
        self._world = World(physics_dt=self.dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
        self._scene = self._world.scene
        self._stage = self._world.stage

        # setup scene
        prim_path = '/'
        if self._simulator_config.env_set.bg_type is None:
            self._scene.add_default_ground_plane()
            # set light exposure
            light = UsdLux.LightAPI.Get(self._stage, '/World/defaultGroundPlane/SphereLight')
            light.GetExposureAttr().Set(5.0)
            # set light translate
            xformPrim = UsdGeom.Xform.Get(self._stage, '/World/defaultGroundPlane/SphereLight')
            xformPrim.ClearXformOpOrder()
            translateOp = xformPrim.AddTranslateOp()
            translateOp.Set(Gf.Vec3d(0., 0., 10.0))
            xformPrim.GetXformOpOrderAttr().Set(['xformOp:translate', 'xformOp:orient', 'xformOp:scale'])
        elif self._simulator_config.env_set.bg_type != 'default':
            source, prim_path = create_scene(self._simulator_config.env_set.bg_path, prim_path_root='background')
            add_reference_to_stage(source, prim_path)

        self.npc: List[NPC] = []
        for npc_config in config.config.npc:
            self.npc.append(NPC(npc_config))

        self.render_interval = self._simulator_config.simulator.rendering_interval if self._simulator_config.simulator.rendering_interval is not None else 5
        log.info(f'rendering interval: {self.render_interval}')
        self.render_trigger = 0

        if self._simulator_config.env_set.save_dataset:
            self.ep_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

            self.image_keys = defaultdict(dict)
            self.not_image_keys = defaultdict(dict)
            self.action_keys = defaultdict(dict)

            self.local_dir = Path(self._simulator_config.env_set.dataset_path) / self._simulator_config.env_set.repo_id
            if self.local_dir.exists() and self._simulator_config.env_set.force_override:
                shutil.rmtree(self.local_dir)

            self.episodes_dir = self.local_dir / "episodes"
            self.episodes_dir.mkdir(parents=True, exist_ok=True)

            self.videos_dir = self.local_dir / "videos"
            self.videos_dir.mkdir(parents=True, exist_ok=True)

            # Logic to resume data recording
            self.rec_info_path = self.episodes_dir / "data_recording_info.json"
            if self.rec_info_path.exists():
                with open(self.rec_info_path) as f:
                    rec_info = json.load(f)
                self.init_episode_index = rec_info["last_episode_index"] + 1
            else:
                self.init_episode_index = 0

            self.futures = list()

            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

            logging.info(f'Saving training data to {str(self.local_dir)} at episode {self.init_episode_index}.')
        else:
            logging.info(f'Training data saving ignored.')

    def close(self):
        if self._simulator_config.env_set.save_dataset:
            episode_idx_list = list()
            default_robot_type = None
            default_action_name = None
            for robot_type, task_robot_action_counter_dict in self.robot_action_counter.items():
                for task_name, robot_action_counter_dict in task_robot_action_counter_dict.items():
                    for task_robot_name, num_actions in robot_action_counter_dict.items():
                        action_robot_info_dict = self.ep_dict[robot_type]
                        for action_name, robot_ep_dict in action_robot_info_dict.items():
                            if task_robot_name in robot_ep_dict:
                                ep_dict = robot_ep_dict[task_robot_name]
                                episode_index = self.robot_episode_index[robot_type][task_name][task_robot_name]
                                fps = self.robot_fps[robot_type][task_name][task_robot_name]
                                default_robot_type = robot_type
                                default_action_name = action_name
                                episode_idx_list.append(episode_index)

                                for key in self.not_image_keys[robot_type][action_name]:
                                    ep_dict[key] = torch.tensor(np.array(ep_dict[key]))

                                for key in self.action_keys[robot_type][action_name]:
                                    ep_dict[key] = torch.tensor(np.array(ep_dict[key]))

                                ep_dict["episode_index"] = torch.tensor([episode_index] * num_actions)
                                ep_dict["frame_index"] = torch.arange(0, num_actions, 1)
                                ep_dict["timestamp"] = torch.arange(0, num_actions, 1) / fps

                                done = torch.zeros(num_actions, dtype=torch.bool)
                                done[-1] = True
                                ep_dict["next.done"] = done

                                ep_path = self.episodes_dir / f"episode_{episode_index}.pth"
                                logging.info("Saving episode dictionary...")
                                torch.save(ep_dict, ep_path)

            max_episode_idx = max(episode_idx_list)
            rec_info = {
                "last_episode_index": max_episode_idx,
            }
            with open(self.rec_info_path, "w") as f:
                json.dump(rec_info, f)

            logging.info("Waiting for threads writing the images on disk to terminate...")
            for _ in tqdm.tqdm(
                    concurrent.futures.as_completed(self.futures), total=len(self.futures), desc="Writting images"
            ):
                pass
            self.executor.shutdown()

            logging.info("Encoding videos")
            for key in self.image_keys[robot_type][action_name]:
                for episode_index in episode_idx_list:
                    tmp_imgs_dir = self.videos_dir / f"{key}_episode_{episode_index:06d}"
                    fname = f"{key}_episode_{episode_index:06d}.mp4"
                    video_path = self.videos_dir / fname
                    if video_path.exists():
                        continue
                    # note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
                    # since video encoding with ffmpeg is already using multithreading.
                    video_utils.encode_video_frames(tmp_imgs_dir, video_path, fps, overwrite=True)
                    shutil.rmtree(tmp_imgs_dir)

            logging.info("Concatenating episodes")
            ep_dicts = []
            for episode_index in tqdm.tqdm(range(max_episode_idx + 1)):
                ep_path = self.episodes_dir / f"episode_{episode_index}.pth"
                ep_dict = torch.load(ep_path)
                ep_dicts.append(ep_dict)
            data_dict = push_dataset_utils.concatenate_episodes(ep_dicts)

            total_frames = data_dict["frame_index"].shape[0]
            data_dict["index"] = torch.arange(0, total_frames, 1)

            video = True
            run_compute_stats = True
            hf_dataset = isaacsim_hdf5_format.to_hf_dataset(data_dict, video, self.image_keys[default_robot_type][default_action_name], self.not_image_keys[default_robot_type][default_action_name], self.action_keys[default_robot_type][default_action_name])
            episode_data_index = dataset_utils.calculate_episode_data_index(hf_dataset)
            info = {
                "codebase_version": CODEBASE_VERSION,
                "fps": fps,
                "video": video,
            }
            if video:
                info["encoding"] = push_dataset_utils.get_default_encoding()

            lerobot_dataset = LeRobotDataset.from_preloaded(
                repo_id=self._simulator_config.env_set.repo_id,
                hf_dataset=hf_dataset,
                episode_data_index=episode_data_index,
                info=info,
                videos_dir=self.videos_dir,
            )
            if run_compute_stats:
                logging.info("Computing dataset statistics")
                stats = compute_stats.compute_stats(lerobot_dataset)
                lerobot_dataset.stats = stats
            else:
                logging.info("Skipping computation of the dataset statistrics")

            hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
            hf_dataset.save_to_disk(str(self.local_dir / "train"))

            meta_data_dir = self.local_dir / "meta_data"
            push_dataset_scripts.save_meta_data(info, stats, episode_data_index, meta_data_dir)

    @property
    def current_tasks(self) -> dict[str, BaseTask]:
        return self._world._current_tasks

    def _warm_up(self, steps=10, render=True):
        for _ in range(steps):
            self._world.step(render=render)

    def add_tasks(self, configs: List[TaskUserConfig]):
        for config in configs:
            task = create_task(config, self._scene)
            self._world.add_task(task)

        self._world.reset()
        self._warm_up()

        if self._simulator_config.env_set.save_dataset:
            # training data generator using LeRobotDataset
            episode_index = self.init_episode_index
            self.robot_frame_counter = defaultdict(lambda: defaultdict(dict))
            self.robot_action_counter = defaultdict(lambda: defaultdict(dict))
            self.robot_action_fps = defaultdict(dict)
            self.robot_fps = defaultdict(lambda: defaultdict(dict))
            self.robot_episode_index = defaultdict(lambda: defaultdict(dict))
            for task in self._simulator_config.tasks:
                for task_robot in task.robots:
                    robot_type = self.current_tasks.get(task.name).robots[task_robot.name].robot_model.type
                    self.robot_frame_counter[robot_type][task.name][task_robot.name] = 0
                    self.robot_action_counter[robot_type][task.name][task_robot.name] = 0
                    self.robot_action_fps[robot_type][task.name] = self.current_tasks[task.name].robots[task_robot.name].user_config.per_inference_frames
                    self.robot_fps[robot_type][task.name][task_robot.name] = int(self.physics_fps / self.current_tasks[task.name].robots[task_robot.name].user_config.per_inference_frames)
                    self.robot_episode_index[robot_type][task.name][task_robot.name] = episode_index
                    episode_index += 1

    def step(self, actions: dict, render: bool = True, return_obs=True, training=False):
        t0 = time.time()
        for task_name, action_dict in actions.items():
            task = self.current_tasks.get(task_name)
            for name, action in action_dict.items():
                if name in task.robots:
                    task.robots[name].apply_action(action)
                    # robot_type = task.robots[name].robot_model.type
                    #
                    # # save features to training dataset
                    # if self._simulator_config.env_set.save_dataset:
                    #     robot_frame_idx = self.robot_frame_counter[robot_type][task_name][name]
                    #     robot_action_fps = self.robot_action_fps[robot_type][task_name]
                    #     if robot_frame_idx % robot_action_fps == 0:
                    #         robot_episode_idx = self.robot_episode_index[robot_type][task_name][name]
                    #         robot_action_idx = self.robot_action_counter[robot_type][task_name][name]
                    #         for action_name in action.keys():
                    #             # init controller keys
                    #             if robot_type not in self.image_keys or robot_type not in self.not_image_keys or robot_type not in self.action_keys:
                    #                 self.image_keys[robot_type][action_name] = [key for key in action_result_dict[action_name].keys() if "image" in key and "action" not in key]
                    #                 self.not_image_keys[robot_type][action_name] = [key for key in action_result_dict[action_name].keys() if "image" not in key and "action" not in key]
                    #                 self.action_keys[robot_type][action_name] = [key for key in action_result_dict[action_name].keys() if "action" in key]
                    #
                    #             for key in self.not_image_keys[robot_type][action_name]:
                    #                 self.ep_dict[robot_type][action_name][name][key].append(action_result_dict[action_name][key])
                    #
                    #             for key in self.action_keys[robot_type][action_name]:
                    #                 self.ep_dict[robot_type][action_name][name][key].append(action_result_dict[action_name][key])
                    #
                    #             for key in self.image_keys[robot_type][action_name]:
                    #                 # Store the reference to the video frame, even tho the videos are not yet encoded
                    #                 fname = f"{key}_episode_{self.robot_episode_index[robot_type][task_name][name]:06d}.mp4"
                    #                 video_path = Path(f"GRUtopia/output/videos/{fname}")
                    #                 if video_path.exists():
                    #                     video_path.unlink()
                    #                 self.ep_dict[robot_type][action_name][name][key].append(
                    #                     {"path": f"videos/{fname}", "timestamp": robot_action_idx / self.robot_fps[robot_type][task_name][name]})
                    #
                    #                 self.futures += [
                    #                     self.executor.submit(
                    #                         save_image, action_result_dict[action_name][key], key, robot_action_idx, robot_episode_idx, self.videos_dir
                    #                     )
                    #                 ]
                    #         self.robot_action_counter[robot_type][task_name][name] = robot_action_idx + 1
                    #     self.robot_frame_counter[robot_type][task_name][name] = robot_frame_idx + 1

        t1 = time.time()
        self.render_trigger += 1
        render = render and self.render_trigger > self.render_interval
        if self.render_trigger > self.render_interval:
            self.render_trigger = 0
        t2 = time.time()
        self._world.step(render=render)
        t3 = time.time()

        # todo: ignore npc here
        # obs = self.get_obs()
        # for npc in self.npc:
        #     try:
        #         npc.feed(obs)
        #     except Exception as e:
        #         log.error(f'fail to feed npc {npc.name} with obs: {e}')
        t4 = time.time()
        # print(f't0={t1-t0}, t1={t2-t1}, t2={t3-t2}, t3={t4-t3}')

        if return_obs:
            obs = self.get_obs(training=training)
            return obs

    def get_obs(self, training=False):
        obs = {}
        for task_name, task in self.current_tasks.items():
            obs[task_name] = task.get_observations(training=training)
        return obs

    def get_video_frame(self, video_frame_dict):
        for task_name, task in self.current_tasks.items():
            task.get_video_frame(video_frame_dict)

    def get_current_time_step_index(self) -> int:
        return self._world.current_time_step_index

    def reset(self, task_robots: List[str] = None):
        if task_robots is None:
            self._world.reset()
            return
        for task_name, robot_name in task_robots:
            self.current_tasks[task_name].robots[robot_name].reset()

    def get_obj(self, name: str) -> XFormPrim:
        return self._world.scene.get_object(name)

    def remove_collider(self, prim_path: str):
        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.removeCollider(build)

    def add_collider(self, prim_path: str):
        build = self._world.stage.GetPrimAtPath(prim_path)
        if build.IsValid():
            utils.setCollider(build, approximationShape=None)
