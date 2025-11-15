# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab_assets.robots.skyvortex import SKYVORTEX_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform, quat_apply


class SkyvortexManager:
    """
    一个用于在 Isaac Lab 仿真环境中管理 Skyvortex 无人机管理类。

    该类封装了对无人机旋翼施加力和计算产生力矩的逻辑，
    从而简化了主仿真循环。
    """

    def __init__(self, scene: InteractiveScene, device: str):
        """
        初始化 Skyvortex 控制器。

        Args:
            skyvortex_asset: 场景中的 Skyvortex 机器人实例。
            device: 仿真设备 (例如 "cuda:0" 或 "cpu")。
        """
        print("Initializing Skyvortex...")
        self.uam = scene["skyvortex"]
        self.device = device
        self.num_envs = scene.num_envs
        self.cfg = scene["skyvortex"].cfg

        # 获取旋翼和底座连杆的信息
        self.num_rotors = len(self.cfg.rotor_z_axes)
        rotor_body_names = [f"rotor_{i}_joint_jointbody" for i in range(0, self.num_rotors)]
        self.rotor_body_ids, _ = self.uam.find_bodies(rotor_body_names)
        self.base_link_id, _ = self.uam.find_bodies("world")

        # 存储旋翼轴向，并初始化一个零力矩张量用于API调用
        self.rotor_z_axes_in_base_frame = torch.tensor(self.cfg.rotor_z_axes, device=device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.rotor_factors = torch.tensor(self.cfg.rotor_factors, device=device).unsqueeze(0).repeat(self.num_envs, 1)
        self.zero_torques = torch.zeros((self.num_envs, self.num_rotors, 3), device=self.device)
        self.zero_forces = torch.zeros((self.num_envs, 1, 3), device=self.device)
        
        print(f"  - Found {self.num_rotors} rotors with body IDs: {self.rotor_body_ids}")
        print(f"  - Found base link with body ID: {self.base_link_id}")
        print("SkyvortexController initialized successfully.")


    def apply_force_and_torque(self, actions: torch.Tensor):
        """
        对旋翼施加升力。

        物理引擎会自动计算由此在底座连杆上产生的力矩。

        Args:
            actions: 一个形状为 (num_envs, num_rotors) 的张量，
                             包含每个旋翼的升力大小。
        """

        #将(num_envs, num_rotors)大小的张量变为(num_envs, num_rotors，3)大小的张量，前两个值为0
        lift_forces = torch.zeros((self.num_envs, self.num_rotors, 3), device=self.device)
        lift_forces[..., 2] = actions

        # 在旋翼刚体上设置外力
        # 我们只施加力；仿真实时处理在机器人上产生的力矩
        self.uam.set_external_force_and_torque(
            forces=lift_forces,
            torques=self.zero_torques,
            body_ids=self.rotor_body_ids
        )

        torque_magnitudes = actions * self.rotor_factors
        torque_rotors = torque_magnitudes.unsqueeze(2) * self.rotor_z_axes_in_base_frame
        torque_base_link = torque_rotors.sum(dim=1)
        torque_base_link = torque_base_link.unsqueeze(1)

        self.uam.set_external_force_and_torque(
            forces=self.zero_forces,
            torques=torque_base_link,
            body_ids=self.base_link_id
        )

        self.uam.write_data_to_sim()

    def get_observations(self) -> torch.Tensor:
        """
        获取无人机观测量
        """
        
        return torch.cat((self.uam.data.root_link_pose_w, self.uam.data.root_link_vel_w),dim=-1)
    
    def get_pos(self) -> torch.Tensor:
        """
        获取无人机位置
        """
        return self.uam.data.root_link_pos_w
    
    def get_orien(self) -> torch.Tensor:
        """
        获取无人机姿态
        """
        return self.uam.data.root_link_quat_w
    
    def get_vel(self) -> torch.Tensor:
        """
        获取无人机速度
        """
        return self.uam.data.root_link_vel_w
    
    def get_ang_vel(self) -> torch.Tensor:
        """
        获取无人机角速度
        """
        return self.uam.data.root_link_ang_vel_w
    
    def get_z_axis_w(self) -> torch.Tensor:
        """
        获取无人机顶侧朝向
        """
        quat = self.uam.data.root_link_quat_w
        z_axis_b = torch.zeros((self.num_envs, 3), device=self.device)
        z_axis_b[..., 2] = 1.0
        z_axis_w = quat_apply(quat, z_axis_b)[...,2]
        return z_axis_w

@configclass
class SkyvortexEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 75.0  # [N]
    action_space = 6
    observation_space = 13
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = SKYVORTEX_CFG.replace(prim_path="/World/envs/env_.*/skyvortex")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024, env_spacing=4.0, replicate_physics=True, #clone_in_fabric=True
    )

    # reset
    max_skyvortex_pos_xy = 3.0  # the skyvortex is reset if its position in the x-y plane exceeds that distance [m]
    max_skyvortex_pos_z = 5.0  # the skyvortex is reset if its height exceeds that position [m]

    # reward scales
    target_pos_base = (0.0, 0.0, 2.5) # 目标位置
    target_orient_base = (1.0, 0.0, 0.0, 0.0) # 目标姿态
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pos = 4.0 # 目标位置奖励权重
    rew_scale_orient = 1.5 # 姿态奖励权重
    rew_scale_vel = -0.5 # 速度惩罚权重
    rew_sacle_ang_vel = -0.5 # 角速度惩罚权重

    rew_scale_action = -0.01 # 动作惩罚权重


class SkyvortexEnv(DirectRLEnv):
    cfg: SkyvortexEnvCfg

    def __init__(self, cfg: SkyvortexEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        # self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale
        self.skyvortex_manager = SkyvortexManager(self.scene, self.device)
        self.pos = self.skyvortex_manager.get_pos()
        self.orien = self.skyvortex_manager.get_orien()
        self.vel = self.skyvortex_manager.get_vel()
        self.ang_vel = self.skyvortex_manager.get_ang_vel()
        self.target_pos = self.scene.env_origins + torch.tensor(self.cfg.target_pos_base, device=self.device)
        self.target_orient = torch.tensor(self.cfg.target_orient_base, device=self.device)
        # self.joint_pos = self.cartpole.data.joint_pos
        # self.joint_vel = self.cartpole.data.joint_vel

    def _setup_scene(self):
        self.skyvortex = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["skyvortex"] = self.skyvortex
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.skyvortex = self.scene["skyvortex"]
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene

 
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = (self.action_scale / 2) * (actions.clone() + 1.0)

    def _apply_action(self) -> None:
        self.skyvortex_manager.apply_force_and_torque(self.actions)

    def _get_observations(self) -> dict:
        obs = self.skyvortex_manager.get_observations()
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pos,
            self.cfg.rew_scale_orient,
            self.cfg.rew_scale_vel,
            self.cfg.rew_sacle_ang_vel,
            self.target_pos,
            self.target_orient,
            self.pos,
            self.orien,
            self.vel,
            self.ang_vel,
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.pos = self.skyvortex_manager.get_pos()
        self.orien = self.skyvortex_manager.get_orien()
        self.vel = self.skyvortex_manager.get_vel()
        self.ang_vel = self.skyvortex_manager.get_ang_vel()

        # 检测是否超过最大步数
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # condition1:检测位置是否超过xy的范围
        relative_pos = self.pos - self.scene.env_origins
        xy_dist = torch.norm(relative_pos[:, :2], dim=-1)
        out_of_bounds_xy = xy_dist > self.cfg.max_skyvortex_pos_xy

        # condition2:检测高度是否超过最大高度
        out_of_z_bounds = relative_pos[:, 2] > self.cfg.max_skyvortex_pos_z
        
        # condition3:检测无人机是否颠倒
        z_axis_w = self.skyvortex_manager.get_z_axis_w()
        upside_down = z_axis_w < 0.1

        termination = out_of_bounds_xy | out_of_z_bounds | upside_down 

        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the state of the environment for the given environment IDs."""
        if env_ids is None:
            env_ids = self.skyvortex._ALL_INDICES
        # call the parent class reset
        super()._reset_idx(env_ids)

        # -- 重置无人机状态 --

        # 获取默认的根状态 (通常是位置姿态速度都为0)
        root_state = self.skyvortex.data.default_root_state[env_ids]

        # 1. 重置位置
        # 加上每个环境自己的原点偏移
        root_state[:, :3] += self.scene.env_origins[env_ids]

        # 2. 重置机械臂位置
        joint_pos = self.skyvortex.data.default_joint_pos[env_ids]
        joint_vel = self.skyvortex.data.default_joint_vel[env_ids]

        # 将计算好的新状态一次性写入仿真
        self.skyvortex.write_root_state_to_sim(root_state, env_ids)
        self.skyvortex.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pos: float,
    rew_scale_orient: float,
    rew_scale_vel: float,
    rew_sacle_ang_vel: float,
    target_pos: torch.Tensor,
    target_orient: torch.Tensor,
    pos: torch.Tensor,
    orien: torch.Tensor,
    vel: torch.Tensor,
    ang_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    # 计算位置奖励
    rew_pos = torch.exp(rew_scale_pos * -torch.norm(pos - target_pos, dim=-1))
    # 计算姿态奖励
    rew_orien = torch.abs(rew_scale_orient * torch.sum(orien * target_orient,dim=1))
    # 计算速度惩罚
    rew_vel = rew_scale_vel * torch.norm(vel, dim=-1)
    # 计算角速度惩罚
    rew_vel = rew_sacle_ang_vel * torch.norm(ang_vel, dim=-1)


    total_reward = rew_alive + rew_termination + rew_pos + rew_orien + rew_vel
    return total_reward