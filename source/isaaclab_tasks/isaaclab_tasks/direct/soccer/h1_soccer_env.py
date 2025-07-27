import gymnasium as gym
import torch

import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, quat_rotate_inverse

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor

from .h1_soccer_env_cfg import H1SoccerEnvCfg
from .utils import RobotContactSensor, generate_target_directions


# Helper functions
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


class H1SoccerEnv(DirectRLEnv):
    cfg: H1SoccerEnvCfg

    def __init__(self, cfg: H1SoccerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self.actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self.previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self.action_scale = self.cfg.action_scale
        self.joint_gears = torch.tensor(self.cfg.joint_gears, dtype=torch.float32, device=self.sim.device)
        self.motor_effort_ratio = torch.ones_like(self.joint_gears, device=self.sim.device)
        self._joint_dof_idx, _ = self.robot.find_joints(".*")

        # Distance-based potential function for a dense reward.
        # Theoretically, potential-based reward shaping won't alter the optimal policy.
        self.potentials = torch.zeros(self.num_envs, dtype=torch.float32, device=self.sim.device)
        self.prev_potentials = torch.zeros_like(self.potentials)
        
        # TODO: Dubins-like potential function for a dense reward.

        # For shooting task, target becomes where the soccer is
        # TODO: Randomize soccer initial position in curriculum
        self.targets = torch.tensor(self.cfg.soccer_init_pos, dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.targets += self.scene.env_origins
        
        # Shooting direction commands
        self.target_shoot_directions = generate_target_directions(torch.tensor(self.num_envs), self.device)
        
        # Position of ball in world frame, for later computation of shooting reward
        self.ball_spawn_pos_w = torch.tensor(self.cfg.soccer_init_pos, device=self.device) + self.scene.env_origins

        # Basis vectors and identical quaternion for computing heading, attitude, etc.
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self.sim.device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.sim.device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self.sim.device).repeat(
            (self.num_envs, 1)
        )
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "shooting_reward",
            ]
        }

    def _setup_scene(self):
        # Robot
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # Soccer ball
        self.soccer = RigidObject(self.cfg.soccer)
        self.scene.rigid_objects["soccer"] = self.soccer

        # Contact sensors
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.robot_contact_sensor = RobotContactSensor(self.num_envs, self.device)

        # Ground plane
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        forces = self.action_scale * self.joint_gears * self.actions
        self.robot.set_joint_effort_target(forces, joint_ids=self._joint_dof_idx)

    def _compute_intermediate_values(self):
        self.torso_position, self.torso_rotation = self.robot.data.root_pos_w, self.robot.data.root_quat_w
        self.velocity, self.ang_velocity = self.robot.data.root_lin_vel_w, self.robot.data.root_ang_vel_w
        self.dof_pos, self.dof_vel = self.robot.data.joint_pos, self.robot.data.joint_vel

        (
            self.up_proj,
            self.heading_proj,
            self.up_vec,
            self.heading_vec,
            self.vel_loc,
            self.angvel_loc,
            self.roll,
            self.pitch,
            self.yaw,
            self.angle_to_target,
            self.dof_pos_scaled,
            self.prev_potentials,
            self.potentials,
        ) = compute_intermediate_values(
            self.targets,
            self.torso_position,
            self.torso_rotation,
            self.velocity,
            self.ang_velocity,
            self.dof_pos,
            self.robot.data.soft_joint_pos_limits[0, :, 0],
            self.robot.data.soft_joint_pos_limits[0, :, 1],
            self.inv_start_rot,
            self.basis_vec0,
            self.basis_vec1,
            self.potentials,
            self.prev_potentials,
            self.cfg.sim.dt,
        )
        
    def _compute_soccer_observations(self):
        self.soccer_position, soccer_velocity = self.soccer.data.root_pos_w, self.soccer.data.root_lin_vel_w
        foot_idx = [self.robot.data.body_names.index(x) for x in ["right_ankle_link", "left_ankle_link"]]
        foot_position = self.robot.data.body_link_pos_w[:, foot_idx, :]   # shape = (num_envs, 2, 3)
        foot_velocity = self.robot.data.body_link_lin_vel_w[:, foot_idx, :]
        foot_quat = self.robot.data.body_link_quat_w[:, foot_idx, :]
        
        (
            self.soccer_pos_torso_local,
            self.soccer_vel_torso_local,
            self.soccer_pos_foot_l_local,
            self.soccer_pos_foot_r_local,
            self.soccer_vel_foot_l_local,
            self.soccer_vel_foot_r_local,
        ) = compute_privileged_soccer_observations(
            self.soccer_position,
            soccer_velocity,
            self.torso_position,
            self.velocity,
            self.torso_rotation,
            foot_position,
            foot_velocity,
            foot_quat
        )

    def _get_observations(self) -> dict:
        # Proprioceptive observations
        obs = torch.cat(
            (
                self.torso_position[:, 2].view(-1, 1),
                self.vel_loc,
                self.angvel_loc * self.cfg.angular_velocity_scale,
                normalize_angle(self.yaw).unsqueeze(-1),
                normalize_angle(self.roll).unsqueeze(-1),
                normalize_angle(self.angle_to_target).unsqueeze(-1),
                self.up_proj.unsqueeze(-1),
                self.heading_proj.unsqueeze(-1),
                self.dof_pos_scaled,
                self.dof_vel * self.cfg.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        
        # Soccer ball observations
        self._compute_soccer_observations()
        obs = torch.cat(
            (
                obs,
                self.soccer_pos_torso_local,
                self.soccer_vel_torso_local,
                self.soccer_pos_foot_l_local,
                self.soccer_pos_foot_r_local,
                self.soccer_vel_foot_l_local,
                self.soccer_vel_foot_r_local,
                self.target_shoot_directions    # Add command shooting direction to observation
            ),
            dim=-1,
        )
        
        observations = {"policy": obs}
        return observations
    
    def _get_rewards(self) -> torch.Tensor:
        locomotion_reward = compute_locomotion_rewards(
            self.actions,
            self.reset_terminated,
            self.cfg.up_weight,
            self.cfg.heading_weight,
            self.heading_proj,
            self.up_proj,
            self.dof_vel,
            self.dof_pos_scaled,
            self.potentials,
            self.prev_potentials,
            self.cfg.actions_cost_scale,
            self.cfg.energy_cost_scale,
            self.cfg.dof_vel_scale,
            self.cfg.death_cost,
            self.cfg.alive_reward_scale,
            self.motor_effort_ratio,
        )
        soccer_shooting_reward = compute_shooting_reward(
            self.ball_spawn_pos_w,
            self.soccer_position,
            self.target_shoot_directions,
            self.cfg.shooting_direction_weight
        )
        total_reward = locomotion_reward + soccer_shooting_reward
        
        # Logging
        self._episode_sums["shooting_reward"] += soccer_shooting_reward
        
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()     # In DirectRLEnv, _get_dones is called before _get_observations
        self._compute_soccer_observations()
        
        # Terminate episode if ball is kicked and leaving it's origin for long enough
        # FIXME: This implicitly encourage robot to make the soccer move as slow as possible.
        vec = self.soccer_position - self.ball_spawn_pos_w
        vec_xy = vec[:, :2]
        vec_norm = vec_xy.norm(dim=1)
        
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = (self.torso_position[:, 2] < self.cfg.termination_height) & (vec_norm > 1.0)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        
        # Reset robot
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)     # Relay on robot being ready

        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Reset soccer ball
        soccer_root_state = self.soccer.data.default_root_state[env_ids]
        # Put soccer in front of the humanoid
        # TODO: Different soccer location in curriculum.
        offset_vec = torch.tensor(self.cfg.soccer_init_pos, device=self.device) # Put soccer slightly above the ground
        zeros_col = torch.zeros((default_root_state.shape[0], 1), device=self.device, dtype=default_root_state.dtype)
        soccer_root_state[:, :3] = torch.cat([default_root_state[:, :2], zeros_col], dim=1) + offset_vec
        self.soccer.write_root_pose_to_sim(soccer_root_state[:, :7], env_ids)
        self.soccer.write_root_velocity_to_sim(soccer_root_state[:, 7:], env_ids)
        
        # TODO: Randomize targets.
        # Targets
        to_target = self.targets[env_ids] - default_root_state[:, :3]
        to_target[:, 2] = 0.0
        self.potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.cfg.sim.dt
        
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        self._compute_intermediate_values()


@torch.jit.script
def compute_locomotion_rewards(
    actions: torch.Tensor,
    reset_terminated: torch.Tensor,
    up_weight: float,
    heading_weight: float,
    heading_proj: torch.Tensor,
    up_proj: torch.Tensor,
    dof_vel: torch.Tensor,
    dof_pos_scaled: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    dof_vel_scale: float,
    death_cost: float,
    alive_reward_scale: float,
    motor_effort_ratio: torch.Tensor,
):
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)

    # aligning up axis of robot and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(up_proj > 0.93, up_reward + up_weight, up_reward)

    # energy penalty for movement
    actions_cost = torch.sum(actions**2, dim=-1)
    electricity_cost = torch.sum(
        torch.abs(actions * dof_vel * dof_vel_scale) * motor_effort_ratio.unsqueeze(0),
        dim=-1,
    )

    # dof at limit cost
    dof_at_limit_cost = torch.sum(dof_pos_scaled > 0.98, dim=-1)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * alive_reward_scale
    progress_reward = potentials - prev_potentials

    total_reward = (
        progress_reward
        + alive_reward
        + up_reward
        + heading_reward
        - actions_cost_scale * actions_cost
        - energy_cost_scale * electricity_cost
        - dof_at_limit_cost
    )
    # adjust reward for fallen agents
    total_reward = torch.where(reset_terminated, torch.ones_like(total_reward) * death_cost, total_reward)
    return total_reward


def compute_shooting_reward(
    ball_spawn_pos: torch.Tensor,
    ball_pos: torch.Tensor,
    target_directions: torch.Tensor,
    shooting_direction_weight: float,
):
    # Reward for alignment with target shooting direction. Use the dot product of normalized planar direction vectors
    vec = ball_pos - ball_spawn_pos
    vec_xy = vec[:, :2]
    dir_xy = target_directions[:, :2]
    
    vec_norm = vec_xy.norm(dim=1)
    dir_norm = dir_xy.norm(dim=1)
    
    valid_mask = (vec_norm > 1e-1)
    
    vec_xy_normed = torch.zeros_like(vec_xy)
    dir_xy_normed = torch.zeros_like(dir_xy)
    
    vec_xy_normed[valid_mask] = vec_xy[valid_mask] / vec_norm[valid_mask].unsqueeze(1)
    dir_xy_normed[valid_mask] = dir_xy[valid_mask] / dir_norm[valid_mask].unsqueeze(1)
    
    align_reward = torch.sum(vec_xy_normed * dir_xy_normed, dim=1).clamp(-1.0, 1.0) * shooting_direction_weight
    align_reward[~valid_mask] = 0.0
    
    return align_reward


@torch.jit.script
def compute_intermediate_values(
    targets: torch.Tensor,
    torso_position: torch.Tensor,
    torso_rotation: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
    inv_start_rot: torch.Tensor,
    basis_vec0: torch.Tensor,
    basis_vec1: torch.Tensor,
    potentials: torch.Tensor,
    prev_potentials: torch.Tensor,
    dt: float,
):
    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2
    )

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position
    )

    dof_pos_scaled = torch_utils.maths.unscale(dof_pos, dof_lower_limits, dof_upper_limits)

    to_target = targets - torso_position
    to_target[:, 2] = 0.0
    prev_potentials[:] = potentials
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    return (
        up_proj,
        heading_proj,
        up_vec,
        heading_vec,
        vel_loc,
        angvel_loc,
        roll,
        pitch,
        yaw,
        angle_to_target,
        dof_pos_scaled,
        prev_potentials,
        potentials,
    )
    

@torch.jit.script
def compute_privileged_soccer_observations(
    soccer_pos_w: torch.Tensor,
    soccer_vel_w: torch.Tensor,
    torso_pos_w: torch.Tensor,
    torso_vel_w: torch.Tensor,
    torso_rot_w: torch.Tensor,
    foot_pos_w: torch.Tensor,
    foot_vel_w: torch.Tensor,
    foot_rot_w: torch.Tensor,
):
    foot_l_pos_w, foot_r_pos_w = foot_pos_w[:, 0, :], foot_pos_w[:, 1, :]
    foot_l_vel_w, foot_r_vel_w = foot_vel_w[:, 0, :], foot_vel_w[:, 1, :]
    foot_l_rot_w, foot_r_rot_w = foot_rot_w[:, 0, :], foot_rot_w[:, 1, :]
    
    # Position of the soccer ball in torso's local frame
    rel_pos_w = soccer_pos_w - torso_pos_w
    soccer_pos_torso_local = quat_rotate_inverse(torso_rot_w, rel_pos_w)
    
    # Velocity of the soccer ball in torso's local frame
    rel_vel_w = soccer_vel_w - torso_vel_w
    soccer_vel_torso_local = quat_rotate_inverse(torso_rot_w, rel_vel_w)
    
    # Velocity of the soccer ball in both feet's local frame
    rel_pos_w = soccer_pos_w - foot_l_pos_w
    soccer_pos_foot_l_local = quat_rotate_inverse(foot_l_rot_w, rel_pos_w)
    rel_pos_w = soccer_pos_w - foot_r_pos_w
    soccer_pos_foot_r_local = quat_rotate_inverse(foot_r_rot_w, rel_pos_w)
    
    # Velocity of the soccer ball in both feet's local frame
    rel_vel_w = soccer_vel_w - foot_l_vel_w
    soccer_vel_foot_l_local = quat_rotate_inverse(foot_l_rot_w, rel_vel_w)
    rel_vel_w = soccer_vel_w - foot_r_vel_w
    soccer_vel_foot_r_local = quat_rotate_inverse(foot_r_rot_w, rel_vel_w)
    
    return (
        soccer_pos_torso_local,
        soccer_vel_torso_local,
        soccer_pos_foot_l_local,
        soccer_pos_foot_r_local,
        soccer_vel_foot_l_local,
        soccer_vel_foot_r_local
    )
    