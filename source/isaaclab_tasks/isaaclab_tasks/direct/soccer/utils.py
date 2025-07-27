import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg


def generate_target_directions(num_envs, device):
    # Sample direction uniformly in [0, 2 * pi]
    # Only consider planar direction for now
    # TODO: Add z in curriculum.
    theta = torch.rand(num_envs, device=device) * 2 * torch.pi
    xy = torch.stack((torch.cos(theta), torch.sin(theta)), dim=1)  # shape [num_envs, 2]
    z = torch.zeros(num_envs, 1, device=device)  # shape [num_envs, 1]
    
    target_directions = torch.cat((xy, z), dim=1)
    return target_directions


class RobotContactSensor:
    """ A wrapper class of built-in ContactSensor to detect collision between robot and the ball. 
    TODO: Generalize to any two bodies.
    """
    H1_LINK_NAMES = [
        'pelvis', 
        'left_hip_yaw_link', 
        'right_hip_yaw_link', 
        'torso_link', 
        'left_hip_roll_link', 
        'right_hip_roll_link', 
        'left_shoulder_pitch_link', 
        'right_shoulder_pitch_link', 
        'left_hip_pitch_link', 
        'right_hip_pitch_link', 
        'left_shoulder_roll_link', 
        'right_shoulder_roll_link', 
        'left_knee_link', 
        'right_knee_link', 
        'left_shoulder_yaw_link', 
        'right_shoulder_yaw_link', 
        'left_ankle_link', 
        'right_ankle_link', 
        'left_elbow_link', 
        'right_elbow_link'
    ]
    BALL_PRIM = "/World/envs/env_.*/Soccer"
    
    def __init__(self, num_envs, device):
        self.num_envs, self.device = num_envs, device
        
        # Create ContactSensorCfg between each robot's link and the soccer ball
        self.sensor_cfgs = [
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{link_name}",
                filter_prim_paths_expr=["/World/envs/env_.*/Soccer"],
                history_length=0,   # No history for basic detection
                update_period=0.005, 
                track_air_time=False
            )
            for link_name in self.H1_LINK_NAMES
        ]
        self.sensors: list[ContactSensor] = [ContactSensor(cfg) for cfg in self.sensor_cfgs]
        
    def has_any_contact(self, ):
        """
        Returns a boolean tensor indicating which environments have any robot link in contact with the ball.
        
        Returns:
            torch.Tensor: Shape (num_envs,), True where any link contacts the ball, False otherwise.
        """
        contacts = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i, sensor in enumerate(self.sensors):
            assert sensor.data.force_matrix_w != None, "force_matrix_w of one contact sensor is None."
            forces = sensor.data.force_matrix_w[:, 0, 0, :]
            force_norms = torch.norm(forces, dim=-1)
            contacts = torch.logical_or(contacts, force_norms > 1e-3)
        return contacts
                
    
    