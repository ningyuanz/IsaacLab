import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import DCMotorCfg


BABY_ALPHA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/ningyuan/codes/IsaacLab/source/isaaclab_assets/data/Robots/weilan/baby_alpha.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.26),    # (0.0, 0.0., 0.26) to just above the ground
        joint_pos={
            "(front|hind)_(left|right)_abad_joint": 0.0,  # abad
            "(front|hind)_(left|right)_hip_joint": -0.52,  # hip
            "(front|hind)_(left|right)_knee_joint": 1.32,  # knee
            # "(front|hind)_(left|right)_foot_joint": 0.0,  # foots
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                "(front|hind)_(left|right)_abad_joint", 
                "(front|hind)_(left|right)_hip_joint", 
                "(front|hind)_(left|right)_knee_joint"
                ],
            effort_limit=9.1,
            saturation_effort=9.1,
            velocity_limit=16.29,    # rad/s
            stiffness=7.0,
            damping=0.15,
            friction=0.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)