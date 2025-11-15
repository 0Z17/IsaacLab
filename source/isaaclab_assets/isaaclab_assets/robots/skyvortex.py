import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os
import torch
from dataclasses import dataclass, field
from typing import List

isaaclab_path = os.environ.get('ISAACLAB_PATH')

if isaaclab_path:
    print(f"setting ISAACLAB_PATH to {isaaclab_path}")
else:
    print("error: ISAACLAB_PATH not set")

##
# Configuration
##

@dataclass
class SkyvortexCfg(ArticulationCfg):
    """Configuration for the SkyVortex robot."""
    rotor_z_axes: List[List[float]] = field(default_factory=lambda: [
        [-0.249999, 0.433009,   0.866028],
        [0.499998,  0.0,        0.866027],
        [-0.249999, -0.433009,  0.866028],
        [-0.249999, 0.433009,   0.866028],
        [0.499998,  0.0,        0.866027],
        [-0.25,     -0.433013,  0.866025],
    ])
    """The z-axes of the rotors in the base frame."""
    
    rotor_factors: List[float] = field(default_factory=lambda: [
        0.06, -0.06, 0.06, -0.06, 0.06, -0.06])
    """The torque factors of the rotors.""" 


SKYVORTEX_CFG = SkyvortexCfg(
    # prim_path="{ENV_REGEX_NS}/skyvortex",
    # spawn=sim_utils.UrdfFileCfg(asset_path=f"{isaaclab_path}/robots/skyvortex/skyvortex.urdf", fix_base=False,
    #                             joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(gains=sim_utils.UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(natural_frequency=100.0, damping_ratio=0.005))),
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{isaaclab_path}/robots/skyvortex/skyvortex/skyvortex.usd",
        usd_path=f"{isaaclab_path}/robots/skyvortex/skyvortex_usd/skyvortex.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            ),
        # copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.2), joint_pos={"operator_1_joint": 0.0}
    ),
    actuators={
            "operator_1_actuator": DCMotorCfg(
                joint_names_expr=["operator_1_joint"],
                effort_limit=33.5,
                saturation_effort=33.5,
                velocity_limit=21.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
    },
)
"""Configuration for the SkyVortex robot."""