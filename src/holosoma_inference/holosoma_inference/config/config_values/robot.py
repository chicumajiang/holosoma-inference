"""Default robot configurations for holosoma_inference.

This module provides pre-configured robot hardware and control parameters
for different robot types.
"""

from __future__ import annotations

from holosoma_inference.config.config_types.robot import RobotConfig

# =============================================================================
# G1 Robot Config
# =============================================================================

# fmt: off

g1_29dof = RobotConfig(
    # Identity
    robot_type="g1_29dof",
    robot="g1",

    # SDK Configuration
    sdk_type="unitree",
    motor_type="serial",
    message_type="HG",
    use_sensor=False,

    # Dimensions
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=14,

    # Default Positions
    # default_dof_angles=(
    #     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
    #     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    #     0.0, 0.0, 0.0,  # waist
    #     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
    #     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    # ),
#  shooting policy的reference的第一帧
    # default_dof_angles = (
    # -0.105, 0.103, 0.212, 0.191, -0.0466, -0.0331,
    # -0.102, -0.159, -0.335, 0.202, -0.042, 0.0422,
    # 0.00445, 0.0047, 0.0607,
    # -1.71, 0.0512, -0.303, 1.68, -0.949, -0.678, 0.127,
    # -1.53, -0.0222, 0.0319, 1.31, 0.39, -0.758, -0.634,
    # ),
    #basketball 013
   default_dof_angles=(
-0.17104841768741608,
        0.09444943070411682,
        0.26210758090019226,
        0.1878250539302826,
        -0.10962207615375519,
        0.0001973411563085392,
        -0.16742774844169617,
        -0.18660227954387665,
        -0.26155975461006165,
        0.1700589507818222,
        0.00045667431550100446,
        0.0711684599518776,
        0.03850064426660538,
        -0.009034798480570316,
        0.09702195972204208,
        -1.183659315109253,
        0.20094215869903564,
        -0.4117770791053772,
        1.0605320930480957,
        -0.3890945613384247,
        -0.10555541515350342,
        0.34226590394973755,
        -1.1783448457717896,
        -0.17074644565582275,
        0.23504137992858887,
        1.054811954498291,
        0.12367469817399979,
        -0.09968511760234833,
        -0.5193578600883484
       ),
#   C001
    # default_dof_angles=(
    #     -0.010286951437592506,
    #     0.09252329915761948,
    #     0.180194690823555,
    #     -0.08726699650287628,
    #     0.11004013568162918,
    #     -0.0845244973897934,
    #     -0.007844709791243076,
    #     -0.07193803787231445,
    #     -0.24306707084178925,
    #     -0.0871921107172966,
    #     0.14837731420993805,
    #     0.05278982222080231,
    #     0.026394356042146683,
    #     0.032452281564474106,
    #     0.15174975991249084,
    #     -1.4699207544326782,
    #     0.10828011482954025,
    #     -0.8348573446273804,
    #     1.6433262825012207,
    #     -1.0889233350753784,
    #     -0.026409432291984558,
    #     0.2116124927997589,
    #     -1.3745002746582031,
    #     -0.1960909366607666,
    #     0.6120838522911072,
    #     1.5883736610412598,
    #     0.975861668586731,
    #     -0.11367375403642654,
    #     -0.13032971322536469
    #    ),
# recovery policy的reference的第一帧
#         default_dof_angles=(
# 0.41115668416023254,
#         0.06297832727432251,
#         0.31799548864364624,
#         -0.08726699650287628,
#         -0.0576867051422596,
#         -0.06092328950762749,
#         0.408286452293396,
#         -0.15160797536373138,
#         -0.3365215063095093,
#         -0.08726699650287628,
#         0.026435405015945435,
#         0.09956204891204834,
#         0.03294156864285469,
#         -0.029027780517935753,
#         0.23954658210277557,
#         -2.313858985900879,
#         0.5601935982704163,
#         -1.0042846202850342,
#         1.365410566329956,
#         0.2326342612504959,
#         0.03872094675898552,
#         -0.2490057498216629,
#         -2.2163236141204834,
#         -0.4963211417198181,
#         0.8253850340843201,
#         1.2174348831176758,
#         -0.7044774889945984,
#         0.09724337607622147,
#         0.0812341719865799
#         ),
    #default_dof_angles=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
    # default_motor_angles=(
    #     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left leg
    #     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right leg
    #     0.0, 0.0, 0.0,  # waist
    #     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # left arm
    #     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,  # right arm
    # ),
    #default_motor_angles=(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0),
    # default_motor_angles = (
    # -0.105, 0.103, 0.212, 0.191, -0.0466, -0.0331,
    # -0.102, -0.159, -0.335, 0.202, -0.042, 0.0422,
    # 0.00445, 0.0047, 0.0607,
    # -1.71, 0.0512, -0.303, 1.68, -0.949, -0.678, 0.127,
    # -1.53, -0.0222, 0.0319, 1.31, 0.39, -0.758, -0.634,
    # ),
    # basketball 013
   default_motor_angles=(
-0.17104841768741608,
        0.09444943070411682,
        0.26210758090019226,
        0.1878250539302826,
        -0.10962207615375519,
        0.0001973411563085392,
        -0.16742774844169617,
        -0.18660227954387665,
        -0.26155975461006165,
        0.1700589507818222,
        0.00045667431550100446,
        0.0711684599518776,
        0.03850064426660538,
        -0.009034798480570316,
        0.09702195972204208,
        -1.183659315109253,
        0.20094215869903564,
        -0.4117770791053772,
        1.0605320930480957,
        -0.3890945613384247,
        -0.10555541515350342,
        0.34226590394973755,
        -1.1783448457717896,
        -0.17074644565582275,
        0.23504137992858887,
        1.054811954498291,
        0.12367469817399979,
        -0.09968511760234833,
        -0.5193578600883484
    ),
#       C001
#    default_motor_angles=(
#         -0.010286951437592506,
#         0.09252329915761948,
#         0.180194690823555,
#         -0.08726699650287628,
#         0.11004013568162918,
#         -0.0845244973897934,
#         -0.007844709791243076,
#         -0.07193803787231445,
#         -0.24306707084178925,
#         -0.0871921107172966,
#         0.14837731420993805,
#         0.05278982222080231,
#         0.026394356042146683,
#         0.032452281564474106,
#         0.15174975991249084,
#         -1.4699207544326782,
#         0.10828011482954025,
#         -0.8348573446273804,
#         1.6433262825012207,
#         -1.0889233350753784,
#         -0.026409432291984558,
#         0.2116124927997589,
#         -1.3745002746582031,
#         -0.1960909366607666,
#         0.6120838522911072,
#         1.5883736610412598,
#         0.975861668586731,
#         -0.11367375403642654,
#         -0.13032971322536469
#     ),
# recovery policy的reference的第一帧
#     default_motor_angles=(
# 0.41115668416023254,
#         0.06297832727432251,
#         0.31799548864364624,
#         -0.08726699650287628,
#         -0.0576867051422596,
#         -0.06092328950762749,
#         0.408286452293396,
#         -0.15160797536373138,
#         -0.3365215063095093,
#         -0.08726699650287628,
#         0.026435405015945435,
#         0.09956204891204834,
#         0.03294156864285469,
#         -0.029027780517935753,
#         0.23954658210277557,
#         -2.313858985900879,
#         0.5601935982704163,
#         -1.0042846202850342,
#         1.365410566329956,
#         0.2326342612504959,
#         0.03872094675898552,
#         -0.2490057498216629,
#         -2.2163236141204834,
#         -0.4963211417198181,
#         0.8253850340843201,
#         1.2174348831176758,
#         -0.7044774889945984,
#         0.09724337607622147,
#         0.0812341719865799
#     ),

    # Mappings
    motor2joint=tuple(range(29)),  # Identity mapping
    joint2motor=tuple(range(29)),  # Identity mapping
    dof_names=(
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_upper_body=(
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ),
    dof_names_lower_body=(
        "left_hip_yaw_joint", "left_hip_roll_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_yaw_joint", "right_hip_roll_joint", "right_hip_pitch_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    ),

    # Link Names
    torso_link_name="torso_link",
    left_hand_link_name="left_rubber_hand",
    right_hand_link_name="right_rubber_hand",

    # Unitree-Specific Constants
    unitree_legged_const={
        "HIGHLEVEL": 238,
        "LOWLEVEL": 255,
        "TRIGERLEVEL": 240,
        "PosStopF": 2146000000.0,
        "VelStopF": 16000.0,
        "MODE_MACHINE": 5,
        "MODE_PR": 0,
    },
    weak_motor_joint_index={
        "left_hip_yaw_joint": 0, "left_hip_roll_joint": 1, "left_hip_pitch_joint": 2,
        "left_knee_joint": 3, "left_ankle_pitch_joint": 4, "left_ankle_roll_joint": 5,
        "right_hip_yaw_joint": 6, "right_hip_roll_joint": 7, "right_hip_pitch_joint": 8,
        "right_knee_joint": 9, "right_ankle_pitch_joint": 10, "right_ankle_roll_joint": 11,
        "waist_yaw_joint": 12, "waist_roll_joint": 13, "waist_pitch_joint": 14,
        "left_shoulder_pitch_joint": 15, "left_shoulder_roll_joint": 16,
        "left_shoulder_yaw_joint": 17, "left_elbow_joint": 18,
        "left_wrist_roll_joint": 19, "left_wrist_pitch_joint": 20, "left_wrist_yaw_joint": 21,
        "right_shoulder_pitch_joint": 22, "right_shoulder_roll_joint": 23,
        "right_shoulder_yaw_joint": 24, "right_elbow_joint": 25,
        "right_wrist_roll_joint": 26, "right_wrist_pitch_joint": 27, "right_wrist_yaw_joint": 28,
    },
    motion={"body_name_ref": ["torso_link"]},
)


# =============================================================================
# T1 Robot Config
# =============================================================================

t1_29dof = RobotConfig(
    # Identity
    robot_type="t1_29dof",
    robot="t1",

    # SDK Configuration
    sdk_type="booster",  # T1 uses booster SDK
    motor_type="serial",
    message_type="HG",  # Using default
    use_sensor=False,

    # Dimensions
    num_motors=29,
    num_joints=29,
    num_upper_body_joints=16,  # T1 has 16 upper body joints (includes head)

    # Default Positions
    default_dof_angles=(
        0.0, 0.0,  # head (yaw, pitch)
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,  # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),
    default_motor_angles=(
        0.0, 0.0,  # head
        0.2, -1.35, 0.0, -0.5, 0.0, 0.0, 0.0,  # left arm
        0.2, 1.35, 0.0, 0.5, 0.0, 0.0, 0.0,  # right arm
        0.0,  # waist
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
        -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
    ),

    # Mappings
    motor2joint=tuple(range(29)),  # Identity mapping
    joint2motor=tuple(range(29)),  # Identity mapping
    dof_names=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),
    dof_names_upper_body=(
        "AAHead_yaw", "Head_pitch",
        "Left_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
        "Left_Wrist_Pitch", "Left_Wrist_Yaw", "Left_Hand_Roll",
        "Right_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
        "Right_Wrist_Pitch", "Right_Wrist_Yaw", "Right_Hand_Roll",
    ),
    dof_names_lower_body=(
        "Waist",
        "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
        "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
        "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
        "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
    ),

    # Link Names
    torso_link_name="Trunk",
    left_hand_link_name=None,
    right_hand_link_name=None,

    # Set unitree-specific values to `None`
    unitree_legged_const=None,
    weak_motor_joint_index=None,
    motion=None,
)


# =============================================================================
# Default Configurations Dictionary
# =============================================================================

DEFAULTS = {
    "g1-29dof": g1_29dof,
    "t1-29dof": t1_29dof,
}
"""Dictionary of all available robot configurations.

Keys use hyphen-case naming convention for CLI compatibility.
"""
