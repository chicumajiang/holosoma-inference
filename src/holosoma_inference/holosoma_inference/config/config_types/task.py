"""Task configuration types for holosoma_inference."""

from __future__ import annotations

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class TaskConfig:
    """Task execution configuration for policy inference."""

    model_path: str | list[str]
    """Path to ONNX model(s). Supports local paths and wandb:// URIs. Required field."""

    rl_rate: float = 50
    """Policy inference rate in Hz."""

    policy_action_scale: float = 0.25
    """Scaling factor applied to policy actions."""

    use_phase: bool = True
    """Whether to use gait phase observations."""

    gait_period: float = 1.0
    """Gait cycle period in seconds."""

    domain_id: int = 0
    """DDS domain ID for communication."""

    interface: str = "lo"
    """Network interface name."""

    use_joystick: bool = False
    """Enable joystick control input."""

    joystick_type: str = "xbox"
    """Joystick type."""

    joystick_device: int = 0
    """Joystick device index."""

    use_sim_time: bool = False
    """Use synchronized simulation time for WBT policies."""

    wandb_download_dir: str = "/tmp"
    """Directory for downloading W&B checkpoints."""

    # Deprecation candidates:
    desired_base_height: float = 0.75
    """Target base height in meters."""

    residual_upper_body_action: bool = False
    """Whether to use residual control for upper body."""

    use_ros: bool = False
    """Use ROS2 for rate limiting."""

    deployment_mode: str = "sim"
    """Deployment mode: sim, read-only, dry-run, or armed."""

    deployment_write_enabled: bool = True
    """Whether the communication layer may publish low-level commands."""

    deployment_configure_control_mode: bool = True
    """Whether the communication layer may change the robot control mode."""

    mocap_timeout_seconds: float = 0.25
    """Maximum age of a real-world mocap message used for control."""

    deployment_max_command_step_rad: float = 0.02
    """Maximum per-cycle target position change in real deployment modes."""

    deployment_max_target_error_rad: float = 0.35
    """Maximum target error allowed while armed."""

    goal_pos_w: tuple[float, float, float] = (0.02310966, -1.098395, 2.0000)
    """World-frame goal position (e.g. basketball hoop) for 160-dim WBT policy."""
