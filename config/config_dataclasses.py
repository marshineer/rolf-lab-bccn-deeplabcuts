from dataclasses import dataclass, field


@dataclass()
class PipelineConfig:
    save_data: bool
    show_video: bool
    visual_plot_check: bool
    overwrite_data: bool
    apriltag_kwargs: dict[str, float]
    mediapipe_kwargs: dict[str, int | float]
    tracked_hand_landmarks: dict[str, int]


@dataclass()
class SessionConfig:
    participant_id: str
    session_id: str
    n_blocks: int
    diode_threshold: int
    separator_threshold: int | None
    skip_valid_blocks: list[int] = field(default_factory=list)
    extra_apriltag_blocks: list[int] = field(default_factory=list)
    apparatus_tag_ids: list[int] = field(default_factory=lambda: [40, 30, 10])


@dataclass
class PostprocessingConfig:
    smoothing_factor: float = 3000
    missing_blocks: list[int] = field(default_factory=list)
    skip_blocks: list[int] = field(default_factory=list)
    skip_last_trial: list[bool] = field(default_factory=list)


