import argparse
from utils_pipeline import VideoProcessingPipeline, PipelineConfig
# from temp_utils import VideoProcessingPipeline, iterate_video_frames
# from preprocessing_utils import PipelineConfig


# Landmark and kwarg references
# https://pupil-apriltags.readthedocs.io/en/stable/api.html
# https://juliarobotics.org/AprilTags.jl/latest/
# https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
apriltag_kwargs = {
    "quad_decimate": 2.0,
    "decode_sharpening": 0.25,
}
hand_landmarks = {
    "Index Finger Tip": 8,
    "Index Finger Base": 5,
    "Wrist": 0,
}
mediapipe_kwargs = {
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}
MAX_HANDS = 2

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "participant_id",
    #     type=int
    # )
    # parser.add_argument(
    #     "session_id",
    #     type=int
    # )
    # args = parser.parse_args()
    PARTICIPANT_ID = 17
    SESSION_ID = "A1"

    pipeline_config = PipelineConfig(
        # participant_id=args.participant_id,
        # session_id=args.session_id,
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        tracked_hand_landmarks=hand_landmarks,
        max_hands=MAX_HANDS,
        save_video=True,
        show_video=False,
        visual_plot_check=True,
        apriltag_kwargs=apriltag_kwargs,
        mediapipe_kwargs=mediapipe_kwargs,
    )

    # Instantiate the pipeline
    pipeline = VideoProcessingPipeline(pipeline_config)

    # Run the pipeline
    data = pipeline.iterate_video_frames()
    data.relative_hand_positions(True)
