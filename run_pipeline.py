import argparse
from utils_pipeline import VideoProcessingPipeline, PipelineConfig
# from temp_utils import VideoProcessingPipeline, iterate_video_frames
# from preprocessing_utils import PipelineConfig


# TODO
#  - Handle cases:
#    -> Extra set of AprilTags before start of trial blocks (eg. P02-A1)
#       >> Probably can just add boolean to config
#       >> Only important if there are no high diode values
#    -> Splice video data when blocks are split into multiple videos (eg. P02-A1)
#       >> Does the gaze data world_index get reset when there are multiple videos/gaze data files? (eg. P02-A1)
#       >> Can I just splice the videos together as a new video, then continue identifying blocks normally?
#       >> Probably need to base this on the diode data. Should I automate this or do it manually?
#    -> Splice diode data when split into multiple files (eg. P02-A1):
#       >> Probably can simply concatenate the DFs and then process after that
#       >> Maybe even just manually stack the csv data, if it's just a couple files
#       >> When a block was unfinished, it gets restarted. How to identify this?
#       >> P05-A2 is almost good, but there's one missing high diode value, indicating a block end.
#           How many more cases are there like this, where there is something very specific wrong?
#           (This one got restarted. Can use to figure out how to handle above condition. Are all
#           restarted blocks missing the high diode value?)
#       >> Maybe when len(first_apriltag_inds_time) > 10, check for len(event_times) < 40
#  - Organize different configs to easily switch between them
#    -> Should be able to quickly run and refer to them
#    -> Probably want to put PipelineConfig class into pipeline_config.py
#    -> Configs should be organized by the type of processing the session requires (i.e. which cases from above)
#  - Start doing post-processing:
#    -> Plot hand position, velocity, acceleration (use cubic spline for smoothing) during trials
#       >> Include standard deviation envelopes
#       >> Align to event onset times
#       >> Separate by trial condition

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
    "max_num_hands": 2,
    "model_complexity": 1,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}
# APRILTAG_THRESHOLD = 50
# EVENT_THRESHOLD = 80
# BLOCK_THRESHOLD = 200
APRILTAG_THRESHOLD = 50
EVENT_THRESHOLD = 80
BLOCK_THRESHOLD = None

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
    # parser.add_argument(
    #     "diode_suffix",
    #     type=str
    # )
    # args = parser.parse_args()
    # PARTICIPANT_ID = 17
    # SESSION_ID = "A1"
    # DIODE_SUFFIX = "01"
    # VIDEO_SUFFIX = ""
    PARTICIPANT_ID = 2
    SESSION_ID = "A1"
    DIODE_SUFFIX = "2"
    VIDEO_SUFFIX = "_01"

    pipeline_config = PipelineConfig(
        # participant_id=args.participant_id,
        # session_id=args.session_id,
        participant_id=PARTICIPANT_ID,
        session_id=SESSION_ID,
        diode_suffix=DIODE_SUFFIX,
        video_suffix=VIDEO_SUFFIX,
        apriltag_threshold=APRILTAG_THRESHOLD,
        event_threshold=EVENT_THRESHOLD,
        block_threshold=BLOCK_THRESHOLD,
        tracked_hand_landmarks=hand_landmarks,
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
