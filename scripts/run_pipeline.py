"""This is the second script that must be run. It extracts the block data from the diode files,
then uses this information to split the session videos into individual blocks. Markers are added
to the block videos, so that a visual inspection of the event onset timing, AprilTag detection
and hand tracking can be performed before further processing the data. The primary output of
this step is the hand tracking data, which is further processed in the next step."""

# import os
from utils_pipeline import VideoProcessingPipeline, load_session_data
from utils.data_loading import get_files_containing, load_pipeline_config, load_session_config


if __name__ == '__main__':
    # Get the pipeline config files
    pipeline_config = load_pipeline_config()

    # Get the config files and iterate through
    config_paths, _ = get_files_containing("data/pipeline_data", "config.json")
    for fpath in config_paths:
        # Load the session config file
        session_config = load_session_config(fpath)

        # Check if the data for this participant session is fully processed
        session_data = load_session_data(session_config.participant_id, session_config.session_id)
        # _, block_files = get_files_containing(os.path.join(fpath, "block_videos"), ".mp4")
        if session_data is not None:
            if len(session_data.hand_landmark_pos_abs) == session_config.n_blocks:
                continue

        # Otherwise, continue processing
        print("\n\n\n#############################################################\n"
              "######################## New Session ########################\n"
              f"############## Participant: {session_config.participant_id} - "
              f"Sesssion: {session_config.session_id} ##############\n"
              "#############################################################\n")

        # Instantiate the pipeline
        pipeline = VideoProcessingPipeline(pipeline_config, session_config)

        # Run the pipeline
        data = pipeline.iterate_video_frames()
