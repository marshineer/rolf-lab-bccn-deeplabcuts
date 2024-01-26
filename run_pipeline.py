from utils_pipeline import VideoProcessingPipeline, load_session_data
from utils import get_files_containing, load_pipeline_config, load_session_config


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
