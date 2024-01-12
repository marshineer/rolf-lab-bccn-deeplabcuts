import os
import json
from utils_pipeline import VideoProcessingPipeline, PipelineConfig, SessionConfig
from utils import get_files_containing


# TODO
#  - Start doing post-processing:
#    -> Plot hand position, velocity, acceleration (use cubic spline for smoothing) during trials
#       >> Include standard deviation envelopes
#       >> Align to event onset times
#       >> Separate by trial condition

# TODO (DONE):
#  - Organize different configs to easily switch between them
#    -> Should be able to quickly run and refer to them
#    -> Probably want to put PipelineConfig class into pipeline_config.json
#    -> Configs should be organized by the type of processing the session requires
#  - The reference AprilTag cannot always be the top left
#    -> It's cut off in some frames (need to use bottom left in P13-A1, for example)
#    -> This is calculated as the AprilTag that is visible in the most video frames
#  - Change participant_id from int to str
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


if __name__ == '__main__':
    with open("pipeline_config.json", "r") as fd:
        pipeline_settings = json.load(fd)
        print(pipeline_settings)
        pipeline_config = PipelineConfig(**pipeline_settings)

    # Get the config files and iterate through
    config_paths, _ = get_files_containing("data/pipeline_data", "config.json")
    for fpath in config_paths:
        # Load the session config file
        with open(os.path.join(fpath, "config.json"), "r") as fd:
            session_settings = json.load(fd)
            print(session_settings)
            session_config = SessionConfig(**session_settings)
        _, block_video_files = get_files_containing(os.path.join(fpath, "block_videos"), "block")
        if len(block_video_files) == session_config.n_blocks:
            continue
        print("\n\n\n#############################################################\n"
              "######################## New Session ########################\n"
              f"############## Participant: {session_config.participant_id} - "
              f"Sesssion: {session_config.session_id} ##############\n"
              "#############################################################\n")

        # Instantiate the pipeline
        pipeline = VideoProcessingPipeline(pipeline_config, session_config)

        # Run the pipeline and save the results
        data = pipeline.iterate_video_frames()

        # with open(fpath, "wb") as f:
        #     pickle.dump(data, f)
        # with open('filename.pickle', 'rb') as handle:
        #     b = pickle.load(handle)
        # data.relative_hand_positions(True)
