"""This is used to find differences between the gaze and diode data end times.
I needed it for debugging at one point, but it's probably not so useful now."""

import os
import cv2
import pandas as pd
from utils.data_loading import get_files_containing, load_video_time, load_video_mp4, load_diode_data


# Check that video, gaze and diode data times match
fpaths, _ = get_files_containing('../data/pipeline_data', 'video_time.csv')
for fpath in fpaths:
    participant_id, session_id = fpath.split("/")[-2:]
    vcap = load_video_mp4(participant_id, session_id)
    n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_time = load_video_time(participant_id, session_id)
    # print(f"Number of frames: {n_frames}")
    if abs(n_frames - len(video_time)) > 20:
        print(f"Problem with video time {participant_id}-{session_id}"
              f"\nVideo frames: {n_frames}\nTime frames: {len(video_time)}\n")
    vcap.release()
    cv2.destroyAllWindows()
    diode_data = load_diode_data(participant_id, session_id)
    if abs(video_time[-1] - diode_data.time.iloc[-1]) > 20:
        print(f"Difference in gaze and diode data times {participant_id}-{session_id}"
              f"\nGaze end time: {video_time[-1]}\nDiode end time: {diode_data.time.iloc[-1]}\n")

# Check that the gaze frames start at zero
fpaths, files = get_files_containing('../data/original_data', 'gaze_positions_on_surface')
for fpath, file in zip(fpaths, files):
    gaze_df = pd.read_csv(os.path.join(fpath, file), usecols=['world_timestamp', 'world_index'])
    column_types = {'time': float, 'video_frame': int}
    gaze_df.columns = ['time', 'video_frame']
    gaze_df.time = gaze_df.time - gaze_df.time.iloc[0]
    gaze_df = gaze_df.drop_duplicates('video_frame').reset_index(drop=True)
    first_frame = gaze_df.video_frame.iloc[0]
    if first_frame != 0:
        participant_id, session_id = fpath.split("/")[-2:]
        print(f"Non-zero first world_index value in gaze data {participant_id}-{session_id}: frame #{first_frame}")
