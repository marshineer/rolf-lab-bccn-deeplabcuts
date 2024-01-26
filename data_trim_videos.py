"""The purpose of this script is to trim videos, to remove sets of AprilTags or blocks of trials that
do not appear in the diode data. It works by making a new video and saving all frames past a certain
point to that new video. The old video is renamed, just in case there is an error during the process."""

import os
import cv2
import time
import numpy as np
from utils import load_video_mp4, load_video_time, load_diode_data


trim_data = [
    # ("P03", "A2", 10000, -1),
    # ("P08", "B1", 593, 24.),
    # ("P09", "B1", 550, -1),
    # ("P09", "B2", 550, -1),
    # ("P11", "A1", 1500, 50.),
]

for participant_id, session_id, video_frame_0, diode_time_0 in trim_data:
    # Determine the file path
    fpath = f"data/pipeline_data/{participant_id}/{session_id}/"
    print(f"Processing {participant_id}-{session_id}")

    # Load the video capture and extract the metadata
    vcap = load_video_mp4(participant_id, session_id)
    n_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcap.get(cv2.CAP_PROP_FPS)
    codec = int(vcap.get(cv2.CAP_PROP_FOURCC))

    # Initialize a new video
    temp_video_name = f"{participant_id}_{session_id}_temp.mp4"
    new_video = cv2.VideoWriter(os.path.join(fpath, temp_video_name), codec, fps, (width, height))
    video_frame_cnt = 0
    start_time = time.time()
    while vcap.isOpened():
        # Read the next frame
        ret, frame = vcap.read()
        if not ret:
            break
        if video_frame_cnt % 10000 == 0:
            print(f"Current frame {video_frame_cnt}/{n_frames}")
        video_frame_cnt += 1
        if video_frame_0 <= video_frame_cnt:
            new_video.write(frame)
    print(f"Took {time.time() - start_time:0.2f} seconds to trim video\n")

    # Close the video
    vcap.release()
    cv2.destroyAllWindows()

    # Save and rename the files
    existing_video_name = f"{participant_id}_{session_id}.mp4"
    os.rename(os.path.join(fpath, existing_video_name), os.path.join(fpath, f"{participant_id}_{session_id}_old.mp4"))
    os.rename(os.path.join(fpath, temp_video_name), os.path.join(fpath, existing_video_name))

    # Load and trim the video time (derived from the gaze data)
    print("Trimming video time")
    video_time = load_video_time(participant_id, session_id)[video_frame_0:]
    video_time -= video_time[0]
    os.rename(os.path.join(fpath, f"{participant_id}_{session_id}_video_time.csv"),
              os.path.join(fpath, f"{participant_id}_{session_id}_video_time_old.csv"))
    np.savetxt(os.path.join(fpath, f"{participant_id}_{session_id}_video_time.csv"), video_time, delimiter=",")

    # Load and trim the diode data
    if diode_time_0 < 0:
        continue
    else:
        print("Trimming diode data")
        diode_df = load_diode_data(participant_id, session_id)
        diode_df_new = diode_df[diode_df.time >= diode_time_0]
        diode_df_new.time -= diode_df_new.time.iloc[0]
        os.rename(os.path.join(fpath, f"{participant_id}_{session_id}_diode_sensor.csv"),
                  os.path.join(fpath, f"{participant_id}_{session_id}_diode_sensor_old.csv"))
        diode_df_new.to_csv(os.path.join(fpath, f"{participant_id}_{session_id}_diode_sensor.csv"), index=False)
