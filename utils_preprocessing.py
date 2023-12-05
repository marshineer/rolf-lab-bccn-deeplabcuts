import numpy as np
import pandas as pd


def preprocess_diode_data(participant_id: int, session_id: str) -> pd.DataFrame:
    """Preprocesses the light diode sensor data.

    Parameters
        participant_id (int): unique participant identifier
        session_id (int): session identifier
        show_plots (bool): whether to display plots as visual checks on the process

    Returns
        diode_df (pd.DataFrame): processed diode data
    """

    diode_path = f"data/participants/P{participant_id}/{session_id}/{participant_id}_{session_id}_01_light.csv"
    diode_df = pd.read_csv(diode_path, usecols=[' timestamp', ' light_value'])
    diode_df.columns = ['time', 'light_value']
    diode_df.time = diode_df.time - diode_df.time.iloc[0]
    return diode_df


def preprocess_gaze_data(participant_id: int, session_id: str, plot_result: bool) -> np.ndarray:
    """Preprocesses the gaze time data.

    The gaze data world_index corresponds to the video frame number, so the gaze data time
    can be used to determine the timestamps of the video frames.

    Parameters
        participant_id (int): unique participant identifier
        session_id (int): session identifier
        plot_result (bool): whether to display plots as visual checks on the process

    Returns
        (np.ndarray): video frame timestamp array
    """

    gaze_path = f"data/participants/P{participant_id}/{session_id}/gaze_positions_on_surface_Phone.csv"
    gaze_df = pd.read_csv(gaze_path, usecols=['world_timestamp', 'world_index'])
    gaze_df.columns = ['time', 'video_frame']
    gaze_df = gaze_df.drop_duplicates('video_frame').reset_index(drop=True)
    gaze_df = add_missing_gaze_rows(gaze_df, plot_result)
    return gaze_df.time.to_numpy('float', copy=True)


def add_missing_gaze_rows(gaze_data: pd.DataFrame, plot_result: bool) -> pd.DataFrame:
    """Adds missing time frames to gaze data.

    Interpolates the times of missing video frames. If the number of frames missing is less
    than the number of time steps, the missing time steps are left at the end of the frame gap.
    These time steps are handled by another method.

    Note: The gaze data is related to the mp4 video data by the video_frame column. This data
    is important for determining the video time, since the gaze times should be accurate, and
    the video frames have no timestamp associated with them. The only way to calculate the video
    time from an mp4 is by counting the frames, which is unreliable if frames are dropped.

    This method assumes that each video frame in the actual mp4 file corresponds to a value in the
    'video_frame' column of the gaze data. Therefore, the values in this column should be continuous.

    Paramters
        gaze_data (pd.DataFrame): original gaze data with columns [time, video_frame]
        plot_result (bool): whether to plot the results as a manual check of the interpolation

    Returns
        gaze_data_updated (pd.DataFrame): gaze data updated with missing video frames
    """

    # Find the median time step in the data
    dt_median = gaze_data.time.diff().median()
    print(f"Median dt: {dt_median}")

    # Interpolate missing frames in gaze data
    gaze_data_updated = gaze_data.copy()
    gaze_data_old = gaze_data.copy()
    gap_inds_old = list(gaze_data_updated.index[gaze_data_updated.video_frame.diff() > 1])
    gap_inds_updated = []
    gap_lens = []

    # While there are still missing rows in the gaze data
    while list(gaze_data_updated.index[gaze_data_updated.video_frame.diff() > 1]):
        # Calculate frame indices
        first_frame_ind = gaze_data_updated.index[gaze_data_updated.video_frame.diff() > 1][0] - 1
        frame_prev = gaze_data_updated.iloc[first_frame_ind]
        frame_next = gaze_data_updated.iloc[first_frame_ind + 1]
        n_missing_frames = int(frame_next.video_frame - frame_prev.video_frame)
        gap_inds_updated.append(first_frame_ind)
        gap_lens.append(n_missing_frames)

        # Calculate frame time information
        time_diff = frame_next.time - frame_prev.time
        n_dts = round(time_diff / dt_median)

        # Calculate the number of dropped frames
        n_dropped_frames = n_dts - n_missing_frames

        # Insert rows into the DataFrame at the proper frame index
        new_df_rows = np.arange(frame_prev.video_frame, frame_next.video_frame, 1 / n_missing_frames)[1:-1]
        new_times = np.arange(frame_prev.time, frame_next.time - (n_dropped_frames * dt_median), dt_median)[1:-1]
        for i, (loc, t) in enumerate(zip(new_df_rows, new_times)):
            gaze_data_updated.loc[loc] = [t, frame_prev.video_frame + i + 1]
        gaze_data_updated = gaze_data_updated.sort_index().reset_index(drop=True)

    # Check after insertion that the number of frames equals the last frame index
    assert len(gaze_data_updated) == int(gaze_data.video_frame.iloc[-1] + 1)

    # Check the interpolation
    if plot_result:
        for ind1, ind2, length in zip(gap_inds_old, gap_inds_updated, gap_lens):
            # Before adding frames
            print(gaze_data_old.iloc[ind1 - 2:ind1 + 2])
            # After adding frames
            print(gaze_data_updated.iloc[ind2 - 2:ind2 + length + 2])
            print()

    return gaze_data_updated


def separate_diode_blocks(
        diode_df: pd.DataFrame,
        new_block_threshold: int,
        apriltag_threshold: int
) -> list[pd.DataFrame]:
    """Separates the diode data into experimental blocks.

    Since blocks tend to be divided by relatively high diode light values, these 'new block' threshold
    crossings can be used to roughly separate the experimental blocks. These blocks are then trimmed to
    begin at the onset of the first of five AprilTags, and end at the onset of the next 'new block' threshold
    crossing.

    This method relies on assuming the pattern of five AprilTags begin a block, and that there is a high
    diode value separating each block, since the exact number of trials in a block is variable. It also
    assumes that there is a high diode value before the first block, and following the last.

    Note: the blocks contain more data here than in their lengths

    Parameters
        diode_df (pd.DataFrame): raw light diode data
        new_block_threshold (int): light diode threshold for signalling a new block
        apriltag_threshold (int): light diode threshold for signalling the presence of an AprilTag

    Returns
        block_list (list[pd.DataFrame]): light diode data separated into individual blocks
    """

    # Find light diode values that indicate a new block begins
    light_values = diode_df.light_value.to_numpy('int', copy=True)
    new_block_crossings = np.where(np.diff(light_values > new_block_threshold))[0] + 1

    # # TODO: need some kind of check in case the light diode starts with a high value
    # # I think this works
    # if light_values[0] > self.diode_threshold_new_block:
    #     new_block_crossings = np.insert(new_block_crossings, 0, 0)
    # TODO: check if light diode value always goes high at the end of a session (required for step below)
    cross_down_inds = new_block_crossings[1::2]
    block_end_inds = new_block_crossings[2::2]

    # Find light diode indices where new blocks begin (i.e. onset of first AprilTag)
    apriltag_event_crossings = np.where(np.diff(light_values > apriltag_threshold))[0] + 1
    # The line below is a little confusing, but it basically finds the first threshold crossing after the
    #  downward crossing of a high value threshold. This should be the onset of the first AprilTag in a block.
    first_apriltag_inds_event_crossings = np.searchsorted(apriltag_event_crossings, cross_down_inds) + 1
    first_apriltag_inds_time = apriltag_event_crossings[first_apriltag_inds_event_crossings]

    # Separate data into blocks
    block_list = []
    for i, ind1 in enumerate(first_apriltag_inds_time):
        if i < len(first_apriltag_inds_time) - 1:
            ind2 = first_apriltag_inds_time[i + 1]
            block = diode_df.iloc[ind1:ind2, :]
        else:
            block = diode_df.iloc[ind1:, :]
        block_time = block.time.to_numpy('float', copy=True)
        block_time -= block_time[0]
        block.loc[:, 'time'] = block_time
        block.reset_index(drop=True, inplace=True)
        block_list.append(block.iloc[:, :block_end_inds[i]])

    return block_list


def get_event_times(block_list: list, event_threshold: int) -> tuple[list[np.ndarray], list[float]]:
    """Extracts the event onset times.

    These times are used to check the synchronization between the video and diode data.

    Parameters
        block_list (list[pd.DataFrame]): light diode data, separated into individual blocks
        event_threshold (int): light diode threshold for signalling event onsets

    Returns
        event_onsets (list[np.ndarray]): event onset times for each block
        block_durations (list[float]): duration of each block
    """

    event_times = []
    block_durations = []
    for i, block in enumerate(block_list):
        # Zero the block time
        block_time = block.time.to_numpy('float', copy=True)

        # Get event onset indices
        light_values = block.light_value.to_numpy('int', copy=True)
        all_event_inds = np.where(np.diff(light_values > event_threshold))[0] + 1

        # Get event onset times
        last_ind = -2 if i < (len(block_list) - 1) else -1
        event_onset_inds = all_event_inds[10:last_ind:2]
        event_times.append(block_time[event_onset_inds])
        block_durations.append(block_time[all_event_inds[last_ind]])

    return event_times, block_durations
