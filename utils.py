import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_EDGES_APRILTAG_SET = 10
N_APRILTAGS = 5
MIN_TRIALS = 40


def get_fourcc(cap: cv2.VideoCapture) -> str:
    """Return the 4-letter string of the codec the video uses.

    Parameters
        cap (cv2.VideoCapture): the OpenCV video capture object

    Returns
        (str): the fourcc codec of the mp4 video
    """
    fourcc_codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    return fourcc_codec.to_bytes(4, byteorder=sys.byteorder).decode()


def get_top_left_coords(corners: list[list]):
    """Finds the top left corner coordinates in a list of rectagle vertices (x, y).

    The origin is the top left corner of the video frame, so this function uses that to determine
    the top left corner of the rectangle. Whichever corner is closest to the origin, is the top left.

    Parameters
        corners (list[list]): list of the rectangle's vertices

    Returns
        top_left_ind (int): index of the top left corner coordinates
    """
    min_dists = [distance_2d(x, y) for x, y in corners]
    return corners[np.argmin(min_dists)]


def distance_2d(x1: float, y1: float, x2: float = 0, y2: float = 0):
    """Calculates the distance between any two points in 2D space.

    Parameters
        x1 (float): x-coordinate of the first point
        y1 (float): y-coordinate of the first point
        x2 (float): x-coordinate of the second point
        y2 (float): y-coordinate of the second point

    Returns
        dist (float): distance between the points
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def get_files_containing(start_dir: str, string_match: str, string_exclude: str = "XXXXXXXXXX"):
    """Returns the files containing a particular string and their relative paths from the project root.

    File paths are relative from a given starting directory.

    References:
    - https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    - https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list

    Parameters
        start_dir (str): directory at which to start the walk
        string_match (str): only get files that contain this string
        string_exclude (str): only get files that do not contain this string

    Returns
        paths (list[str]): sorted list of file directory paths
        files (list[str]): sorted list of the file names
    """
    files = []
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(start_dir):
        for file in filenames:
            if string_match in file and string_exclude not in file:
                files.append(file)
                paths.append(dirpath)
    paths_sorted = [path for _, path in sorted(zip(files, paths))]
    files.sort()
    return paths_sorted, files


def load_diode_data(participant_id: str, session_id: str) -> pd.DataFrame:
    """Loads the preprocessed light diode sensor data as a dataframe.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        diode_df (pd.DataFrame): processed diode data
    """
    diode_path = f"data/pipeline_data/{participant_id}/{session_id}/{participant_id:02}_{session_id}_diode_sensor.csv"
    return pd.read_csv(diode_path)


def load_video_time(participant_id: str, session_id: str) -> np.ndarray:
    """Loads the preprocessed light diode sensor data as a dataframe.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        diode_df (pd.DataFrame): processed diode data
    """
    time_path = f"data/pipeline_data/{participant_id}/{session_id}/{participant_id:02}_{session_id}_video_time.csv"
    return pd.read_csv(time_path).to_numpy('float')


def get_block_data(
        diode_df: pd.DataFrame,
        diode_threshold: int,
        separator_threshold: int | None,
        n_blocks: int = 10,
        show_plots: bool = False,
) -> tuple[list[pd.DataFrame], list[int], list[np.ndarray]]:
    """Separates the diode data into experimental blocks.

    Since blocks tend to be divided by relatively high diode light values, these "separator" threshold
    crossings can be used to determine the end of an experimental block. The separator values may be
    greater than the event and AprilTag values, or approximately the same. Each of these cases must be
    handled differently.

    This function assumes a set of five AprilTags begin each block. Since the exact number of trials in
    a block is variable, only a minimum number of trials is required for a block to be considered valid.

    There are various special cases which must be handled.

    Cases concerning the diode light values:
    1. Separator values exist, and are much greater than the AprilTag and event onset values
        -> separator_threshold >> diode_threshold
    2. Separator values exist but are approximately the same as AprilTag and event onset values
        -> separator_threshold == diode_threshold
    3. Separator values do not exist, or are very low
        -> separator_threshold == None

    Cases concerning the progression of blocks with a session
    1. An AprilTag set is followed by invalid block
         -> Identified by checking the number of trials (valid blocks contain >= 40 trials)
    2. One (or more) blocks has a different diode threshold than the rest (increasing, decreasing or outlier)
         -> Thresholds must be chosen individually for each experiment

    Outline of function calculations:
     - Split the diode data into blocks using the AprilTag sets
     - Calculate the event onset indices based on the cases described above
     - Truncate the blocks to remove excess data (to help speed up video processing later)
     - Determine which blocks are valid (>=40 trials)
     - Check that the number of valid blocks found is the number expected

    Parameters
        diode_df (pd.DataFrame): raw light diode data
        diode_threshold (int): light diode threshold for all diode events
        separator_threshold (int): light diode threshold for signalling the end of a block
        n_blocks (int): number of valid blocks in the diode data (10 unless otherwise specified)
        show_plots (bool): if True, show plots to visually check the process

    Returns
        valid_blocks (list[pd.DataFrame]): light diode data separated into "valid" blocks (>=40 trials)
        valid_block_inds (list[int]): indices of the AprilTag sets that correspond to valid blocks
        event_onset_times (list[np.ndarray]): event onset times for each block
    """

    # Identify AprilTag sets
    first_apriltag_inds = get_apriltag_sets(diode_df, diode_threshold)

    # Separate data into valid blocks
    all_blocks = separate_blocks(diode_df, first_apriltag_inds)

    valid_blocks = []
    valid_block_inds = []
    event_onset_times = []
    for i, block in enumerate(all_blocks):
        # Calculate a list of all threshold crossings
        block_time = block.time.to_numpy('float', copy=True)
        block_light_values = block.light_value.to_numpy('int', copy=True)
        block_crossings = np.where(np.diff(block_light_values > diode_threshold))[0]

        # Get the event onset times, calculated depending on the threshold scenario
        if separator_threshold is None:
            last_event_ind = None
        elif separator_threshold > diode_threshold:
            last_event_ind = -2
        elif separator_threshold == diode_threshold:
            event_crossing_times = block_time[block_crossings[N_EDGES_APRILTAG_SET:]]
            if event_crossing_times.size < MIN_TRIALS:
                continue
            event_durations = np.diff(event_crossing_times)[::2]
            avg_event_duration = np.mean(event_durations)
            last_event_ind = np.where(event_durations > (3 * avg_event_duration))[0][0] * 2 + N_EDGES_APRILTAG_SET
        else:
            raise ValueError("The separator_threshold provided is invalid.")
        if separator_threshold is not None and len(valid_blocks) == (n_blocks - 1):
            last_event_ind = -1
        event_onset_inds = block_crossings[N_EDGES_APRILTAG_SET:last_event_ind:2]
        event_onset_times.append(block_time[event_onset_inds])

        # If there are not enough trials in the block, it is invalid
        if len(event_onset_inds) < MIN_TRIALS:
            continue

        # Trim the block to remove excess data, then store valid blocks
        if last_event_ind is not None:
            # TODO: Check whether separator diode values occur in all blocks when they occur at all
            block_end_ind = block_crossings[last_event_ind]
        else:
            avg_onset_diff = np.mean(np.diff(event_onset_times[-1]))
            block_end_time = event_onset_times[-1][-1] + avg_onset_diff
            if block_end_time > block_time[-1]:
                block_end_ind = None
            else:
                block_end_ind = np.where(block_time > block_end_time)[0][0]
        if block_end_ind is not None:  # TODO: check to ensure this handles all cases
            valid_blocks.append(block.iloc[:block_end_ind, :])
        else:
            valid_blocks.append(block)
        valid_block_inds.append(i)

        # As a visual check
        if show_plots:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(block_time, block_light_values)
            ax.vlines(block_time[block_crossings], 0, diode_threshold, colors='r', linewidths=3)
            ax.vlines(event_onset_times[-1], 0, diode_threshold // 2, colors='g', linewidths=3)
            ax.plot(valid_blocks[-1].time, valid_blocks[-1].light_value)
            plt.show()
            plt.close()

    assert len(valid_blocks) == n_blocks

    return valid_blocks, valid_block_inds, event_onset_times


def get_apriltag_sets(diode_df: pd.DataFrame, diode_threshold: int) -> list[int]:
    """Returns the index of the first AprilTag in each set.

    Each index corresponds to a time in the diode light sensor data.

    Parameters
        diode_df (pd.DataFrame): raw light diode data
        diode_threshold (int): light diode threshold for all diode events
        separator_threshold (int): light diode threshold for signalling the end of a block
        n_blocks_missing (int): number of blocks missing or unusable from data

    Returns
        first_apriltag_inds (list[int]): indices marking the start of each AprilTag set
    """

    light_values = diode_df.light_value.to_numpy('int', copy=True)
    diode_time = diode_df.time.to_numpy('float', copy=True)
    all_crossings = np.where(np.diff(light_values > diode_threshold))[0]
    first_apriltag_inds = []
    skip_tags = 0
    for i, ind in enumerate(all_crossings[:-N_EDGES_APRILTAG_SET]):
        # Once a full AprilTag set is identified, skip over it
        if skip_tags > 0:
            skip_tags -= 1
            continue
        ind_set = all_crossings[i:i + N_EDGES_APRILTAG_SET]
        time_set = diode_time[ind_set]
        # Identify AprilTag sets by the time between the first and last edge (~9 seconds)
        if 9.0 < (time_set[-1] - time_set[0]) < 9.2:
            n_tags = 0
            # Each AprilTag should be visible for ~1 second
            for t1, t2 in zip(time_set[::2], time_set[1::2]):
                if 0.9 < (t2 - t1) < 1.10:
                    n_tags += 1
            # There should be exactly 5 AprilTags in a set
            if n_tags == N_APRILTAGS:
                skip_tags = N_EDGES_APRILTAG_SET - 1
                first_apriltag_inds.append(ind)

    return first_apriltag_inds


def separate_blocks(diode_df: pd.DataFrame, apriltag_inds: list[int]) -> list[pd.DataFrame]:
    """Returns a list of the diode data separated by AprilTags.

    Parameters
        diode_df (pd.DataFrame): raw light diode data
        apriltag_inds (list[int]): indices marking the start of each AprilTag set

    Returns
        all_blocks (list[pd.DataFrame]): all diode data blocks, separated by AprilTag sets
    """

    all_blocks = []
    for i, ind1 in enumerate(apriltag_inds):
        if i < len(apriltag_inds) - 1:
            ind2 = apriltag_inds[i + 1]
            block = diode_df.iloc[ind1:ind2, :]
        else:
            block = diode_df.iloc[ind1:, :]
        block.time -= block.time.iloc[0]
        block.reset_index(drop=True, inplace=True)
        all_blocks.append(block)

    return all_blocks
