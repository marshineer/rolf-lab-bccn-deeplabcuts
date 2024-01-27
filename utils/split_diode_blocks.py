import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

N_EDGES_APRILTAG_SET = 10
N_APRILTAGS = 5
MIN_TRIALS = 40
MIN_TRIAL_SEPARATION = 1.5


def get_block_data(
        diode_df: pd.DataFrame,
        diode_threshold: int,
        separator_threshold: int | None,
        n_blocks: int,
        skip_blocks: list[int],
        extra_apriltag_blocks: list[int],
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
        skip_blocks (list[int]): indices of blocks that are invalid (manually check diode data to determine this)
        extra_apriltag_blocks (list[int]): indices of blocks where AprilTags appears more than 5 times in a set
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
        last_event_ind = get_last_event_ind(
            block_crossings,
            block_time,
            diode_threshold,
            separator_threshold,
            len(valid_blocks),
            n_blocks,
        )
        event_onset_inds = block_crossings[N_EDGES_APRILTAG_SET:last_event_ind:2]
        if i in extra_apriltag_blocks:
            event_onset_inds = event_onset_inds[1:]

        # Remove extra event times
        event_times_temp = block_time[event_onset_inds]
        event_diffs = np.diff(event_times_temp)
        too_close_onsets = np.argwhere(event_diffs < MIN_TRIAL_SEPARATION)
        if too_close_onsets.size > 0:
            event_times_temp = np.delete(event_times_temp, too_close_onsets + 1)

        # If there are not enough trials in the block, it is invalid
        if len(event_times_temp) < MIN_TRIALS:
            continue
        event_onset_times.append(event_times_temp)

        # Trim the block to remove excess data, then store valid blocks
        if last_event_ind is not None:
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
            fig, ax = plt.subplots(1, 1, figsize=(16, 5))
            ax.plot(block_time, block_light_values)
            ax.vlines(block_time[block_crossings], 0, diode_threshold, colors='r', linewidths=3)
            ax.vlines(event_onset_times[-1], 0, diode_threshold // 2, colors='g', linewidths=3)
            ax.plot(valid_blocks[-1].time, valid_blocks[-1].light_value)
            plt.show()
            plt.close()

    # Modify flagged block data
    for i in skip_blocks[::-1]:
        del valid_blocks[i]
        del valid_block_inds[i]
        del event_onset_times[i]

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
        if 8.9 < time_set[-1] - time_set[0] < 9.3:
            n_tags = 0
            # Each AprilTag should be visible for ~1 second
            for t1, t2 in zip(time_set[::2], time_set[1::2]):
                if 0.9 < t2 - t1 < 1.10:
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


def get_last_event_ind(
        block_crossings: np.ndarray,
        block_time: np.ndarray,
        diode_threshold: int,
        separator_threshold: int | None,
        n_valid_blocks: int,
        n_total_blocks: int,
) -> int:
    if separator_threshold is None:
        last_event_ind = None
    elif separator_threshold > diode_threshold:
        last_event_ind = -2
    elif separator_threshold == diode_threshold:
        event_crossing_times = block_time[block_crossings[N_EDGES_APRILTAG_SET:]]
        if event_crossing_times.size // 2 < MIN_TRIALS:
            return 0
        event_durations = np.diff(event_crossing_times)[::2]
        avg_event_duration = np.mean(event_durations)
        long_durations = np.where(event_durations > (3 * avg_event_duration))[0]
        if long_durations.size > 0:
            return long_durations[0] * 2 + N_EDGES_APRILTAG_SET
        else:
            return -2
    else:
        raise ValueError("The separator_threshold provided is invalid.")
    if separator_threshold is not None and n_valid_blocks == (n_total_blocks - 1):
        last_event_ind = -1

    return last_event_ind
