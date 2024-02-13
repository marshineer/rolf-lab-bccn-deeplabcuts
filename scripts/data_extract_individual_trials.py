import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(".."))
from data_calculate_hand_speeds import TransformedHandData, load_transformed_hand
from config.config_dataclasses import PostprocessingConfig
from utils.pipeline import SessionData, load_session_data
from utils.data_loading import get_files_containing, load_postprocessing_config
from utils.split_diode_blocks import MIN_TRIALS, MIN_TRIAL_SEPARATION


MAX_ZERO_DT = 6
MAX_SPEED = 500


# TODO:
#  - Add overwrite parameter in argparse
#  - Add docstring to module
#  - Add plotting function
#     -> Histogram of differences between diode onsets and each jatos onset type (flash on, flash off, change)
@dataclass
class TrialData:
    participant_id: str
    session_id: str
    block_id: int
    start_time: float
    end_time: float
    time_vec: np.ndarray
    hand_positions: dict[int, np.ndarray]
    hand_speeds: dict[int, np.ndarray]
    change_time: float
    tap_times_on: np.ndarray
    tap_times_off: np.ndarray
    tap_positions: np.ndarray
    dot_positions: np.ndarray
    flash_change: bool
    shift_change: bool
    trial_usable: bool = True


def get_session_trials(
        participant_id: str,
        session_id: str,
        jatos_fpath: str,
        plot_onsets: bool = False
) -> None:
    """Extracts the individual trial data for a particular session.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]
        jatos_fpath (str): filepath to the jatos JSON data
        plot_onsets (bool): if True, plot a visual comparison of the jatos and diode change onset times
    """

    # Define the file name
    filename = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_trial_data.pkl"

    # Load the post-processing config
    postprocess_config = load_postprocessing_config(f"../data/pipeline_data/{participant_id}/{session_id}", False)

    # Get the event onset and time data for the session
    session_data = load_session_data(participant_id, session_id)

    # Load the hand tracking data
    hand_data: TransformedHandData = load_transformed_hand(participant_id, session_id)

    # Separate the jatos data into blocks and trials
    jatos_blocks, jatos_apriltag_times = parse_jatos_json(jatos_fpath)

    # Remove unused data from jatos events
    filter_unparsed_blocks(jatos_blocks, postprocess_config)
    filter_unparsed_blocks(jatos_apriltag_times, postprocess_config)

    # Check that the jatos and diode data have the same number of blocks
    assert len(jatos_blocks) == len(session_data.event_onsets_blocks)

    # Remove invalid trials
    jatos_blocks = filter_unsuccessful_trials(jatos_blocks)

    # Extract the jatos data for each trial in the session
    session_trials = []
    for n_block, jatos_block in enumerate(jatos_blocks):
        # Skip any blocks with errors identified during video processing
        if n_block in postprocess_config.skip_blocks:
            continue
        print(f"Extracting trials for {session_data.participant_id}-{session_data.session_id}, Block {n_block}")

        # Synchronize jatos and video/diode trial times
        jatos_t0 = sync_trial_times(
                        jatos_apriltag_times[n_block],
                        session_data.apriltag123_visible[n_block],
                        session_data.block_times[n_block]
                    )

        block_trials = extract_block_trials(jatos_block, session_data, hand_data, jatos_t0, n_block, plot_onsets)
        session_trials.append(block_trials)

    # Save the transformed hand data
    with open(filename, "wb") as f:
        pickle.dump(session_trials, f)


def parse_jatos_json(jatos_path: str) -> tuple[list[list[dict[str, Any]]], list[np.ndarray]]:
    """Parses the session event data from a jatos JSON file.

    Parameters
        jatos_path (str): full file path to the jatos JSON file for a particular session

    Returns
        jatos_blocks (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
        jatos_apriltag_edge_times (list[np.ndarray]): AprilTag leading and trailing edge times for each block
    """

    jatos_blocks = []
    jatos_apriltag_edge_times = []
    with open(jatos_path) as f:
        for line in f:
            if len(json.loads(line)) > 80:
                json_data = json.loads(line)

                # Extract the AprilTag edges and convert from ms -> s
                jatos_apriltags = json_data[2]["change_timestamps"]
                jatos_apriltag_edge_times.append(np.stack(jatos_apriltags) / 1000)

                # Extract the trial data
                jatos_trials = json_data[4:]
                del_inds = [idx for idx, d in enumerate(jatos_trials) if len(d) not in [13, 39]]
                for i in del_inds[::-1]:
                    del jatos_trials[i]
                jatos_trial_data = jatos_trials[::2]

                # Remove unsuccessful trials
                valid_trial_data = [trial for trial in jatos_trial_data if trial["success"]]
                assert len(valid_trial_data) == 40

                jatos_blocks.append(valid_trial_data)

    # Confirm that there are 10 blocks of data for the session
    assert len(jatos_blocks) == 10

    return jatos_blocks, jatos_apriltag_edge_times


def filter_unparsed_blocks(
        jatos_data: list[Any],
        postprocess_config: PostprocessingConfig,
) -> None:
    """Removes blocks from jatos JSON data, which could not be parsed or had parsing errors.

    Parsing errors are identified during preprocessing of the diode and pipeline processing of videos.

    Parameters
        jatos_data (list[Any]): parsed jatos data, separated into session blocks and trials
        postprocess_config (PostprocessingConfig): dataclass containing post-processing parameter values
    """

    postprocess_config.missing_blocks.sort()
    postprocess_config.missing_blocks.reverse()
    for i in postprocess_config.missing_blocks:
        del jatos_data[i]


def filter_unsuccessful_trials(jatos_data: list[list[dict[str, Any]]]):
    """Removes all trials that are not tagged as successful in the jatos JSON.

    Parameters
        jatos_data (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials

    Returns
        valid_trial_data (list[list[dict[str, Any]]]): jatos data containing only the 40 valid trials
    """

    valid_trial_data = [[trial for trial in block if trial["success"]] for block in jatos_data]
    for i, block_trials in enumerate(valid_trial_data):
        # Check that there are the correct number of trials in the data
        assert len(block_trials) == MIN_TRIALS

    return valid_trial_data


def sync_trial_times(
        jatos_apriltag_times: np.ndarray,
        video_apriltag_times: np.ndarray,
        block_time: np.ndarray
) -> float:
    """Find the offset required to synchronize the jatos trial times with the diode data.

    The jatos and diode data both contain leading and trailing edge times for the AprilTags that begin
    a new block. Usually the jatos data contains all edges except the first leading edge (either 9 or
    11, depending on the number of times the AprilTags flash on the screen). In this case, the trailing
    edge of the first AprilTag in the set is used to determine the offset. In the other case, the jatos
    data contains 10 AprilTag edges. This occurs only when the AprilTags flash 6 times. In this case,
    The last trailing edge of the AprilTag set is used to determine the offset.

    Parameters
        jatos_apriltag_times (np.ndarray): AprilTag edge times parsed from the jatos JSON data
        video_apriltag_times (np.ndarray): AprilTag edge times detected during video processing
        block_time (np.ndarray): time vector for the trial block

    Returns
        jatos_t0 (float): offset required to sync diode and jatos trial data for the block
    """

    if jatos_apriltag_times.size % 2 == 1:
        # Use last trailing edge of video tags (works when there are an even number of times in jatos data)
        video_tag_edge_last = block_time[np.argwhere(video_apriltag_times == 1).squeeze()[-1]]
        jatos_t0 = jatos_apriltag_times[-1] - video_tag_edge_last
    else:
        # Use first trailing edge of video tags (only works when there are 9 or 11 times in the jatos data)
        video_tag_edge1 = block_time[np.argwhere(video_apriltag_times < 1)[0][0] - 1]
        jatos_t0 = jatos_apriltag_times[0] - video_tag_edge1

    return jatos_t0


def extract_block_trials(
        jatos_block: list[dict[str, Any]],
        session_data: SessionData,
        hand_data: TransformedHandData,
        jatos_t0: float,
        n_block: int,
        plot_onsets: bool,
) -> list[TrialData]:
    """Extracts the data for individual trials from the jatos JSON data.

    The data extracted by this function will be used in later analyses.

    Parameters
        jatos_data (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
        block_dts (list[float]): offsets required to sync diode and jatos data for each block
        session_data (SessionData): class containing the pipeline session data
        hand_data (TransformedHandData): class containing the transformed hand data
        n_block (int): index of the block of trials currently being extracted
        plot_onsets (bool): if True, plot a visual comparison of the jatos and diode event onset times

    Returns
        block_trials (list[TrialData]): individual trail data for the block
    """

    # Remove trials that do not exist in both the jatos and diode data
    change_times, jatos_block = match_jatos_diode_trials(jatos_block, session_data, n_block, jatos_t0, plot_onsets)

    block_trials = []
    for n_trial, trial_dict in enumerate(jatos_block):
        # Populate the TrialData class instance
        trial_data = populate_trial_dataclass(
                            trial_dict,
                            session_data,
                            hand_data,
                            change_times[n_trial],
                            jatos_t0,
                            n_block
                        )

        block_trials.append(trial_data)

    return block_trials


def match_jatos_diode_trials(
        jatos_block: list[dict[str, Any]],
        session_data: SessionData,
        n_block: int,
        jatos_t0: float,
        plot_onsets: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Filters both the diode and jatos trials such that only trials common to both remain.

    There are more diode onset events than jatos events, because the jatos events have already been
    filtered to exclude unsuccessful trials. Therefore, we must find the diode trials that correspond
    to the valid jatos trials. Additionally, there are some jatos trials that were missed during the
    parsing of diode change onsets. Therefore, these are removed from the jatos data.

    Parameters
        jatos_block (list[dict[str, Any]]): parsed jatos data containing all trials for the given block
        session_data (SessionData): class containing the pipeline session data
        n_block (int): index of the block of trials currently being processed
        jatos_t0 (float): offset required to sync diode and jatos trial data for the block
        plot_onsets (bool): if True, plot a visual comparison of the jatos and diode event onset times

    Returns
        change_times (np.ndarray): change onset times for trials that exist in both the diode and jatos data
        jatos_block (list[dict[str, Any]]): jatos block containing trials that exist in both the diode and jatos data
    """

    # Extract the change onsets parsed from the diode data
    video_change_times = session_data.event_onsets_blocks[n_block].copy()

    # Calculate the change onset times from the jatos data
    jatos_start_times = np.stack([(trial["startTime"] / 1000) - jatos_t0 for trial in jatos_block])
    jatos_change_dt = np.stack([trial["change_onset"] / 1000 for trial in jatos_block])
    jatos_change_times = jatos_change_dt + jatos_start_times
    assert jatos_change_times.size == 40

    # Find trials common to both jatos and diode data
    matching_jatos_inds, matching_diode_inds = get_matching_trial_inds(jatos_change_times, video_change_times)
    change_times = video_change_times[matching_diode_inds]
    jatos_block = [jatos_block[index] for index in matching_jatos_inds]

    if plot_onsets:
        plot_change_onset_comparison(
            jatos_change_times,
            video_change_times,
            matching_jatos_inds,
            matching_diode_inds,
            session_data.participant_id,
            session_data.session_id,
            n_block,
        )

    return change_times, jatos_block


def get_matching_trial_inds(
        jatos_change_times: np.ndarray,
        diode_change_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Finds the indices of the diode and jatos trials that are common to both sets of data.

    Parameters
        jatos_change_times (np.ndarray): all (valid) successful jatos change onset times
        diode_change_times (np.ndarray): all diode change onset times (successful and unsuccessful)

    Returns
        matching_jatos_inds (list[int]): diode trial indices that match the valid (successful) jatos trials
        matching_diode_inds (list[int]): jatos (successful) trial indices that match the parsed diode trials
    """

    matching_diode_inds = []
    remove_jatos_inds = []
    i_diode = 0
    for i_jatos, t in enumerate(jatos_change_times):
        max_depth = diode_change_times.size - i_diode - 1
        i_diode_new = match_trials(t, i_diode, diode_change_times, max_depth)
        if i_diode_new is None:
            remove_jatos_inds.append(i_jatos)
        else:
            matching_diode_inds.append(i_diode_new)
            i_diode = i_diode_new + 1

    matching_jatos_inds = np.delete(np.arange(jatos_change_times.size), remove_jatos_inds)

    return matching_jatos_inds, np.stack(matching_diode_inds)


def match_trials(
        jatos_change_time: float,
        diode_change_ind: int,
        diode_change_times: np.ndarray,
        max_depth: int,
        n_missed: int = 0,
) -> int | None:
    """Recursive function that finds the next diode onset index which matches a successful jatos trial.

    The jatos and diode trials are considered matching if the difference between onset times is less
    than a minimum time between trials. In reality, the difference should be much less than this. If
    no matching trial is found, it means the jatos trial is missing from the diode data (undetected
    during parsing, or no flash occurred to indicate a change onset).

    Parameters
        jatos_change_time (float): onset time of the valid jatos trial
        diode_change_ind (int): index of the diode trial currently being checked
        diode_change_times (np.ndarray): all diode onset times (successful and unsuccessful)
        max_depth (int): maximum number of skipped (unsuccessful) diode trials to search
        n_missed (int): current number of skipped (unsuccessful) diode trials

    Returns
        (int): index of the diode trial that matches the next valid jatos trial
    """

    if abs(jatos_change_time - diode_change_times[diode_change_ind]) < MIN_TRIAL_SEPARATION:
        return diode_change_ind
    elif n_missed >= max_depth:
        print("Jatos trial missing in diode data.")
        return None
    else:
        return match_trials(jatos_change_time, diode_change_ind + 1, diode_change_times, max_depth, n_missed + 1)


def plot_change_onset_comparison(
        jatos_onsets: np.ndarray,
        video_onsets: np.ndarray,
        matching_jatos_inds: np.ndarray,
        matching_diode_inds: np.ndarray,
        participant_id: str,
        session_id: str,
        n_block: int,
) -> None:
    """Plot a comparison of the jatos and diode event onset times, to ensure they are properly synchronized.

    Parameters
        jatos_onsets (np.ndarray): all jatos change onset times (successful trials)
        video_onsets (np.ndarray): all parsed diode change onset times (successful and unsuccessful trials)
        matching_jatos_inds (np.ndarray): jatos trial indices that exist in the parsed diode data
        matching_diode_inds (np.ndarray): diode trial indices that match the successful jatos trials
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]
        n_block (int): index of the block of trials currently being processed
    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.vlines(jatos_onsets, 3, 4, "r", label="Jatos (Successful)")
    ax.vlines(video_onsets, 2, 3, "b", label="Diode (All)")
    ax.vlines(video_onsets[matching_diode_inds], 1, 2, "r", linestyles="--", label="Diode (Successful)")
    ax.vlines(jatos_onsets[matching_jatos_inds], 0, 1, "b", linestyles="--", label="In Diode and Jatos")
    ax.set_xlabel("Block Time [seconds]", fontsize=24)
    ax.set_title(f"Comparison of Change Onset Times ({participant_id}-{session_id}, Block {n_block})", fontsize=20)
    ax.legend(loc=0)
    plt.show()
    plt.close()


def populate_trial_dataclass(
        trial_dict: dict[str, Any],
        session_data: SessionData,
        hand_data: TransformedHandData,
        change_time: float,
        jatos_t0: float,
        n_block: int,
) -> TrialData:
    """Populates an instance of the TrialData dataclass.

    Extracts the data from the jatos JSON that will be used in later analyses. The dot positions are given
    relative to the center of the screen, whereas the positions where the participant touches (taps) the
    screen are given relative to the corner of the screen. Therefore, the tap positions must be offset to
    be in the same frame of reference as the dot positions.

    The tap times are offset by t0, which is a factor required to synchronize the diode/video data with the
    data from the jatos JSON. This synchronization can be checked visually by setting show_plot to True in
    get_trial_data().

    Parameters
        trial_dict (dict[str, Any]): jatos JSON trial data
        session_data (SessionData): class containing the pipeline session data
        hand_data (TransformedHandData): class containing the transformed hand data
        change_time (float): block time at which the change onset occurs
        jatos_t0 (float): offset required to sync diode and jatos trial data for the block
        n_block (int): index of the block of trials currently being extracted

    Returns
        trial_data (TrialData): class containing the individual trail data
    """

    # Calculate trial start and end times
    t_start, t_end = get_trial_boundary_times(trial_dict, jatos_t0)

    # Get the trial indices for the interpolated time (kinematic data)
    trial_inds = get_trial_inds(hand_data.time_interpolated[n_block], t_start, t_end)

    # Extract the hand kinematic data for the trial
    hand_pos_trial, hand_speed_trial = get_hand_kinematics(hand_data, n_block, trial_inds)

    # Calculate the tap times
    tap_times_on, tap_times_off = get_tap_times(trial_dict, jatos_t0)

    # Calculate tap positions
    tap_positions = get_tap_positions(trial_dict)

    # Calculate the true dot positions
    dot_positions = get_dot_positions(trial_dict, tap_times_on, change_time)

    # Populate the trial dataclass
    trial_data = TrialData(
        participant_id=session_data.participant_id,
        session_id=session_data.session_id,
        block_id=n_block,
        start_time=t_start,
        end_time=t_end,
        time_vec=hand_data.time_interpolated[n_block][trial_inds],
        hand_positions=hand_pos_trial,
        hand_speeds=hand_speed_trial,
        change_time=change_time,
        tap_times_on=tap_times_on,
        tap_times_off=tap_times_off,
        tap_positions=tap_positions,
        dot_positions=dot_positions,
        flash_change=trial_dict["flashShown"],
        shift_change=trial_dict["stimJumped"],
    )

    # Determine whether the trial is unusable
    block_time = session_data.block_times[n_block]
    trial_inds_video = get_trial_inds(block_time, t_start, t_end)
    is_trial_usable(trial_data, trial_inds_video, block_time, session_data.reference_pos_abs[n_block])

    # # Plot the trial
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    # aligned_time = trial_data.time_vec - trial_data.change_time
    # plot_speed = trial_data.hand_speeds[8][-1, :]
    # ax.plot(aligned_time, plot_speed)
    # ax.vlines(0, np.min(plot_speed), np.max(plot_speed), 'r')
    # ax.set_title(f"{trial_data.participant_id}-{trial_data.session_id}, Block {trial_data.block_id}")
    # plt.show()
    # plt.close()

    return trial_data


def get_trial_boundary_times(trial_dict: dict[str, Any], jatos_t0: float) -> tuple[float, float]:
    """Gets the start and end times of the trial.

    The start and end times are determined by the times of the first and last screen tap.

    Parameters
        trial_dict (dict[str, Any]): jatos JSON trial data
        jatos_t0 (float): offset required to sync diode and jatos trial data for the block

    Returns
        t_start (float): time of the first screen tap
        t_end (float): time of the last (6th) screen tap
    """

    # Calculate the start and end times
    t_start = trial_dict["startTime"] / 1000 - jatos_t0
    t_end = trial_dict["endTime"] / 1000 - jatos_t0

    # Check that the first touch and recorded trial start time are the same
    assert np.round(t_start, 3) == np.round(trial_dict["touchOn"][0] / 1000 - jatos_t0, 3)

    return t_start, t_end


def get_trial_inds(block_time: np.ndarray, trial_start_time: float, trial_end_time: float) -> np.ndarray:
    """Returns the indices of the block time vector, which correspond to a particular trial.

    Parameters
        block_time (np.ndarray): time vector for the entire block
        trial_start_time (float): block time at which the trial begins
        trial_end_time (float): block time at which the trial begins

    Returns
        (np.ndarray): indices in the block time vector associated with the trial
    """

    dt = np.median(np.diff(block_time))
    return np.arange(block_time.size)[(trial_start_time - dt <= block_time) & (block_time < trial_end_time + dt)]


def get_hand_kinematics(
        hand_data: TransformedHandData,
        n_block: int,
        trial_inds: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Extracts the kinematic hand data for a particular trial.

    Parameters
        hand_data (TransformedHandData): class containing the transformed hand data
        n_block (int): index of the block of trials currently being extracted
        trial_inds (np.ndarray): indices in the block time vector associated with the trial

    Returns
        hand_pos_trial (dict[int, np.ndarray]): transformed hand landmark position data for the trial
        hand_speed_trial (dict[int, np.ndarray]): transformed hand landmark speed data for the trial
    """

    hand_pos_trial = {}
    hand_speed_trial = {}
    for landmark_id in hand_data.hand_landmarks.values():
        hand_pos_trial[landmark_id] = hand_data.hand_pos_interpolated[n_block][landmark_id][:, trial_inds]
        try:
            hand_speed_trial[landmark_id] = hand_data.hand_speed[n_block][landmark_id][:, trial_inds]
        except IndexError:
            if trial_inds[-1] == hand_data.hand_speed[n_block][landmark_id].shape[1]:
                hand_speed_trial[landmark_id] = hand_data.hand_speed[n_block][landmark_id][:, trial_inds[:-1]]
            else:
                raise IndexError

    return hand_pos_trial, hand_speed_trial


def get_tap_times(trial_dict: dict[str, Any], jatos_t0: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns the tap on and off times for a particular trial.

    Tapping on occurs when the finger touches the screen, and off when it leaves the screen.

    Parameters
        trial_dict (dict[str, Any]): jatos JSON trial data
        jatos_t0 (float): offset required to sync diode and jatos trial data for the block

    Returns
        tap_times_on (np.ndarray): block times at which the 6 screen taps begin
        tap_times_off (np.ndarray): block times at which the 6 screen taps end
    """

    tap_times_on = (np.stack(trial_dict["touchOn"]) / 1000) - jatos_t0
    if len(trial_dict["touchOff"]) == 5:
        trial_dict["touchOff"].append(trial_dict["endTime"])
    tap_times_off = (np.stack(trial_dict["touchOff"]) / 1000) - jatos_t0

    return tap_times_on, tap_times_off


def get_dot_positions(trial_dict: dict[str, Any], tap_times: np.ndarray, change_time: float) -> np.ndarray:
    """Returns the true dot positions before and after the change onset.

    In the jatos data, there are two sets of dot positions (before and after change onset). However,
    there is only one set of true positions, since the dots that are tapped before the change onset
    disappear. This combines the two sets of positions into a single set of "true" positions.

    Parameters
        trial_dict (dict[str, Any]): jatos JSON trial data
        tap_times (np.ndarray): times at which the 6 screen taps occur
        change_time (float): time at which the change onset occurs

    Returns
        dot_positions (np.ndarray): true positions of the dots before and after the change onset
    """

    shift_mask = tap_times > change_time
    dot_positions = np.stack((trial_dict["position_x"], trial_dict["position_y"]))
    dot_pos_shifted = np.stack((trial_dict["shifted_position_x"], trial_dict["shifted_position_y"]))
    dot_positions[:, shift_mask] = dot_pos_shifted[:, shift_mask]

    return dot_positions


def get_tap_positions(trial_dict: dict[str, Any]) -> np.ndarray:
    """Returns the positions of the six screen taps.

    The tap coordinate frame of reference is in the corner of the phone screen, rather than the center
    like the other coordinates. Therefore, the coordinates must be shifted to account for this.

    Parameters
        trial_dict (dict[str, Any]): jatos JSON trial data

    Returns
        (np.ndarray): positions where the participant taps the screen
    """

    # Calculate the pixel offsets for the center of the screen
    x_center = trial_dict["windowWidth"] / 2
    y_center = trial_dict["windowHeight"] / 2

    return np.stack((np.array(trial_dict["touchX"]) - x_center, np.array(trial_dict["touchY"]) - y_center))


def is_trial_usable(
        trial_data: TrialData,
        trial_inds: np.ndarray,
        block_time: np.ndarray,
        apriltag_ref_pos: dict[int, np.ndarray],
) -> None:
    """Determines whether the trial is usable.

    There are three situations where the trial is unuasable:
      1. The event onset occurs after the last tap
      2. The video data ends before the last trial ends
      3. The hand speeds are beyond a reasonably physical limit
      4. There are an unacceptable number of missed detections of the reference AprilTags

    The event onset occurs after the last tap when the participat moves extremely quickly. This is
    an infrequent occurrence. On the other hand, missed AprilTag detections and probblems with hand
    tracking and speed calculations are much more common. Therefore, most unusable trials are due to
    these occurrences.

    Parameters
        trial_data (TrialData): class containing the individual trail data
        trial_inds (np.ndarray): indices in the block time vector associated with the trial
        block_time (np.ndarray): time vector for the entire block
        ref_pos_block (dict[int, np.ndarray]): coordinates of the reference AprilTags
    """

    # If the event onset occurs after the last tap, the trial cannot be used
    if trial_data.change_time > trial_data.end_time:
        trial_data.trial_usable = False
        return None

    # If the video data ends before the trial ends, the trial cannot be used
    if block_time[-1] < trial_data.end_time:
        trial_data.trial_usable = False
        return None

    # If the hand speed is unreasonably high, there was likely a problem during the tracking or spline fitting
    for speed_arr in trial_data.hand_speeds.values():
        if np.max(speed_arr) > MAX_SPEED:
            trial_data.trial_usable = False
            return None

    # If there are an unacceptable number of consecutive missed detections of the reference AprilTags
    for tag_id, reference_tag_pos in apriltag_ref_pos.items():
        tag_zero_runs = find_zero_runs(reference_tag_pos[0, trial_inds])
        if tag_zero_runs.size > 0:
            trial_data.trial_usable = False
            break


def find_zero_runs(data: np.ndarray) -> np.ndarray:
    """Finds the indices of consecutive zeros greater than a minimum length in a 1D array.

    Adapted from:
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array

    Each row in the output array represents a different run of zeros, greater than the minimum length.
    The two columns in the row are the first and last index of the run.

    Parameters
        data (np.ndarray): data in which to find consecutive zeros

    Returns
        zero_inds (np.ndarray): indices of the zero runs
    """

    # Create an array that is 1 where the data is 0, and pad each end with an extra 0
    iszero = np.concatenate(([0], np.equal(data, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    if ranges.size > 0:
        ranges = ranges[np.diff(ranges, axis=-1).squeeze() > MAX_ZERO_DT, :]

    return ranges


def load_session_trials(participant_id: str, session_id: str) -> list[list[TrialData]] | None:
    """Loads the jatos trial dataclasses for a particular session.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        (list[list[TrialData]]): individual trail data for each block and trial in the session
    """

    session_path = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_trial_data.pkl"
    if os.path.exists(session_path):
        with open(session_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def main(plot_onsets: bool) -> None:
    """Parses and extracts the required data from a jatos JSON file.

    Parameters
        plot_onsets (bool): if True, plot a visual comparison of the jatos and diode change onset times
    """

    jatos_paths, jatos_files = get_files_containing("../data/original_data", "jatos")
    for jatos_path, jatos_file in zip(jatos_paths, jatos_files):
        # Get the participant and session IDs
        participant_id, session_id = jatos_path.split("/")[-2:]

        # Define the file path to the jatos data
        get_session_trials(participant_id, session_id, os.path.join(jatos_path, jatos_file), plot_onsets)


if __name__ == "__main__":
    main(False)
