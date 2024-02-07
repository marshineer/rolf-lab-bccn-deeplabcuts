import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(".."))
from config.config_dataclasses import PostprocessingConfig
from utils.pipeline import SessionData, load_session_data
from utils.data_loading import get_files_containing, load_postprocessing_config


N_TRIALS = 40
MAX_ZERO_DT = 6


@dataclass
class TrialData:
    participant_id: str
    session_id: str
    trial_onset_dt: float
    trial_tap_times: np.ndarray
    trial_tap_pos: np.ndarray
    trial_dot_pos1: np.ndarray
    trial_dot_pos2: np.ndarray
    trial_flash: bool
    trial_shift: bool
    trial_usable: bool = True


def get_trial_data(
        participant_id: str,
        session_id: str,
        jatos_fpath: str,
        plot_onsets: bool = False
) -> None:
    # Define the file name
    filename = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_trial_data.pkl"

    # Load the post-processing config
    print(f"Parsing jatos file for session {participant_id}-{session_id}")
    postprocess_config = load_postprocessing_config(f"../data/pipeline_data/{participant_id}/{session_id}", False)

    # Get the event onset and time data for the session
    session_data = load_session_data(participant_id, session_id)

    # Separate the jatos data into blocks and trials
    jatos_blocks = parse_jatos_json(jatos_fpath)

    # Remove unused data from jatos events
    filter_unparsed_blocks(postprocess_config, jatos_blocks)

    # Check that the jatos and diode data have the same number of blocks
    assert len(jatos_blocks) == len(session_data.event_onsets_blocks)

    # Synchronize diode and jatos trial times
    jatos_block_dts = sync_trial_times(jatos_blocks, session_data.event_onsets_blocks)

    # Remove invalid trials
    jatos_blocks = filter_unsuccessful_trials(jatos_blocks)

    if plot_onsets:
        # Visually compare jatos and diode event onsets to ensure the first diode onset is correct
        plot_event_onset_comparison(jatos_blocks, jatos_block_dts, session_data)
    else:
        # Extract the jatos data for each trial
        jatos_session = extract_jatos_trial_data(jatos_blocks, jatos_block_dts, session_data)

        # Save the transformed hand data
        with open(filename, "wb") as f:
            pickle.dump(jatos_session, f)


def parse_jatos_json(jatos_path: str) -> list[list[dict[str, Any]]]:
    """Parses the session event data from a jatos JSON file.

    Parameters
        jatos_path (str): full file path to the jatos JSON file for a particular session

    Returns
        jatos_blocks (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
    """

    jatos_blocks = []
    with open(jatos_path) as f:
        for line in f:
            if len(json.loads(line)) > 80:
                json_data = json.loads(line)[4:]
                del_inds = [idx for idx, d in enumerate(json_data) if len(d) not in [13, 39]]
                for i in del_inds[::-1]:
                    del json_data[i]
                jatos_blocks.append(json_data[::2])

    # Confirm that there are 10 blocks of data for the session
    assert len(jatos_blocks) == 10

    return jatos_blocks


def filter_unparsed_blocks(
        postprocess_config: PostprocessingConfig,
        jatos_data: list[list[dict[str, Any]]],
) -> None:
    """Removes blocks from jatos JSON data, which could not be parsed or had parsing errors.

    Parsing errors are identified during preprocessing of the diode and pipeline processing of videos.

    Parameters
        postprocess_config (PostprocessingConfig): dataclass containing post-processing parameter values
        jatos_data (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
    """

    # Remove blocks that could not be parsed from the video/diode data
    postprocess_config.missing_blocks.sort()
    postprocess_config.missing_blocks.reverse()
    for i in postprocess_config.missing_blocks:
        del jatos_data[i]


def sync_trial_times(jatos_data: list[list[dict[str, Any]]], diode_onsets: list[np.ndarray]) -> list[float]:
    """Find the offset required to synchronize the jatos trial times with the diode data.

    The offset is equal to the difference between the end of the last AprilTag and the first event onset
    in each block.

    Parameters
        jatos_data (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
        diode_onsets (list[np.ndarray]): event onset times from diode data for each block

    Returns
        time_offsets (list[float]): offsets required to sync diode and jatos data for each block
    """

    time_offsets = []
    for i, jatos_block in enumerate(jatos_data):
        time_offsets.append((jatos_block[0]["startTime"] + jatos_block[0]["change_onset"]) / 1000 - diode_onsets[i][0])

    return time_offsets


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
        assert len(block_trials) == N_TRIALS

    return valid_trial_data


def plot_event_onset_comparison(
        jatos_data: list[list[dict[str, Any]]],
        block_dts: list[float],
        session_data: SessionData,
) -> None:
    """Plot a comparison of the jatos and diode event onset times, to ensure they are properly synchronized.

    Parameters
        jatos_data (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
        block_dts (list[float]): offsets required to sync diode and jatos data for each block
        session_data (SessionData): class containing the pipeline session data
    """

    for i, (jatos_block, t0, diode_onsets) in enumerate(zip(jatos_data, block_dts, session_data.event_onsets_blocks)):
        jatos_onsets = np.array([(trial["startTime"] + trial["change_onset"]) / 1000 for trial in jatos_block])
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.vlines(jatos_onsets - t0, 0, 1, 'r', label="JSON")
        ax.vlines(diode_onsets, 1, 2, 'b', label="Diode")
        ax.vlines(session_data.block_times[i][-1], 0, 2, 'g', label="Block End")
        ax.set_title(f"{session_data.participant_id}-{session_data.session_id}, Block {i}", fontsize=24)
        plt.legend()
        plt.show()
        plt.close()


def extract_jatos_trial_data(
        jatos_data: list[list[dict[str, Any]]],
        block_dts: list[float],
        session_data: SessionData,
) -> list[list[TrialData]]:
    """Extracts the data for individual trials from the jatos JSON data.

    The data extracted by this function will be used in later analyses.

    Parameters
        jatos_data (list[list[dict[str, Any]]]): parsed jatos data, separated into session blocks and trials
        block_dts (list[float]): offsets required to sync diode and jatos data for each block
        session_data (SessionData): class containing the pipeline session data

    Returns
        jatos_session (TrialData): individual trail data for each block and trial in the session
    """

    jatos_session = []
    block_itr = zip(jatos_data, session_data.reference_pos_abs, session_data.block_times, block_dts)
    for block_trials, ref_pos_block, block_time, t0 in block_itr:
        jatos_block_trials = []
        for j, trial_dict in enumerate(block_trials):
            # Populate the TrialData class instance
            trial_data = populate_trial_dataclass(trial_dict, session_data, t0)

            # Determine whether the trial is unusable
            trial_inds = get_trial_inds(trial_dict, t0, block_time)
            is_trial_usable(trial_data, trial_inds, block_time, ref_pos_block)

            jatos_block_trials.append(trial_data)
        jatos_session.append(jatos_block_trials)

    return jatos_session


def populate_trial_dataclass(
        trial_dict: dict[str, Any],
        session_data: SessionData,
        t0: float
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
        t0 (float): time offset between end of AprilTags and first event onset in diode data

    Returns
        trial_data (TrialData): class containing the individual trail data
    """

    # Calculate the pixel offsets for the center of the screen
    x_center = trial_dict["windowWidth"] / 2
    y_center = trial_dict["windowHeight"] / 2

    # Set the mid-point of the touch as the touch time
    if len(trial_dict["touchOff"]) == 6:
        touch_dts = np.stack(trial_dict["touchOn"]) - np.stack(trial_dict["touchOff"])
    else:
        touch_dts = np.zeros(6)
        touch_dts[:5] = np.stack(trial_dict["touchOn"])[:5] - np.stack(trial_dict["touchOff"])
        touch_dts[-1] = np.mean(touch_dts[:5])

    # Use the scheduled flash time as the flash onset
    try:
        dt_onset = trial_dict["flashOnTime"][0] / 1000
    except IndexError:
        print("No flash onset time (participant tapped all dots too quickly)")
        dt_onset = (trial_dict["touchOn"][5] - trial_dict["touchOn"][0]) / 1000

    # Populate the trial dataclass
    trial_data = TrialData(
        participant_id=session_data.participant_id,
        session_id=session_data.session_id,
        trial_onset_dt=dt_onset,
        trial_tap_times=(np.stack(trial_dict["touchOn"]) + touch_dts) / 1000 - t0,
        trial_tap_pos=np.stack(
            (np.array(trial_dict["touchX"]) - x_center, np.array(trial_dict["touchY"]) - y_center)),
        trial_dot_pos1=np.stack((trial_dict["position_x"], trial_dict["position_y"])),
        trial_dot_pos2=np.stack((trial_dict["shifted_position_x"], trial_dict["shifted_position_y"])),
        trial_flash=trial_dict["flashShown"],
        trial_shift=trial_dict["stimJumped"],
    )

    return trial_data


def get_trial_inds(trial_dict: dict[str, Any], t0: float, block_time: np.ndarray) -> np.ndarray:
    """Returns the indices of the block time vector, which correspond to a particular trial.

    Parameters
        trial_dict (dict[str, Any]): jatos JSON trial data
        t0 (float): time offset between end of AprilTags and first event onset in diode data
        block_time (np.ndarray): time vector for the entire block

    Returns
        (np.ndarray): indices in the block time vector associated with the trial
    """

    # Calculate the start and end times of the trial (first and last touch)
    dt = np.median(np.diff(block_time))
    t_start = trial_dict["touchOn"][0] / 1000 - t0
    t_end = trial_dict["touchOn"][5] / 1000 - t0

    # Check that the first touch and recorded trial start time are the same
    assert np.round(t_start, 3) == np.round(trial_dict["startTime"] / 1000 - t0, 3)

    return np.arange(block_time.size)[(t_start - dt <= block_time) & (block_time < t_end + dt)]


def is_trial_usable(
        trial_data: TrialData,
        trial_inds: np.ndarray,
        block_time: np.ndarray,
        ref_pos_block: dict[int, np.ndarray],
) -> None:
    """Determines whether the trial is usable.

    There are two situations where the trial is unuasable:
      1. The event onset occurs after the last tap.
      2. There are an unacceptable number of missed detections of the reference AprilTags.

    The event onset occurs after the last tap when the participat moves extremely quickly. This is
    an infrequent occurrence. On the other hand, missed AprilTag detections are much more common.
    Therefore, most unusable trials are due to this.

    Parameters
        trial_data (TrialData): class containing the individual trail data
        trial_inds (np.ndarray): indices in the block time vector associated with the trial
        block_time (np.ndarray): time vector for the entire block
        ref_pos_block (dict[int, np.ndarray]): coordinates of the reference AprilTags
    """

    if block_time[-1] < trial_data.trial_tap_times[-1]:
        trial_data.trial_usable = False
        return None

    for tag_id, reference_tag_pos in ref_pos_block.items():
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
        (TrialData): class containing the individual trail data
    """

    session_path = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_trial_data.pkl"
    if os.path.exists(session_path):
        with open(session_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def main(plot_onsets: bool):
    """Parses and extracts the required data from a jatos JSON file."""

    jatos_paths, jatos_files = get_files_containing("../data/original_data", "jatos")
    for jatos_path, jatos_file in zip(jatos_paths, jatos_files):
        # Get the participant and session IDs
        participant_id, session_id = jatos_path.split("/")[-2:]

        # Define the file path to the jatos data
        get_trial_data(participant_id, session_id, os.path.join(jatos_path, jatos_file), plot_onsets)


if __name__ == "__main__":
    main(False)
