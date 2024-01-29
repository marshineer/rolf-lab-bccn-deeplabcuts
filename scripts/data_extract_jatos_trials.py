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
from utils_pipeline import SessionData, load_session_data
from utils.data_loading import get_files_containing, load_postprocessing_config


N_TRIALS = 40
MAX_ZERO_DT = 6


@dataclass
class TrialData:
    participant_id: str
    session_id: str
    trial_time: np.ndarray
    trial_inds: np.ndarray
    trial_onset_dt: float
    trial_tap_times: np.ndarray
    trial_tap_pos: np.ndarray
    trial_dot_pos1: np.ndarray
    trial_dot_pos2: np.ndarray
    trial_flash: bool
    trial_jump: bool
    trial_usable: bool = True


def get_trial_data(plot_onsets: bool = True) -> None:
    """Parses and extracts the required data from the jatos JSON file."""

    jatos_paths, jatos_files = get_files_containing("../data/original_data", "jatos")
    for jatos_path, jatos_file in zip(jatos_paths, jatos_files):
        # Define the file name
        participant_id, session_id = jatos_path.split("/")[-2:]
        filename = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_trial_data.pkl"

        # Load the post-processing config
        print(f"Parsing jatos file for session {participant_id}-{session_id}")
        postprocess_config = load_postprocessing_config(f"../data/pipeline_data/{participant_id}/{session_id}", False)

        # Get the event onset and time data for the session
        session_data = load_session_data(participant_id, session_id)

        # Extract the events
        jatos_blocks = parse_jatos_json(os.path.join(jatos_path, jatos_file))

        # Remove unused data from jatos events
        filter_unparsed_blocks(postprocess_config, jatos_blocks)

        # Check that the jatos and diode data have the same number of blocks
        assert len(jatos_blocks) == len(session_data.event_onsets_blocks)

        # Synchronize diode and jatos trial times
        jatos_block_dts = sync_trial_times(jatos_blocks, session_data.event_onsets_blocks)

        # Remove invalid trials
        jatos_blocks = filter_unsuccessful_trials(jatos_blocks)

        # Visually compare jatos and diode event onsets to ensure the first diode onset is correct
        if plot_onsets:
            plot_event_onset_comparison(jatos_blocks, jatos_block_dts, session_data)

        # Extract the jatos data for each trial
        jatos_session = extract_jatos_trial_data(jatos_blocks, jatos_block_dts, session_data)

        # Save the transformed hand data
        with open(filename, "wb") as f:
            pickle.dump(jatos_session, f)


def parse_jatos_json(jatos_path: str):
    """Parses the session event data from a jatos JSON file."""

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
        # plot_onsets: bool
) -> None:
    """Filters blocks from jatos JSON data, which could not be parsed or had parsing errors."""

    # Remove blocks that could not be parsed from the video/diode data
    postprocess_config.missing_blocks.sort()
    postprocess_config.missing_blocks.reverse()
    for i in postprocess_config.missing_blocks:
        del jatos_data[i]


def sync_trial_times(jatos_data: list[list[dict[str, Any]]], diode_onsets: list[np.ndarray]) -> list[float]:
    """Find the offset required to synchronize the jatos trial times with the diode data."""

    time_offsets = []
    for i, jatos_block in enumerate(jatos_data):
        time_offsets.append((jatos_block[0]["startTime"] + jatos_block[0]["change_onset"]) / 1000 - diode_onsets[i][0])

    return time_offsets


def filter_unsuccessful_trials(jatos_data: list[list[dict[str, Any]]]):
    """Removes all trials that are not tagged as successful in the jatos JSON."""

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
    """Plot a comparison of the jatos and diode event onset times, to ensure they are properly synchronized."""

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


def get_jatos_onsets(jatos_data: list[list[dict[str, Any]]]) -> list[np.ndarray]:
    event_onsets = []
    for jatos_block in jatos_data:
        event_onsets.append(np.array([trial["change_onset"] / 1000 for trial in jatos_block]),)

    return event_onsets


def extract_jatos_trial_data(
        jatos_data: list[list[dict[str, Any]]],
        block_dts: list[float],
        session_data: SessionData,
) -> list[list[TrialData]]:

    jatos_session = []
    block_itr = zip(jatos_data, session_data.reference_pos_abs, session_data.block_times, block_dts)
    for block_trials, ref_pos_block, block_time, t0 in block_itr:
        jatos_block_trials = []
        for j, trial_dict in enumerate(block_trials):
            trial_data = populate_trial_dataclass(trial_dict, session_data, block_time, t0)

            # Remove any trials which are unusable
            filter_unusable_trials(trial_data, block_time, ref_pos_block)

            jatos_block_trials.append(trial_data)
        jatos_session.append(jatos_block_trials)

    return jatos_session


def populate_trial_dataclass(
        trial_dict: dict[str, Any],
        session_data: SessionData,
        block_time: np.ndarray,
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
        """

    # Get the trial indices
    trial_inds = get_trial_inds(trial_dict, t0, block_time)

    # Calculate the pixel offsets for the center of the screen
    x_center = trial_dict["windowWidth"] / 2
    y_center = trial_dict["windowHeight"] / 2

    # Populate the trial dataclass
    trial_data = TrialData(
        participant_id=session_data.participant_id,
        session_id=session_data.session_id,
        trial_time=block_time[trial_inds],
        trial_inds=trial_inds,
        trial_onset_dt=trial_dict["change_onset"] / 1000,
        trial_tap_times=np.stack(trial_dict["touchOn"]) / 1000 - t0,
        trial_tap_pos=np.stack(
            (np.array(trial_dict["touchX"]) - x_center, np.array(trial_dict["touchY"]) - y_center)),
        trial_dot_pos1=np.stack((trial_dict["position_x"], trial_dict["position_y"])),
        trial_dot_pos2=np.stack((trial_dict["shifted_position_x"], trial_dict["shifted_position_y"])),
        trial_flash=trial_dict["flashShown"],
        trial_jump=trial_dict["stimJumped"],
    )

    return trial_data


def get_trial_inds(trial_dict: dict[str, Any], t0: float, block_time: np.ndarray):
    # Calculate the start and end times of the trial (first and last touch)
    dt = np.median(np.diff(block_time))
    t_start = trial_dict["touchOn"][0] / 1000 - t0
    t_end = trial_dict["touchOn"][5] / 1000 - t0

    # Check that the first touch and recorded trial start time are the same
    assert int(t_start) == int(trial_dict["startTime"] / 1000 - t0)

    # return np.arange(block_time.size)[(t_start <= block_time) & (block_time < t_end)]
    return np.arange(block_time.size)[(t_start - dt <= block_time) & (block_time < t_end + dt)]


def filter_unusable_trials(trial_data: TrialData, block_time: np.ndarray, ref_pos_block: dict[int, np.ndarray]) -> None:
    # Determine whether the trial is unusable
    if block_time[-1] < trial_data.trial_tap_times[-1]:
        trial_data.trial_usable = False
        return

    for tag_id, reference_tag_pos in ref_pos_block.items():
        tag_zero_runs = find_zero_runs(reference_tag_pos[0, trial_data.trial_inds])
        if tag_zero_runs.size > 0:
            trial_data.trial_usable = False
            break


def find_zero_runs(data: np.ndarray) -> np.ndarray:
    """Finds the indices of consecutive zeros greater than a minimum length in a 1D array.

    Adapted from:
    https://stackoverflow.com/questions/24885092/finding-the-consecutive-zeros-in-a-numpy-array

    Each list entry is an array representing a different run of zeros, greater than the minimum length.
    The arrays contain all the indices of the zeros for that run.

    Parameters
        data (np.ndarray): data in which to find consecutive zeros

    Returns
        zero_inds (list[np.ndarray]): indices of the zero runs
    """

    # Create an array that is 1 where the data is 0, and pad each end with an extra 0
    iszero = np.concatenate(([0], np.equal(data, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))

    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    if ranges.size > 0:
        ranges = ranges[np.diff(ranges, axis=-1).squeeze() > MAX_ZERO_DT]

    return ranges


if __name__ == "__main__":
    get_trial_data(False)
