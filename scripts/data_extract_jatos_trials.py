import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from dataclasses import dataclass

from config.config_dataclasses import PostprocessingConfig
from utils_pipeline import load_session_data
from utils.split_diode_blocks import MIN_TRIAL_SEPARATION
from utils.data_loading import get_files_containing, load_postprocessing_config


@dataclass
class TrialData:
    participant_id: str
    session_id: str
    diode_event_onset: np.ndarray
    jatos_start_time: np.ndarray
    jatos_event_onset_dt: np.ndarray
    jatos_tap_time: list[np.ndarray]
    jatos_tap_pos: list[np.ndarray]
    jatos_dot_pos1: list[np.ndarray]
    jatos_dot_pos2: list[np.ndarray]
    jatos_flash: list[bool]
    jatos_jump: list[bool]


def get_trial_data(show_plot: bool = True) -> None:
    """Parses and extracts the required data from the jatos JSON file."""

    jatos_paths, jatos_files = get_files_containing("../data/original_data", "jatos")
    for jatos_path, jatos_file in zip(jatos_paths, jatos_files):
        # Load the post-processing config
        participant_id, session_id = jatos_path.split("/")[-2:]
        print(f"Parsing jatos file for session {participant_id}-{session_id}")
        postprocess_config = load_postprocessing_config(f"../data/pipeline_data/{participant_id}/{session_id}", False)

        # Extract the events
        jatos_blocks = parse_jatos_json(os.path.join(jatos_path, jatos_file))

        # Remove unused data from jatos events
        filter_unparsed_blocks(postprocess_config, jatos_blocks)

        # Get the event onset and time data for the session
        diode_onsets = get_diode_event_onsets(participant_id, session_id)

        # Check that the jatos and diode data have the same number of blocks
        assert len(jatos_blocks) == len(diode_onsets)

        # Only keep trials that are in both the jatos and diode data
        match_diode_jatos_trials(jatos_blocks, diode_onsets, participant_id, session_id, show_plot)

        # Extract the relevant data
        jatos_session = extract_jatos_trial_data(jatos_blocks, diode_onsets, participant_id, session_id)

        # Save the transformed hand data
        filename = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_trial_data.pkl"
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


def filter_unparsed_blocks(postprocess_config: PostprocessingConfig, jatos_events: list[list[dict[str, Any]]]):
    """Filters blocks from jatos JSON data, which could not be parsed or had parsing errors."""

    # Remove blocks that could not be parsed from the video/diode data
    postprocess_config.missing_blocks.sort()
    postprocess_config.missing_blocks.reverse()
    for i in postprocess_config.missing_blocks:
        del jatos_events[i]


def get_diode_event_onsets(participant_id: str, session_id: str) -> list[np.ndarray]:
    # Load the session data
    session_data = load_session_data(participant_id, session_id)
    diode_event_onsets = session_data.event_onsets_blocks

    return diode_event_onsets


def match_diode_jatos_trials(
        jatos_data: list[list[dict[str, Any]]],
        diode_onsets: list[np.ndarray],
        participant_id: str,
        session_id: str,
        show_plot: bool,
) -> None:
    for i, (jatos_data_block, diode_onsets_block) in enumerate(zip(jatos_data, diode_onsets)):
        # This assumes the jatos trial data is the ground truth
        if len(jatos_data_block) > diode_onsets_block.size:  # Event onsets were missed when parsing the diode data
            print(f"{participant_id}-{session_id}, Block {i} -> {len(jatos_data_block)} jatos trials > "
                  f"{diode_onsets_block.size} diode trials. Remove jatos trials missing from diode data.")

            # Begin both the jatos and diode block data at the first event onset (for comparison)
            jatos_onset_times, diode_onset_times = zero_trial_times(jatos_data_block, diode_onsets_block)

            # Determine which trials are in both the jatos and diode data
            matching_inds = get_matching_trial_inds(jatos_onset_times, diode_onset_times)
            jatos_data[i] = [jatos_data_block[idx] for idx in matching_inds]

            # Check by plotting the event onset times comparison
            if show_plot:
                matching_json_times = [jatos_onset_times[idx] for idx in matching_inds]
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                ax.vlines(matching_json_times, -1, 0, 'g', label="Diode")
                ax.vlines(jatos_onset_times, 0, 1, 'r', label="JSON")
                ax.vlines(diode_onset_times, 1, 2, 'b', label="Diode")
                ax.set_title(f"{participant_id}-{session_id}, Block {i}", fontsize=24)
                plt.legend()
                plt.show()
                plt.close()

        if len(jatos_data_block) < diode_onsets_block.size:  # The diode data has an extra event
            print(f"{participant_id}-{session_id}, Block {i} -> {len(jatos_data_block)} jatos trials < "
                  f"{diode_onsets_block.size} diode trials. Trim last diode trial.")

            # The extra diode trial was always found to be at the end of the block
            diode_onsets[i] = diode_onsets_block[:-1]


def zero_trial_times(
        jatos_data_block: list[dict[str, Any]],
        diode_onsets_block: np.ndarray,
) -> tuple[list[float], np.ndarray]:
    """Begin both the jatos and diode block data at the first event onset (for comparison)"""

    t0 = jatos_data_block[0]["startTime"] + jatos_data_block[0]["change_onset"]
    jatos_onset_times = [(trial["startTime"] + trial["change_onset"] - t0) / 1000 for trial in jatos_data_block]
    diode_onset_times = diode_onsets_block - diode_onsets_block[0]

    return jatos_onset_times, diode_onset_times


def get_matching_trial_inds(jatos_onset_times: list[float], diode_onset_times: np.ndarray) -> list[int]:
    matching_inds = []
    i_json = 0
    for t in diode_onset_times:
        if abs(t - jatos_onset_times[i_json]) < MIN_TRIAL_SEPARATION:  # No missed diode event
            matching_inds.append(i_json)
            i_json += 1
        elif abs(t - jatos_onset_times[i_json + 1]) < MIN_TRIAL_SEPARATION:  # One missed diode event
            matching_inds.append(i_json + 1)
            i_json += 2
        elif abs(t - jatos_onset_times[i_json + 2]) < MIN_TRIAL_SEPARATION:  # Two consecutive missed events
            matching_inds.append(i_json + 2)
            i_json += 3
        elif abs(t - jatos_onset_times[i_json + 2]) < MIN_TRIAL_SEPARATION:  # Three consecutive missed events
            matching_inds.append(i_json + 3)
            i_json += 4

    return matching_inds


def extract_jatos_trial_data(
        jatos_blocks: list[list[dict[str, Any]]],
        diode_onsets: list[np.ndarray],
        participant_id: str,
        session_id: str,
) -> list[TrialData]:
    trial_data_blocks = []
    for i, jatos_data in enumerate(jatos_blocks):
        t0 = jatos_data[0]["startTime"] + jatos_data[0]["change_onset"] - MIN_TRIAL_SEPARATION * 1000
        trial_data = TrialData(
            participant_id,
            session_id,
            diode_onsets[i] - diode_onsets[i][0] + MIN_TRIAL_SEPARATION,
            np.array([(trial["startTime"] - t0) / 1000 for trial in jatos_data]),
            np.array([trial["change_onset"] / 1000 for trial in jatos_data]),
            [np.array([(t - t0) / 1000 for t in trial["touchOn"]]) for trial in jatos_data],
            [np.stack((trial["touchX"], trial["touchY"])) for trial in jatos_data],
            [np.stack((trial["position_x"], trial["position_y"])) for trial in jatos_data],
            [np.stack((trial["shifted_position_x"], trial["shifted_position_y"])) for trial in jatos_data],
            [trial["stimJumped"] == 1 for trial in jatos_data],
            [trial["flashShown"] == 1 for trial in jatos_data],
        )
        trial_data_blocks.append(trial_data)

    return trial_data_blocks


if __name__ == "__main__":
    get_trial_data(False)
