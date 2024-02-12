import os
import sys
import pickle
import argparse

sys.path.insert(0, os.path.abspath(".."))
from config.config_dataclasses import PostprocessingConfig
from utils.data_loading import get_files_containing, load_postprocessing_config
from data_extract_individual_trials import load_session_trials, TrialData


# TODO:
#  - Load the jatos trial data, and the hand speed data
#  - Remove any session blocks identified in config_post.json
#  - Separate hand position & speed data using the jatos trial start/end times (first and last tap times)
#    -> Use the trial start and end times to find the inds
#  - Group trials by experimental type (probably in a dict, with type as the key)


GROUPED_TRIALS_PATH = f"../data/combined_sessions/grouped_trials.pkl"


def load_grouped_trials(overwrite: bool) -> dict[str, list[str] | dict[str, list[TrialData]]]:
    """Loads the trial data, which is grouped across all participants and sessions.

    Parameters
        overwrite (bool): if True, initialize a new grouped data dict

    Returns
        (dict): TrialData instances for all existing sessions, grouped by experiment and change type
    """

    if not overwrite and os.path.exists(GROUPED_TRIALS_PATH):
        with open(GROUPED_TRIALS_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return {
            "Session List": [],
            "Experiment A": {
                "Shift and Flash": [],
                "Shift Only": [],
                "Flash Only": [],
                "No Change": [],
            },
            "Experiment B": {
                "Shift and Flash": [],
                "Shift Only": [],
                "Flash Only": [],
                "No Change": [],
            },
        }


def filter_blocks_with_errors(
        trial_data: list[list[TrialData]],
        postprocess_config: PostprocessingConfig,
) -> None:
    """Remove blocks that contain erroneous data.

    Parameters
        trial_data (list[list[TrialData]]): individual trail data for each block and trial in the session
        postprocess_config (PostprocessingConfig): dataclass containing post-processing parameter values
    """

    postprocess_config.skip_blocks.sort()
    postprocess_config.skip_blocks.reverse()
    for i in postprocess_config.skip_blocks:
        del trial_data[i]


def group_trials(
        session_trials: list[list[TrialData]],
        grouped_data: dict[str, list[str] | dict[str, list[TrialData]]],
        experiment_type: str,
        participant_session_uid: str
) -> None:
    """Group the trials from a session by the experiment type.

    There are two types of changes which may occur during an experiment (shift and flash), resulting
    in four possible experiment types (shift, flash, shift and flash, no change).

    Parameters
        session_trials (list[list[TrialData]]): individual trail data for each block and trial in the session
        grouped_data (dict): TrialData instances for all existing sessions, grouped by experiment and change type
        experiment_type (str): type of the experiment (either an A or B session)
        participant_session_uid (str): unique session identifier, combining the participant and session IDs
    """

    for block in session_trials:
        for trial in block:
            if trial.shift_change and trial.flash_change:
                grouped_data[f"Experiment {experiment_type}"]["Shift and Flash"].append(trial)
            elif trial.shift_change:
                grouped_data[f"Experiment {experiment_type}"]["Shift Only"].append(trial)
            elif trial.flash_change:
                grouped_data[f"Experiment {experiment_type}"]["Flash Only"].append(trial)
            else:
                grouped_data[f"Experiment {experiment_type}"]["No Change"].append(trial)
    grouped_data["Session List"].append(participant_session_uid)


def main(overwrite: bool):
    # Load or initialize the grouped trial dict
    grouped_data = load_grouped_trials(overwrite)

    fpaths, files = get_files_containing("../data/pipeline_data", "trial_data.pkl")
    for fpath, file in zip(fpaths, files):
        if "backup" in fpath:
            continue
        # Separate the participant and session IDs, and ensure they are valid
        participant_id, session_id = fpath.split("/")[-2:]
        experiment_type = session_id[0]
        if experiment_type not in ["A", "B"]:
            raise Exception("Invalid Session ID")

        # Check if the session trials are already in the saved data
        participant_session_uid = f"{participant_id}-{session_id}"
        if participant_session_uid in grouped_data["Session List"]:
            continue

        # Get the trial data for the session
        session_trials = load_session_trials(participant_id, session_id)

        # Group trials by experiment and change type
        group_trials(session_trials, grouped_data, experiment_type, participant_session_uid)

    # Save the transformed hand data
    with open(GROUPED_TRIALS_PATH, "wb") as f:
        pickle.dump(grouped_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="If True, overwrite the existing data"
    )
    args = parser.parse_args()

    main(args.overwrite)
