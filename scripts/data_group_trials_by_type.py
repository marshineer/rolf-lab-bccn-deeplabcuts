import os
import sys
import pickle
import argparse

sys.path.insert(0, os.path.abspath(".."))
from utils.data_loading import get_files_containing
from data_extract_individual_trials import load_session_trials, TrialData


# dict[str, list[str] | dict[str, list[TrialData]]]
def load_grouped_trials(participant_id: str, overwrite: bool = False) -> dict[str, list | dict[str, list[TrialData]]]:
    """Loads the participant trial data, which is grouped across experiment conditions and sessions.

    Parameters
        filepath (str): filepath to the grouped trial dataclass
        overwrite (bool): if True, initialize a new grouped dataclass

    Returns
        (dict): TrialData instances for all existing sessions, grouped by experiment and change type
    """

    grouped_fpath = f"../data/pipeline_data/{participant_id}/{participant_id}_grouped_trial_data.pkl"
    if not overwrite and os.path.exists(grouped_fpath):
        with open(grouped_fpath, "rb") as f:
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


def group_participant_trials(
        session_trials: list[list[TrialData]],
        grouped_trials: dict[str, list | dict[str, list[TrialData]]],
        session_id: str,
) -> None:
    """Group the trials from a session by the experiment type.

    There are two types of changes which may occur during an experiment (shift and flash), resulting
    in four possible experiment types (shift, flash, shift and flash, no change).

    Parameters
        session_trials (list[list[TrialData]]): individual trail data for each block and trial in the session
        grouped_trials (ParticipantTrials): TrialData instances for sessions of a particular participant
    """

    for block in session_trials:
        for trial in block:
            if trial.shift_change and trial.flash_change:
                grouped_trials[f"Experiment {session_id[0]}"]["Shift and Flash"].append(trial)
            elif trial.shift_change:
                grouped_trials[f"Experiment {session_id[0]}"]["Shift Only"].append(trial)
            elif trial.flash_change:
                grouped_trials[f"Experiment {session_id[0]}"]["Flash Only"].append(trial)
            else:
                grouped_trials[f"Experiment {session_id[0]}"]["No Change"].append(trial)
    grouped_trials["Session List"].append(session_id)


def main(overwrite: bool):
    fpaths, files = get_files_containing("../data/pipeline_data", "trial_data.pkl", "grouped")
    participant_list = list(set([fpath.split("/")[-2] for fpath in fpaths]))
    participant_list.sort()
    for participant_id in participant_list:
        # Load or initialize the grouped trial dataclass
        grouped_fpath = f"../data/pipeline_data/{participant_id}/"
        grouped_trials = load_grouped_trials(participant_id, overwrite)

        session_paths, session_files = get_files_containing(grouped_fpath, "trial_data.pkl", "grouped")
        for fpath, file in zip(session_paths, session_files):
            # Load the trial data for the session
            session_id = fpath.split("/")[-1]
            session_trials = load_session_trials(participant_id, session_id)

            # Check if the session trials are already in the saved data
            if session_id in grouped_trials["Session List"]:
                continue

            # Group trials by experiment and change type
            print(f"Grouping {participant_id}-{session_id}")
            group_participant_trials(session_trials, grouped_trials, session_id)

        # Save the transformed hand data
        with open(os.path.join(grouped_fpath, f"{participant_id}_grouped_trial_data.pkl"), "wb") as f:
            pickle.dump(grouped_trials, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="If True, overwrite the existing data"
    )
    args = parser.parse_args()

    main(args.overwrite)
