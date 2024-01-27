"""This script plots either the full diodes data for every session that has been preprocessed,
or (if proper arguments are given) it plots the full diode data for a single session, as well as
the individual block data (including extracted event onset and block end times)."""

import os
import sys
import argparse
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(".."))
from utils.split_diode_blocks import get_block_data
from utils.data_loading import get_files_containing, load_diode_data, load_session_config
# from utils_pipeline import SessionConfig


def plot_diode_data(participant_id: str = None, session_id: str = None) -> None:
    if participant_id is None or session_id is None:
        diode_paths, diode_files = get_files_containing("../data/pipeline_data", "diode_sensor.csv")
        for path, file in zip(diode_paths, diode_files):
            participant_id, session_id = file.split("_")[:2]
            diode_df = load_diode_data(participant_id, session_id)
            fig, ax = plt.subplots(1, 1, figsize=(16, 5))
            ax.plot(diode_df.time, diode_df.light_value)
            ax.set_xlabel("Time")
            ax.set_ylabel("Diode Brightness")
            ax.set_title(f"Participant {participant_id}, Session {session_id}", fontsize=20)
            plt.show()
            plt.close()
    else:
        diode_df = load_diode_data(participant_id, session_id)
        fig1, ax1 = plt.subplots(1, 1, figsize=(16, 6))
        ax1.plot(diode_df.time, diode_df.light_value)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Diode Brightness")
        ax1.set_title(f"Participant {participant_id}, Session {session_id}", fontsize=20)
        plt.show()
        plt.close()

        session_config = load_session_config(f"../data/pipeline_data/{participant_id}/{session_id}")
        get_block_data(
            diode_df,
            session_config.diode_threshold,
            session_config.separator_threshold,
            session_config.n_blocks,
            session_config.skip_valid_blocks,
            session_config.extra_apriltag_blocks,
            True,
        )
        plt.close(fig1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-pid", "--participant_id",
        type=str,
        default=None,
        help="Participant ID 'PXX'"
    )
    parser.add_argument(
        "-sid", "--session_id",
        type=str,
        default=None,
        help="Session ID, 'AX' or 'BX'"
    )
    args = parser.parse_args()

    plot_diode_data(args.participant_id, args.session_id)
