"""This script plots individual trials for a given block. It generates two figures.
The first figure shows the index fingertip position, including the change onset time
and the time of each finger tap. The second figure shows the same information in the
top subplot, as shows the total fingertip speed in the bottom plot. The plots for
each trial of the block are generated in sequence. When one plot is closed, the plot
for the next trial generates."""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(".."))
from utils.data_loading import load_postprocessing_config
from utils.pipeline import load_session_data, INDEX_FINGER_TIP_ID
from data_calculate_hand_speeds import (
    cubic_spline_filter,
    load_transformed_hand,
    TransformedHandData,
    DT_SPEED,
    SMOOTHING
)
from data_extract_individual_trials import load_session_trials, TrialData


if __name__ == "__main__":
    # Define an argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "participant",
        type=str,
        required=True,
        help="Unique participant identifier"
    )
    parser.add_argument(
        "session",
        choices=["A1", "A2", "B1", "B2"],
        type=str,
        required=True,
        help="Unique session identifier",
    )
    parser.add_argument(
        "block",
        type=int,
        required=True,
        help="Block number"
    )
    args = parser.parse_args()

    # Load the post-processing config
    postprocess_config = load_postprocessing_config(f"../data/pipeline_data/{args.participant}/{args.session}", False)

    # Load the session data
    session_data = load_session_data(args.participant, args.session)

    # Load the hand tracking data
    hand_pos_data: TransformedHandData = load_transformed_hand(args.participant, args.session)

    # Get the trial data for the session
    session_trials: list[list[TrialData]] = load_session_trials(args.participant, args.session)

    # Calculate the index fingertip speed
    block_itrs = zip(
        hand_pos_data.time_video,
        hand_pos_data.hand_position,
        session_data.event_onsets_blocks,
        session_trials,
    )
    for b, (block_time, hand_pos, onset_times, block_trials) in enumerate(block_itrs):
        print(f"{args.participant}-{args.session}, Block {b}")

        # Extract the time and fingertip position data
        x_pos, y_pos = hand_pos[INDEX_FINGER_TIP_ID][0, :], hand_pos[INDEX_FINGER_TIP_ID][1, :]
        time_interpolated = np.arange(0, block_time[-1], DT_SPEED)
        tap_times = np.hstack([trial.tap_times_on for trial in block_trials])
        jatos_onset_times = np.hstack([trial.tap_times_on + trial.tap_times_on[0] for trial in block_trials])

        # Calculate the fingertip speed
        x_spline = cubic_spline_filter(block_time, x_pos, time_interpolated, SMOOTHING)
        y_spline = cubic_spline_filter(block_time, y_pos, time_interpolated, SMOOTHING)
        nan_mask = np.isnan(x_spline)
        time_interpolated = time_interpolated[~nan_mask]
        x_spline = x_spline[~nan_mask]
        y_spline = y_spline[~nan_mask]
        x_speed = np.diff(x_spline) / np.diff(time_interpolated)
        y_speed = np.diff(y_spline) / np.diff(time_interpolated)
        total_speed = np.sqrt(x_speed ** 2 + y_speed ** 2)

        # Plot the position data and the fitted cubic spline
        fig0, ax0 = plt.subplots(1, 1, figsize=(10, 6))
        ax0.scatter(block_time, x_pos, label="X (Video)")
        ax0.scatter(block_time, y_pos, label="Y (Video)")
        ax0.vlines(onset_times, min(min(x_pos), min(y_pos)), max(max(x_pos), max(y_pos)), 'r', label="Onset Time")
        ax0.vlines(tap_times, min(min(x_pos), min(y_pos)), max(max(x_pos), max(y_pos)), 'g', label="Tap Times")
        ax0.plot(time_interpolated, x_spline, label=f"Cubic Spline")
        ax0.set_ylabel("Index Fingertip Positions [normalized pixels]", fontsize=14)
        # ax0.set_title(f"Cubic Spline Fitting of Hand Positions ({args.participant}-{args.session}, Block {b})",
        #                 fontsize=20)
        ax0.set_ylim([min(min(x_pos), min(y_pos)), max(max(x_pos), max(y_pos))])
        ax0.set_xlim([tap_times[0] - 0.3, tap_times[5] + 0.3])
        ax0.legend(loc=1)
        ax0.set_xlabel("Block Time [seconds]", fontsize=14)
        plt.show()

        # Plot the position and speed data
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        # Position
        ax[0].scatter(block_time, x_pos, label="Fingertip X (Video)")
        ax[0].scatter(block_time, y_pos, label="Fingertip Y (Video)")
        ax[0].vlines(onset_times, min(min(x_pos), min(y_pos)), max(max(x_pos), max(y_pos)), 'r', label="Onset Time")
        ax[0].vlines(tap_times, min(min(x_pos), min(y_pos)), max(max(x_pos), max(y_pos)), 'g', label="Tap Times")
        ax[0].plot(time_interpolated, x_spline, label=f"Cubic Spline")
        # Speed
        ax[1].plot(time_interpolated[:-1], total_speed, label=f"Fingertip Speed")
        ax[1].vlines(onset_times, min(total_speed), max(total_speed), 'r', label="Onset Time")
        ax[1].vlines(jatos_onset_times, min(total_speed), max(total_speed), 'b')
        ax[1].vlines(tap_times, min(total_speed), max(total_speed), 'g', label="Tap Times")

        # Set plot variables
        # Position plot
        ax[0].set_ylabel("Index Fingertip Positions", fontsize=14)
        # ax[0].set_title(f"Cubic Spline Fitting of Hand Positions ({args.participant}-{args.session}, Block {b})",
        #                 fontsize=20)
        ax[0].set_ylim([min(min(x_pos), min(y_pos)), max(max(x_pos), max(y_pos))])
        ax[0].set_xlim([tap_times[0] - 0.3, tap_times[5] + 0.3])
        ax[0].legend(loc=1)
        # Speed plot
        ax[1].set_xlabel("Block Time [seconds]", fontsize=14)
        ax[1].set_ylabel("Index Fingertip Total Speed", fontsize=16)
        ax[1].set_xlim([tap_times[0] - 0.3, tap_times[5] + 0.3])
        ax[1].set_ylim([0, 1000])
        ax[1].legend(loc=1)

        plt.show()
        plt.close()
