import argparse
import matplotlib.pyplot as plt
from utils import get_files_containing, load_diode_data
from data_preprocessing import separate_diode_blocks


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
    parser.add_argument(
        "-ht", "--high_threshold",
        type=int,
        default=200,
        help="High value light diode threshold"
    )
    parser.add_argument(
        "-at", "--apriltag_threshold",
        type=int,
        default=80,
        help="AprilTag light diode threshold"
    )
    parser.add_argument(
        "-et", "--event_threshold",
        type=int,
        default=50,
        help="Event onset light diode threshold"
    )
    args = parser.parse_args()

    if args.participant_id is None or args.session_id is None:
        diode_paths, diode_files = get_files_containing("data/pipeline_data/", "diode_sensor.csv")
        for path, file in zip(diode_paths, diode_files):
            participant_id, session_id = file.split("_")[:2]
            diode_df = load_diode_data(participant_id, session_id)
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.plot(diode_df.time, diode_df.light_value)
            ax.set_xlabel("Time")
            ax.set_ylabel("Diode Brightness")
            ax.set_title(f"Participant {participant_id}, Session {session_id}", fontsize=20)
            plt.show()
            plt.close()
    else:
        diode_df = load_diode_data(args.participant_id, args.session_id)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(diode_df.time, diode_df.light_value)
        plt.show()
        plt.close()

        diode_df_blocks = separate_diode_blocks(diode_df, args.apriltag_threshold, args.high_threshold)
        for block_df in diode_df_blocks:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(block_df.time, block_df.light_value)
            # plt.show()
            plt.close()
        # event_onsets = get_event_times(diode_df_blocks, args.event_threshold)
