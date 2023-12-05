import argparse
import matplotlib.pyplot as plt
from utils_preprocessing import preprocess_diode_data, separate_diode_blocks, get_event_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "participant_id",
        type=int
    )
    parser.add_argument(
        "session_id",
        type=str
    )
    parser.add_argument(
        "diode_suffix",
        type=str
    )
    args = parser.parse_args()
    # PARTICIPANT_ID = 17
    # SESSION_ID = "A1"
    # DIODE_SUFFIX = "01"
    # PARTICIPANT_ID = 2
    # SESSION_ID = "A1"
    # DIODE_SUFFIX = "2"

    diode_df = preprocess_diode_data(args.participant_id, args.session_id, args.diode_suffix)
    # diode_df = preprocess_diode_data(PARTICIPANT_ID, SESSION_ID, DIODE_SUFFIX)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(diode_df.time, diode_df.light_value)
    plt.show()
    plt.close()

    diode_df_blocks = separate_diode_blocks(diode_df, 80, 200)
    # diode_df_blocks = separate_diode_blocks(diode_df, 80)
    for block_df in diode_df_blocks:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(block_df.time, block_df.light_value)
        # plt.show()
        plt.close()
    # event_onsets = get_event_times(diode_df_blocks, 50)
