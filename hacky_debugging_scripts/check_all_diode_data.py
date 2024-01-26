"""Hacky script used to quickly visually check all diode data before processing."""

import os
import matplotlib.pyplot as plt
from utils.data_loading import get_files_containing
from scripts.data_preprocessing import format_diode_df


# https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
if __name__ == '__main__':
    # Get all diode files
    diode_paths, diode_files = get_files_containing("../data/original_data/", "light.csv")
    print(diode_files)

    session_files = [(diode_files[0], diode_paths[0])]
    session_str = diode_files[0][:5]
    participant_id = None
    session_id = None
    diode_suffix = None
    for file, fpath in zip(diode_files[1:], diode_paths[1:]):
        # if len(file) > 15:
        #     continue
        if session_str == file[:5]:
            session_files.append((file, fpath))
        else:
            fig, axes = plt.subplots(1, len(session_files), figsize=(8 * len(session_files), 5), sharey=True)
            for i, (session_file, session_path) in enumerate(session_files):
                diode_df = format_diode_df(os.path.join(session_path, session_file))
                if len(session_files) > 1:
                    axes[i].plot(diode_df.time, diode_df.light_value)
                    axes[i].set_xlabel("Time")
                    axes[0].set_ylabel("Diode Brightness")
                else:
                    axes.plot(diode_df.time, diode_df.light_value)
                    axes.set_xlabel("Time")
                    axes.set_ylabel("Diode Brightness")
            participant_id, session_id = session_file.split("_")[:2]
            fig.suptitle(f"Participant {participant_id}, Session {session_id}", fontsize=20)
            plt.show()
            plt.close()
            session_files = [(file, fpath)]
        session_str = file[:5]
    fig, axes = plt.subplots(1, len(session_files), figsize=(8 * len(session_files), 5))
    for i, (session_file, session_path) in enumerate(session_files):
        # diode_df = load_diode_data(participant_id, session_id)
        diode_df = format_diode_df(os.path.join(session_path, session_file))
        axes.plot(diode_df.time, diode_df.light_value)
        axes.set_xlabel("Time")
        axes.set_ylabel("Diode Brightness")
    participant_id, session_id = session_file.split("_")[:2]
    fig.suptitle(f"Participant {participant_id}, Session {session_id}", fontsize=20)
    plt.show()
