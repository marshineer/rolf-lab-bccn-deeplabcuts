import os
import cv2
import json
import numpy as np
import pandas as pd
from config.config_dataclasses import PipelineConfig, SessionConfig, PostprocessingConfig


def get_files_containing(start_dir: str, string_match: str, string_exclude: str = "XXXXXXXXXX"):
    """Returns the files containing a particular string and their relative paths from the project root.

    File paths are relative from a given starting directory.

    References:
    - https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    - https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list

    Parameters
        start_dir (str): directory at which to start the walk
        string_match (str): only get files that contain this string
        string_exclude (str): only get files that do not contain this string

    Returns
        paths (list[str]): sorted list of file directory paths
        files (list[str]): sorted list of the file names
    """
    files = []
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(start_dir):
        for file in filenames:
            if string_match in file and string_exclude not in file:
                files.append(file)
                paths.append(dirpath)
    paths_sorted = [path for _, path in sorted(zip(files, paths))]
    files.sort()
    return paths_sorted, files


def load_diode_data(participant_id: str, session_id: str) -> pd.DataFrame:
    """Loads the preprocessed light diode sensor data as a dataframe.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        diode_df (pd.DataFrame): processed diode data for a single session
    """
    diode_path = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_diode_sensor.csv"
    return pd.read_csv(diode_path)


def load_video_time(participant_id: str, session_id: str) -> np.ndarray:
    """Loads the video time data, derived from the gaze data files.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        (np.ndarray): video frame timestamps for a single session
    """
    time_path = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}_video_time.csv"
    return pd.read_csv(time_path).to_numpy('float').squeeze()


def load_video_mp4(participant_id: str, session_id: str) -> cv2.VideoCapture:
    """Loads the preprocessed session video as an OpenCV video capture object.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        (cv2.VideoCapture): single video file for an experimental session
    """
    video_path = f"../data/pipeline_data/{participant_id}/{session_id}/{participant_id}_{session_id}.mp4"
    return cv2.VideoCapture(video_path)


def load_block_video_mp4(participant_id: str, session_id: str, block_id: int) -> cv2.VideoCapture:
    """Loads the pipeline processed block video as an OpenCV video capture object.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]
        block_id (int): block identifier

    Returns
        (cv2.VideoCapture): single video file for an experimental session
    """
    video_path = f"../data/pipeline_data/{participant_id}/{session_id}/block_videos/block_{block_id}.mp4"
    return cv2.VideoCapture(video_path)


def load_pipeline_config(print_config: bool = True) -> PipelineConfig:
    """Loads the pipeline configuration dataclass from a JSON file.

    Parameters
        config_path (str): file path to the configuration

    Returns
        (PipelineConfig): configuration dataclass for the entire pipeline
    """
    with open("../config/pipeline_config.json", "r") as fd:
        pipeline_settings = json.load(fd)
        if print_config:
            print(pipeline_settings)
        return PipelineConfig(**pipeline_settings)


def load_session_config(config_path: str, print_config: bool = True) -> SessionConfig:
    """Loads the session configuration dataclass from a JSON file.

    Parameters
        config_path (str): file path to the configuration

    Returns
        (SessionConfig): configuration dataclass for a particular session
    """
    with open(os.path.join(config_path, "config.json"), "r") as fd:
        session_settings = json.load(fd)
        if print_config:
            print(session_settings)
        return SessionConfig(**session_settings)


def load_postprocessing_config(config_path: str, print_config: bool = True) -> PostprocessingConfig:
    """Loads the post-processing configuration dataclass from a JSON file.

    Parameters
        config_path (str): file path to the configuration

    Returns
        (PostprocessingConfig): configuration dataclass for a particular session's post-processing
    """
    with open(os.path.join(config_path, "config_post.json"), "r") as fd:
        processing_settings = json.load(fd)
        if print_config:
            print(processing_settings)
        return PostprocessingConfig(**processing_settings)
