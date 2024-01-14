import os
import cv2
import time
import pickle
from pathlib import Path
from typing import NamedTuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe.python.solutions as mp
from pupil_apriltags import Detector, Detection

from utils import (
    get_block_data,
    get_fourcc,
    get_top_left_coords,
    load_diode_data,
    load_video_time,
    load_video_mp4,
)


# Experiment constants
N_APRILTAGS_MAX = 4

# AprilTag detection constants
PHONE_TAG_IDS = [1, 2, 3]
APPARATUS_TAG_IDS = [40, 30, 10]
N_FRAME_BUFFER = 6  # 30 fps -> 3 frames ~ 0.1s
MIN_SET_LEN = 8.9
MAX_SET_LEN = 9.3

# Hand tracking constants
INDEX_FINGER_TIP_IDX = 8
HAND_LANDMARK_CONNECTIONS = mp.hands.HAND_CONNECTIONS


@dataclass()
class PipelineConfig:
    save_data: bool
    show_video: bool
    visual_plot_check: bool
    overwrite_data: bool
    apriltag_kwargs: dict[str, float]
    mediapipe_kwargs: dict[str, int | float]
    tracked_hand_landmarks: dict[str, int]


@dataclass()
class SessionConfig:
    participant_id: str
    session_id: str
    n_blocks: int
    diode_threshold: int
    separator_threshold: int | None
    skip_valid_blocks: list[int] = field(default_factory=list)
    extra_apriltag_blocks: list[int] = field(default_factory=list)


class SessionData:
    """This class stores all the data processed by the pipeline.

    References
    https://pupil-apriltags.readthedocs.io/en/stable/api.html
    https://juliarobotics.org/AprilTags.jl/latest/
    https://mediapipe.readthedocs.io/en/latest/solutions/hands.html

    Attributes
        participant_id (str): unique participant identifier
        session_id (str): session identifier
        saved_block_inds (list[int]): indices of the blocks for data that has already been saved
        last_frame_saved (int): last frame included in the saved data
        n_frames (int): number of frames in the video
        tracked_landmark_ids (list[int]): IDs of the tracked hand landmarks
        tracked_landmark_names (list[str]): names of the tracked hand landmarks
        apriltag_kwargs (dict[str, float]): key word arguments for the AprilTag detector class
        mediapipe_kwargs (dict[str, int | float]): key word arguments for hand tracker class
        video_time (np.ndarray): video frame timestamps for the full-length video
        block_frames (list[tuple[int, int]]): first and last frames of each block
        block_times (list[np.ndarray]): video frame timestamps for each block
        apriltag123_visible (list[np.ndarrhay]): array indicating in which frames AprilTags are visible on the phone
        first_trial_frame (list[int]): last video frome of the five AprilTag set, used to trim hand position data
        hand_landmark_pos_abs (list[dict[int, np.ndarray]]): absolute position data of the hand landmarks
        reference_pos_abs (list[dict[int, np.ndarray]]): absolute position of the apparatus AprilTags (IDs 40, 30, 10)
        hand_landmark_pos_trans (list[dict[int, np.ndarray]]): transformed position data of the hand landmarks
        reference_pos_rot (list[np.ndarray]): rotated and interpolated position of reference AprilTag's top-left corner
        diode_df_blocks (list[pd.DataFrame]): light diode sensor data, collected from the phone screen
        event_onsets (list[np.ndarray]): event onset times for each block
    """

    def __init__(
            self,
            participant_id: str,
            session_id: str,
            video_time: np.ndarray,
            tracked_landmarks: dict[str, int],
            diode_df_blocks: list[pd.DataFrame],
            event_onsets: list[np.ndarray],
            apriltag_kwargs: dict[str, float],
            mediapipe_kwargs: dict[str, int | float],
    ):

        # Session parameters
        self.participant_id: str = participant_id
        self.session_id: str = session_id
        self.saved_block_inds: list[int] = []
        self.last_frame_saved: int = -1
        self.n_frames: int = video_time.size
        self.tracked_landmark_ids: list[int] = list(tracked_landmarks.values())
        self.tracked_landmark_names: list[str] = list(tracked_landmarks.keys())
        self.apriltag_detector_kwargs: dict[str, float] = apriltag_kwargs
        self.mediapipe_kwargs: dict[str, int | float] = mediapipe_kwargs

        # Full video data
        self.video_time: np.ndarray = video_time

        # Block specific data
        self.block_frames: list[tuple[int, int]] = []
        self.block_times: list[np.ndarray] = []
        self.apriltag123_visible: list[np.ndarray] = []
        self.first_trial_frame: list[int] = []
        self.hand_landmark_pos_abs: list[dict[int, np.ndarray]] = []
        self.reference_pos_abs: list[dict[int, np.ndarray]] = []
        self.hand_landmark_pos_trans: list[dict[int, np.ndarray]] = []
        self.reference_pos_rot: list[np.ndarray] = []
        self.diode_df_blocks: list[pd.DataFrame] = diode_df_blocks
        self.event_onsets_blocks: list[np.ndarray] = event_onsets

    def relative_hand_positions(self, show_plot: bool) -> None:
        """Calculates hand positions relative to a reference point.

        Since the head-mounted camera is constantly moving, the hand positions are calculated relative
        to a reference point. In this case, the top-left corner of the top-left AprilTag that is
        affixed to the apparatus frame is chosen. It was chosen because it is considered least likely
        to be covered during the trials (assuming the two_part_videos use their right index finger).

        Since the participant's hands are often off camera while the AprilTags are shown on the phone,
        this method trims the hand landmark and reference position data to begin when the last AprilTag
        disappears. Furthermore, the data is checked for frame where the hands or reference position is
        not tracked, and interpolates the positions for these frames. This assumption is reasonable when
        only a few consecutive frames are lost, due to the smoothness/continuity of video position in
        time.

        Parameters
            shot_plot (bool): if True, plot the relative hand landmark positions
        """

        # itr_lists = zip(
        #     self.hand_landmark_pos_abs,
        #     self.reference_pos_abs,
        #     self.block_times,
        #     self.first_trial_frame
        # )
        itr_lists = zip(self.hand_landmark_pos_abs, self.reference_pos_abs, self.block_times, self.first_trial_frame)
        for lm_pos, ref_pos, time_vec, ind0 in itr_lists:
            # TODO: Determine which AprilTag is the best reference
            reference_tag_id = self.get_reference_id(ref_pos)
            # Interpolate the missing reference AprilTag positions
            # # ref_x, ref_y = self.interpolate_pos(time_vec, ref_xy_rot)
            # ref_x, ref_y = self.interpolate_pos(time_vec, ref_pos[reference_tag_id])
            interp_ref_pos = {}
            for tag_id in APPARATUS_TAG_IDS:
                interp_ref_pos[tag_id] = np.zeros((2, time_vec))
                interp_ref_pos[tag_id] = self.interpolate_pos(time_vec, ref_pos[tag_id])
                # ref_x, ref_y = self.interpolate_pos(time_vec, ref_pos[tag_id])
                # interp_ref_pos[tag_id] = np.stack((ref_x, ref_y))
            # TODO: Rotate the frame to be square to the apparatus' AprilTags
            rotation_matrices, out_shape = self.get_rotation_mat(interp_ref_pos, reference_tag_id)

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            relative_lm_pos = {}
            for lm, lbl in zip(self.tracked_landmark_ids, self.tracked_landmark_names):
                # Interpolate and calculate hand landmark positions, relative to the reference
                # lm_x, lm_y = self.interpolate_pos(time_vec, lm_pos[lm])
                lm_xy = self.interpolate_pos(time_vec, lm_pos[lm])
                # TODO: Plot the interpolated position data to show the off-axis skew
                # TODO: Do the translation first
                #  - I want to move the origin, then rotate about the new origin to align with the frame
                # https://gamedev.stackexchange.com/questions/16719/what-is-the-correct-order-to-multiply-scale-rotation-and-translation-matrices-f
                # https://learn.microsoft.com/en-us/dotnet/desktop/winforms/advanced/why-transformation-order-is-significant?view=netframeworkdesktop-4.8#:~:text=The%20matrix%20multiplication%20is%20done,%2C%20then%20rotate%2C%20then%20translate.
                # rel_x, rel_y = lm_x[ind0:] - ref_x[ind0:], lm_y[ind0:] - ref_y[ind0:]
                # relative_lm_pos[lm] = np.stack((lm_x[ind0:] - ref_x[ind0:], lm_y[ind0:] - ref_y[ind0:]))
                rel_xy = lm_xy[:, ind0:] - interp_ref_pos[reference_tag_id][:, ind0:]
                relative_lm_pos[lm] = np.matmul(rel_xy, rotation_matrices, out=out_shape)
                # TODO: Apply the rotation to the hand landmark positions
                #  - use np.matmul() to multiply the (n, m, p) rotation matrix by the (n, p) position data in one step
                if show_plot:
                    ax.plot(time_vec[ind0:], self.hand_landmark_pos_trans[lm][0, :], label=f"X: {lbl}")
                    ax.plot(time_vec[ind0:], self.hand_landmark_pos_trans[lm][1, :], label=f"Y: {lbl}")
            ax.legend()
            if show_plot:
                plt.show()
            plt.close()
            self.hand_landmark_pos_trans.append(relative_lm_pos)

    # TODO: Complete the logic for this function
    @staticmethod
    def get_reference_id(reference_tag_pos: dict[int, np.ndarray]) -> int:
        """Determines which AprilTag attached to the frame is visible in the most frames."""

        # Check time series for all AprilTags
        # If tag 40 is visible for most of the session (and has no gaps bigger than X frames), interpolate and use it
        # Otherwise, determine which tag has the fewest missing frames
        reference_tag_id = APPARATUS_TAG_IDS[0]
        return reference_tag_id

    @staticmethod
    # def interpolate_pos(time_data: np.ndarray, position_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def interpolate_pos(time_data: np.ndarray, position_data: np.ndarray) -> np.ndarray:

        """Interpolates the hand landmark and reference position data.

        Parameters
            time_data (np.ndarray): the timestamp for each block frame
            position_data (np.ndarray): the x-y position for each block frame

        Returns
            x_interp (np.ndarray): the interpolated x values of the data
            y_interp (np.ndarray): the interpolated y values of the data
            y_interp (np.ndarray): the interpolated x- and y-values of the data
        """

        # TODO: what happens when there are big gaps in the data (i.e. AprilTags missing for many frames)?
        min_value = 0.1
        interp_mask = (position_data[0, :] > min_value) & (position_data[1, :] > min_value)
        x_interp = np.interp(time_data, time_data[interp_mask], position_data[0, :][interp_mask])
        y_interp = np.interp(time_data, time_data[interp_mask], position_data[1, :][interp_mask])
        # TODO: maybe stack x and y before returning (I think in all cases, they need to be stacked anyway)
        return np.stack((x_interp, y_interp))
        # return x_interp, y_interp

    # TODO: Complete the logic for this function
    @staticmethod
    def get_rotation_mat(
            reference_tag_pos: dict[int, np.ndarray],
            reference_tag_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate and apply rotation matrix to x-y coordinate data."""

        # Calculate the rotation matrix, based on the top-left corner of three AprilTags (40, 30, 10)
        pos_shape = reference_tag_pos[reference_tag_id].shape
        # rot_ref_pos = np.zeros(pos_shape)
        rotation_matrices = np.zeros((pos_shape[1], pos_shape[0], pos_shape[0]))
        for i in range(pos_shape[1]):
            frame_i_pos = {}
            for tag_id in APPARATUS_TAG_IDS:
                frame_i_pos[tag_id] = reference_tag_pos[tag_id][:, i]
            x_vec = np.ndarray([frame_i_pos[10][0] - frame_i_pos[40][0], frame_i_pos[10][1] - frame_i_pos[40][1]])
            y_vec = np.ndarray([frame_i_pos[30][0] - frame_i_pos[40][0], frame_i_pos[30][1] - frame_i_pos[40][1]])
            rot_mat = np.identity(x_vec.size)
            rotation_matrices[i, :, :] = rot_mat
            # rot_ref_pos[i, :] = reference_tag_pos[reference_tag_id][i] @ rot_mat  # TODO: check this order is correct

        # TODO: I think there's a better way to format this data
        #  - I still need to rotation the reference position after calculating the rotation matrix,
        #       but I can't hard code which tag_id corresponds to the reference
        # itr_pos = zip(
        #     reference_tag_pos[40][0, :],
        #     reference_tag_pos[40][1, :],
        #     reference_tag_pos[30][0, :],
        #     reference_tag_pos[30][1, :],
        #     reference_tag_pos[10][0, :],
        #     reference_tag_pos[10][1, :],
        # )
        # for x_40, y_40, x_30, y_30, x_10, y_10 in itr_pos:
        #     # Calculate the rotation matrix for each frame
        #     x_vec = np.ndarray([x_10 - x_40, y_10 - y_40])
        #     y_vec = np.ndarray([x_30 - x_40, y_30 - y_40])
        #     rot_mat = np.identity(x_vec.size)
        #     # Apply the rotation matrix to the reference coordinates
        #     pass
        return rotation_matrices, np.zeros(pos_shape)

    # TODO: When should smoothing be done? After finding relative position?
    @staticmethod
    def smooth_pos_data():
        pass

    @staticmethod
    def calculate_velocity():
        pass


@dataclass
class BlockVariables:
    block_id: int = -1
    frame_0: int = -1
    frame_1: int = -1
    frame_cnt: int = -1
    time: np.ndarray = None
    block_video: cv2.VideoWriter = None

    apriltag_cnt: int = -1
    n_apriltag_max: int = N_APRILTAGS_MAX
    apriltag_detected: bool = False
    apriltag_last_frame: int = -1

    event_onsets: np.ndarray = None
    event_ind: int = -1


class VideoProcessingPipeline:
    """Processes a session video and extracts hand positions from the participant's head-mounted camera.

    Preprocessing steps
      - Split the diode light value data into experimental blocks:
        -> Extract AprilTag on/off times, event onset times and block end times (t0 = start of block).
        -> (Optional) plot each block to visually check performance of time extraction.
      - Preprocess (clean) the gaze data:
        -> Interpolate missing rows in the gaze data (assumes there should be no missing world_index values).
        -> Dropped frames locations are identified by time differences that are an approximate multiple of a
          single median time step. These dropped frames are ignored for now.

    Video processing steps
      - Detect AprilTags:
        -> AprilTags on the phone screen (IDs: 1, 2, 3) are used to identify the start of an experiment block.
        -> AprilTags on the apparatus (IDs: 10, 20, 30, 40) provide a reference point for hand positions.
      - Track hands:
        -> Uses the mediapipe library to track hands by landmark (eg. INDEX_FINGER_TIP) positions.
        -> Hand positions are given relative to the AprilTags on the surrounding apparatus.
        -> This positional data is saved for later processing.
      - Add markup to frame (optional):
        -> This can be done to visually check synchronization.
        -> Visual markers are added to the frame, to align diode events with video events.
        -> Synchronization is good if the markers appear at the proper times.
          >> A blue square should cover the top left AprilTags on the phone, when it is visible.
          >> A white circle should flash at the moment of an event onset.

    Note: The video frames are written as separate blocks.
      - Requires syncing the video and diode data streams using the first AprilTag occurrence in each block.

    Attributes
        participant_id (str): unique participant identifier
        session_id (str): session identifier
        save_data (bool): whether to save the individual block videos and session data
        show_video (bool): whether to stream the video during processing
        show_plots (bool): whether to display plots as visual checks on the process
        data_path (str): relative file path to which the session data is saved
        overwrite_session (bool): if True, overwrite any existing saved session data
        video_capture: (cv2.VideoCapture): mp4 video to be processed
        pixel_width (int): video width, in pixels
        pixel_height (int): video height, in pixels
        fps (float): video frames rate (frames per second)
        fourcc_codec (str): four character code (fourcc) of the mp4 video file
        n_frames (int): number of frames in the video
        valid_block_inds (list[int]): indices in the block data corresponding to valid experiment blocks (>=40 trials)
        diode_df_blocks (list[pd.DataFrame]): light diode sensor data, collected from the phone screen
        event_onset_blocks (list[np.ndarray]): event onset times for each block
        block_cnt (int): index of the block currently being processed (either valid or invalid)
        video_time (np.ndarray): video frame timestamps
        apriltag_detector (Detector): AprilTag detection object
        tracked_landmarks (dict[str, int]): hand landmark names (keys) and their mediapipe IDs (values)
        hands_model (mp.hands.Hands): mediapipe hand tracking object
        hand_ind (int): index of the hand to draw and save position data for
    """

    def __init__(self, pipeline_config: PipelineConfig, session_config: SessionConfig):
        """"
        Parameters
            pipeline_config (PipelineConfig)
                save_data (bool): whether to save the individual block videos and session data
                show_video (bool): whether to stream the video during processing
                visual_plot_check (bool): whether to display plots as visual checks on the process
                overwrite_data (bool): if True, overwrite any existing saved session data
                apriltag_kwargs (dict[str, float]): key word arguments for the AprilTag detector class
                mediapipe_kwargs (dict[str, int | float]): key word arguments for hand tracker class
                tracked_hand_landmarks (dict[str, int]): all mediapipe hand landmark indices to be tracked
            session_config (SessionConfig)
                participant_id (str): unique participant identifier
                session_id (str): session identifier
                n_blocks (int): number of valid blocks in the saved data
                diode_threshold (int): light diode threshold for all events, AprilTags and separators
                separator_threshold (int | None): light diode threshold block separation signals
        """

        # Processing parameters
        self.participant_id: str = session_config.participant_id
        self.session_id: str = session_config.session_id
        self.save_data: bool = pipeline_config.save_data
        self.show_video: bool = pipeline_config.show_video
        self.show_plots: bool = pipeline_config.visual_plot_check

        # Session file parameters
        data_dir = f"data/pipeline_data/{session_config.participant_id}/{session_config.session_id}"
        data_file = f"{session_config.participant_id}_{session_config.session_id}_pipeline_data.pkl"
        self.data_path = os.path.join(data_dir, data_file)
        self.overwrite_session = pipeline_config.overwrite_data

        # Initialize a video capture object and store its parameters
        video_capture = load_video_mp4(session_config.participant_id, session_config.session_id)
        self.video_capture: cv2.VideoCapture = video_capture
        self.pixel_width: int = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.pixel_height: int = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = video_capture.get(cv2.CAP_PROP_FPS)
        self.fourcc_codec: int = cv2.VideoWriter_fourcc(*get_fourcc(video_capture))
        self.n_frames: int = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Split the diode data into blocks
        diode_df = load_diode_data(session_config.participant_id, session_config.session_id)
        block_data, block_inds, block_events = get_block_data(
                                                    diode_df,
                                                    session_config.diode_threshold,
                                                    session_config.separator_threshold,
                                                    session_config.n_blocks,
                                                    skip_blocks=session_config.skip_valid_blocks,
                                                    extra_apriltag_blocks=session_config.extra_apriltag_blocks,
                                                )
        self.valid_block_inds: list[int] = block_inds
        self.diode_df_blocks: list[pd.DataFrame] = block_data
        self.event_onset_blocks: list[np.ndarray] = block_events
        self.block_cnt: int = -1

        # Preprocess the gaze data
        self.video_time: np.ndarray = load_video_time(session_config.participant_id, session_config.session_id)

        # AprilTag detection variables
        self.apriltag_detector: Detector = Detector(**pipeline_config.apriltag_kwargs)
        self.apriltag_kwargs: dict[str, float] = pipeline_config.apriltag_kwargs
        self.extra_apriltag_blocks = session_config.extra_apriltag_blocks

        # Hand tracking variables
        self.tracked_landmarks: dict[str, int] = pipeline_config.tracked_hand_landmarks
        self.hands_model: mp.hands.Hands = mp.hands.Hands(**pipeline_config.mediapipe_kwargs)
        self.mediapipe_kwargs: dict[str, int | float] = pipeline_config.mediapipe_kwargs
        self.hand_ind = None

    def iterate_video_frames(self) -> SessionData:
        """Stores hand positions for all video frames, and writes block videos with markers.

        Returns
            session_data (SessionData): class object containing all relevant experiment session data
        """

        # Initialize variables for analyzing video
        video_frame_cnt = -1

        # Initialize the classes used to store experimental data
        block_vars = BlockVariables()
        if self.overwrite_session or not os.path.exists(self.data_path):
            session_data = SessionData(
                self.participant_id,
                self.session_id,
                self.video_time,
                self.tracked_landmarks,
                self.diode_df_blocks,
                self.event_onset_blocks,
                self.apriltag_kwargs,
                self.mediapipe_kwargs,
            )
        else:
            with open(self.data_path, "rb") as f:
                session_data = pickle.load(f)
            self.block_cnt = session_data.saved_block_inds[-1]
            block_vars.block_id = len(session_data.saved_block_inds) - 1
            block_vars.apriltag_cnt = N_APRILTAGS_MAX
        print(f"Valid block indices: {self.valid_block_inds}")

        run_time_t0 = time.time()
        run_time_t0_5000 = time.time()
        while self.video_capture.isOpened():
            # Read the next frame
            ret, frame = self.video_capture.read()
            video_frame_cnt += 1
            if video_frame_cnt < session_data.last_frame_saved:
                continue
            block_vars.frame_cnt += 1
            if video_frame_cnt == self.n_frames:
                print("End of video file.")
                break
            elif not ret:
                print("Error retrieving frame. Exiting.")
                break
            if video_frame_cnt % 5000 == 0:
                print(f"Current frame: {video_frame_cnt}/{self.n_frames}, "
                      f"runtime: {time.time() - run_time_t0_5000:0.2f} seconds")
                run_time_t0_5000 = time.time()

            # Convert frame from BGR to grayscale and RGB
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_out = frame

            # Detect AprilTags
            tags = self.detect_apriltags_phone(
                frame_grayscale,
                video_frame_cnt,
                block_vars,
                session_data,
            )

            # If the frame is in a block
            block_frame_range = block_vars.frame_1 - block_vars.frame_0
            if block_vars.frame_cnt < block_frame_range and self.block_cnt in self.valid_block_inds:
                # Track hands
                # TODO:
                #  - hand tracking is super slow, see if it can be sped up
                #    -> https://stackoverflow.com/questions/74048393/can-i-speed-up-processing-live-video-from-webcam
                tracking_results = self.hands_model.process(frame_rgb)
                self.get_hand_position(
                    block_vars.frame_cnt,
                    tracking_results,
                    session_data
                )

                # Update reference points
                for tag in tags:
                    if tag.tag_id in APPARATUS_TAG_IDS:
                        xy_corner = get_top_left_coords(tag.corners)
                        session_data.reference_pos_abs[-1][tag.tag_id][:, block_vars.frame_cnt] = xy_corner

                # Draw hand landmarks and connections in frame
                if self.save_data or self.show_video:
                    self.markup_video_frame(
                        frame_rgb,
                        tracking_results,
                        block_vars,
                        tags,
                    )
                    frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Write blocks to new video
                if self.save_data:
                    if block_vars.block_id >= 0:
                        block_vars.block_video.write(frame_out)

            # Stream the frame (as video)
            if self.show_video:
                cv2.imshow('frame', frame_out)
                if cv2.waitKey(1) == ord('q'):
                    break

            # Save the block
            if video_frame_cnt == block_vars.frame_1:
                if self.save_data:
                    self.save_block_data(video_frame_cnt, block_vars, session_data)
                if self.show_plots:
                    self.plot_last_block_data(block_vars, session_data)

        print(f"It took {time.time() - run_time_t0:0.2f} seconds to iterate through the entire video")
        print(f"Reported frames: {self.n_frames}")
        print(f"Counted frames: {video_frame_cnt}")
        if self.n_frames == video_frame_cnt:
            print("No corrupted frames! Hooray!")
        self.video_capture.release()
        cv2.destroyAllWindows()

        return session_data

    def detect_apriltags_phone(
            self,
            frame_grey: np.ndarray,
            video_frame_cnt: int,
            block_vars: BlockVariables,
            session_data: SessionData,
    ) -> list[Detection]:
        """Detects the AprilTags in a video frame.

        If a new AprilTag is found, one of three things happens:
            1. The AprilTag counter is incremented.
            2. If it is the same AprilTag as was previously detected, this is determined.
            3. If it is the first AprilTag in a block, the block variables are reset.

        If the end of the 5th (or 6th) AprilTag is detected at approximately the correct time,
        it is a signal that the trials are about to begin.

        Parameters
            frame_grey (np.ndarray): the current video frame, converted to grayscale
            video_frame_cnt (int): the number of the current video frame
            block_vars (BlockVariables): all variables relevant to each block
            session_data (SessionData): all variables tracked for the session

        Returns
            tags (list[Detection]): a list of AprilTags detected in the current video frame
        """

        tags = self.apriltag_detector.detect(frame_grey)
        for tag in tags:
            if tag.tag_id in PHONE_TAG_IDS:  # AprilTag detected on phone screen
                if not block_vars.apriltag_detected:  # New AprilTag onset
                    if video_frame_cnt - block_vars.apriltag_last_frame < N_FRAME_BUFFER:  # Missed AprilTag detection
                        self.missed_apriltag_detection(frame_grey)
                    elif block_vars.apriltag_cnt == block_vars.n_apriltag_max or self.block_cnt < 0:  # New AprilTag set
                        self.block_cnt += 1
                        if self.block_cnt in self.valid_block_inds:  # Reset block data
                            self.reset_block_data(
                                video_frame_cnt,
                                block_vars,
                                session_data
                            )
                        else:
                            block_vars.apriltag_cnt = 0
                            block_vars.n_apriltag_max = N_APRILTAGS_MAX
                            block_vars.frame_0 = video_frame_cnt
                        if self.block_cnt in self.extra_apriltag_blocks:
                            block_vars.n_apriltag_max = N_APRILTAGS_MAX + 1
                    else:  # Next AprilTag in set
                        block_vars.apriltag_cnt += 1
                        assert block_vars.apriltag_cnt <= block_vars.n_apriltag_max
                block_vars.apriltag_detected = True
                block_vars.apriltag_last_frame = video_frame_cnt
                if self.block_cnt in self.valid_block_inds:
                    session_data.apriltag123_visible[-1][block_vars.frame_cnt] = 1
            else:
                # No AprilTag detected on phone
                if block_vars.apriltag_cnt == N_APRILTAGS_MAX and block_vars.apriltag_detected:
                    # If it is the 5th AprilTag, check that the timing is approximately correct
                    tag_time = session_data.video_time[block_vars.frame_0:video_frame_cnt]
                    print(f"Duration of AprilTag set: {tag_time[-1] - tag_time[0]:0.3f} seconds")
                    assert MIN_SET_LEN < (tag_time[-1] - tag_time[0]) < MAX_SET_LEN
                    block_vars.event_ind = 0
                    if self.block_cnt in self.valid_block_inds:
                        session_data.first_trial_frame.append(block_vars.frame_cnt)
                block_vars.apriltag_detected = False
            break
        return tags

    def reset_block_data(
            self,
            video_frame_cnt: int,
            block_vars: BlockVariables,
            session_data: SessionData,
    ) -> None:
        """Resets the block variables when a new block is found.

        Parameters
            video_frame_cnt (int): the number of the current video frame
            block_vars (BlockVariables): all variables relevant to each block
            session_data (SessionData): all variables tracked for the session
        """

        # Block counters
        block_vars.frame_cnt = 0
        block_vars.block_id += 1
        assert block_vars.block_id <= len(self.valid_block_inds)
        block_vars.apriltag_cnt = 0
        block_vars.n_apriltag_max = N_APRILTAGS_MAX
        block_vars.event_ind = -1
        print(f"\nValid block {block_vars.block_id + 1}/{len(self.valid_block_inds)} starting.")

        # First (frame_0) and last (frame_1) video frames of block
        # block_vars.time = session_data.diode_df_blocks[block_vars.block_id].time.to_numpy('float', copy=True)
        block_vars.frame_0 = video_frame_cnt
        block_duration = session_data.diode_df_blocks[block_vars.block_id].time.iloc[-1]
        block_end_time = session_data.video_time[block_vars.frame_0] + block_duration
        # block_end_time = session_data.video_time[block_vars.frame_0] + block_vars.time[-1]
        block_vars.frame_1 = np.where(session_data.video_time > block_end_time)[0][0] - 1
        session_data.block_frames.append((block_vars.frame_0, block_vars.frame_1))
        print(f"First and last block frames are {block_vars.frame_0} and {block_vars.frame_1}")

        # Block time series
        block_vars.time = session_data.video_time[block_vars.frame_0:block_vars.frame_1].squeeze()
        block_vars.time -= block_vars.time[0]
        session_data.block_times.append(block_vars.time)
        n_inds_block = block_vars.frame_1 - block_vars.frame_0
        session_data.apriltag123_visible.append(np.zeros(n_inds_block))
        landmark_pos_dict = {lm: np.zeros((2, n_inds_block)) for lm in session_data.tracked_landmark_ids}
        session_data.hand_landmark_pos_abs.append(landmark_pos_dict)
        session_data.reference_pos_abs.append({ref: np.zeros((2, n_inds_block)) for ref in APPARATUS_TAG_IDS})
        block_vars.event_onsets = session_data.event_onsets_blocks[block_vars.block_id]

        # OpenCV video writer, used to create block videos
        video_path = f"data/pipeline_data/{session_data.participant_id}/{session_data.session_id}/block_videos"
        Path(video_path).mkdir(parents=True, exist_ok=True)
        video_name = f"block_{block_vars.block_id}.mp4"
        block_vars.block_video = cv2.VideoWriter(
            os.path.join(video_path, video_name),
            self.fourcc_codec,
            self.fps,
            (self.pixel_width, self.pixel_height)
        )

    def save_block_data(
            self,
            video_frame_cnt: int,
            block_vars: BlockVariables,
            session_data: SessionData,
    ) -> None:
        """Releases the current block video and saves the data.

        Parameters
            video_frame_cnt (int): the number of the current video frame
            block_vars (BlockVariables): all variables relevant to each block
            session_data (SessionData): all variables tracked for the session
        """

        print(f"Valid block {block_vars.block_id + 1}/{len(self.valid_block_inds)} finishing.")
        if self.block_cnt in self.valid_block_inds:  # Wrap up previous block
            block_vars.block_video.release()
            session_data.saved_block_inds.append(self.block_cnt)
            session_data.last_frame_saved = video_frame_cnt
            with open(self.data_path, "wb") as f:
                pickle.dump(session_data, f)

    def get_hand_position(
            self,
            block_frame_cnt: int,
            tracking_results: NamedTuple,
            session_data: SessionData,
    ) -> None:
        """Gets the positions of all tracked hand landmarks.

        Parameters
            block_frame_cnt (int): the number of the current video frame of the block
            tracking_results (NamedTuple): AprilTag tracking objects
            session_data (SessionData): all variables tracked for the session
        """
        # Get the positions of all relevant hand landmarks
        if tracking_results.multi_hand_landmarks is not None:
            # Track the hand that is highest on the screen (lowest y-value)
            lowest_y = self.pixel_height
            for ind, hand_lm in enumerate(tracking_results.multi_hand_landmarks):
                y_index_tip = hand_lm.landmark[INDEX_FINGER_TIP_IDX].y * self.pixel_height
                if y_index_tip < lowest_y:
                    self.hand_ind = ind
                lowest_y = y_index_tip

            # Save the hand landmark position data
            try:
                pointing_hand_pos = tracking_results.multi_hand_landmarks[self.hand_ind].landmark
            except IndexError:
                print("Doesn't work")
            for landmark_id in session_data.tracked_landmark_ids:
                landmark_pos = pointing_hand_pos[landmark_id]
                cx, cy = landmark_pos.x * self.pixel_width, landmark_pos.y * self.pixel_height
                session_data.hand_landmark_pos_abs[-1][landmark_id][:, block_frame_cnt] = cx, cy

    def markup_video_frame(
            self,
            frame_rgb: np.ndarray,
            tracking_results: NamedTuple,
            block_vars: BlockVariables,
            apriltags: list[Detection],
    ) -> None:
        """Adds markers to a video frame for visually checking synchronization with diode data.

        Parameters
            frame_rgb (np.ndarray): the current video frame, converted to the RGB colour scheme
            tracking_results (NamedTuple): AprilTag tracking objects
            block_vars (BlockVariables): all variables relevant to each block
            apriltags (list[Detection]): a list of AprilTags detected in the current video frame
"""

        # Draw hand landmarks in frame
        if tracking_results is not None:
            self.draw_hand_connections(frame_rgb, tracking_results)

        # Add AprilTag timing markers to video, based on diode data
        if block_vars.apriltag_detected:
            apriltag_data = apriltags[0]
            p1_rectangle = tuple([int(coord) for coord in apriltag_data.corners[0]])
            p2_rectangle = tuple([int(coord) for coord in apriltag_data.corners[2]])
            frame_rgb = cv2.rectangle(frame_rgb, p1_rectangle, p2_rectangle, (0, 0, 255), -1)

        # Add event onset markers to video, based on diode data
        # TODO: maybe replace circles with markers
        # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga0ad87faebef1039ec957737ecc633b7b
        if block_vars.event_ind < block_vars.event_onsets.size:
            if block_vars.time[block_vars.frame_cnt] > block_vars.event_onsets[block_vars.event_ind]:
                try:
                    finger_tip_pos = tracking_results.multi_hand_landmarks[self.hand_ind].landmark[INDEX_FINGER_TIP_IDX]
                    cx = finger_tip_pos.x * self.pixel_width
                    cy = finger_tip_pos.y * self.pixel_height
                except TypeError:
                    cx, cy = 640, 360
                r_circle = 15
                cv2.circle(frame_rgb, (int(cx), int(cy) - 50), r_circle, (255, 255, 255), -1)
                block_vars.event_ind += 1

    def draw_hand_connections(self, frame: np.ndarray, tracking_results: NamedTuple) -> None:
        """Draws the hand landmarks and connections in the current frame.

        Parameters
            frame (np.ndarray): current video frame
            tracking_results (NamedTuple): output object of mediapipe hand tracking
        """

        if tracking_results.multi_hand_landmarks is not None:
            hand = tracking_results.multi_hand_landmarks[self.hand_ind]
            for landmark_id, landmark_pos in enumerate(hand.landmark):
                # Get the image dimensions
                height, width, colour = frame.shape

                # Finding the coordinates of each landmark
                cx, cy = int(landmark_pos.x * width), int(landmark_pos.y * height)

                # Creating a circle around each landmark
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

            # Drawing the landmark connections
            # https://stackoverflow.com/questions/69240807/how-to-change-colors-of-the-tracking-points-and-connector-lines-on-the-output-vi
            landmark_colour = mp.drawing_utils.DrawingSpec((255, 0, 0))
            mp.drawing_utils.draw_landmarks(frame, hand, HAND_LANDMARK_CONNECTIONS, landmark_colour)

    @staticmethod
    def missed_apriltag_detection(frame_grey: np.ndarray) -> None:
        """Attempts to detect missed AprilTags by increasing the Detector's sharpening parameter.

        Parameters
            frame_grey (np.ndarray): video frame in which to detect AprilTags
        """

        sharp_detector = Detector(quad_sigma=0.8, decode_sharpening=0.75)
        tags2 = sharp_detector.detect(frame_grey)
        for tag2 in tags2:
            if tag2.tag_id in PHONE_TAG_IDS:
                print("Missed AprilTag detected with increased sharpening")
            else:
                print("Missed AprilTag still undetected")

    @staticmethod
    def plot_last_block_data(
            block_vars: BlockVariables,
            session_data: SessionData
    ) -> None:
        """Plots the data from the previous experiment block, for visual inspection.

        Parameters
            block_vars (BlockVariables): all variables relevant to each block
            session_data (SessionData): all variables tracked for the session
        """

        # Generate an event onset time series
        event_times = block_vars.event_onsets
        block_time = session_data.block_times[-1]
        event_onset_binary = np.zeros(block_time.size)
        for t in event_times:
            event_onset_binary[np.where(block_time > t)[0][0]] = 80

        # Plot the AprilTag and event onset comparison
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        diode_block_data = session_data.diode_df_blocks[block_vars.block_id]
        ax.plot(block_time, event_onset_binary, "C1", label='Event Video')
        ax.plot(block_time, session_data.apriltag123_visible[-1], "C2", label='AprilTag Video')
        ax.plot(diode_block_data.time, diode_block_data.light_value, "C0", label='Diode Value')
        ax.legend()
        plt.show()
