import cv2
import time
from typing import NamedTuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mediapipe.python.solutions as mp
from pupil_apriltags import Detector, Detection

from utils import get_fourcc, get_top_left_coords
from utils_preprocessing import (
    preprocess_gaze_data,
    preprocess_diode_data,
    separate_diode_blocks,
    get_event_times,
)


# Experiment constants
N_BLOCKS_MAX = 9
N_APRILTAGS_MAX = 4

# AprilTag detection constants
PHONE_TAG_IDS = [1, 2, 3]
APPARATUS_TAG_IDS = [10, 20, 30, 40]
N_FRAME_BUFFER = 12  # 30 fps -> 12 frames ~ 0.4s

# Hand tracking constants
INDEX_FINGER_TIP_IDX = 8
HAND_LANDMARK_CONNECTIONS = mp.hands.HAND_CONNECTIONS

# Data processing constants
REFERENCE_TAG_ID = 40
HIGH_THRESHOLD = 200
APRILTAG_THRESHOLD = 50
EVENT_THRESHOLD = 80


@dataclass()
class PipelineConfig:
    participant_id: int
    session_id: str
    tracked_hand_landmarks: dict[str, int]
    save_video: bool
    show_video: bool
    visual_plot_check: bool
    apriltag_kwargs: dict = field(default_factory=dict)
    mediapipe_kwargs: dict = field(default_factory=dict)
    max_hands: int = 2


class SessionData:
    """This class stores all the data processed by the pipeline.

    Attributes
        self.participant_id: int = participant_id
        self.session_id: int = session_id
        self.n_frames: int = video_time.size
        self.tracked_landmark_ids: list[int] = list(tracked_landmarks.values())
        self.tracked_landmark_names: list[str] = list(tracked_landmarks.keys())
        self.video_time: np.ndarray = video_time
        self.block_inds: list[tuple[int, int]] = []
        self.block_times: list[np.ndarray] = []
        self.apriltag123_visible: list[np.ndarray] = []
        self.last_apriltag_frame: list[int] = []
        self.hand_landmark_pos: list[dict[int, np.ndarray]] = []
        self.reference_pos: list[dict[int, np.ndarray]] = []
        self.diode_df_blocks: list[pd.DataFrame] = diode_df_blocks
        self.event_onsets_blocks: list[np.ndarray] = event_onsets
    """
    def __init__(
            self,
            participant_id: int,
            session_id: str,
            video_time: np.ndarray,
            tracked_landmarks: dict[str, int],
            diode_df_blocks: list[pd.DataFrame],
            event_onsets: list[np.ndarray],
            block_durations: list[float],
    ):

        # Session parameters
        self.participant_id: int = participant_id
        self.session_id: str = session_id
        self.n_frames: int = video_time.size
        self.tracked_landmark_ids: list[int] = list(tracked_landmarks.values())
        self.tracked_landmark_names: list[str] = list(tracked_landmarks.keys())

        # Full video data
        self.video_time: np.ndarray = video_time

        # Block specific data
        self.block_inds: list[tuple[int, int]] = []
        self.block_times: list[np.ndarray] = []
        self.block_durations: list[float] = block_durations
        self.apriltag123_visible: list[np.ndarray] = []
        self.last_apriltag_frame: list[int] = []
        self.hand_landmark_pos: list[dict[int, np.ndarray]] = []
        self.reference_pos: list[dict[int, np.ndarray]] = []
        self.diode_df_blocks: list[pd.DataFrame] = diode_df_blocks
        self.event_onsets_blocks: list[np.ndarray] = event_onsets

    @staticmethod
    def interpolate_pos(time_data: np.ndarray, position_data: np.ndarray):
        """Interpolates the hand landmark and reference position data.

        Parameters
            time_data (np.ndarray): the timestamp for each block frame
            position_data (np.ndarray): the x-y position for each block frame

        Returns
            x_interp (np.ndarray): the interpolated x values of the data
            y_interp (np.ndarray): the interpolated y values of the data
        """

        min_value = 0.1
        interp_mask = (position_data[0, :] > min_value) & (position_data[1, :] > min_value)
        x_interp = np.interp(time_data, time_data[interp_mask], position_data[0, :][interp_mask])
        y_interp = np.interp(time_data, time_data[interp_mask], position_data[1, :][interp_mask])
        return x_interp, y_interp

    def relative_hand_positions(self, show_plot: bool):
        """Calculates hand positions relative to a reference point.

        Since the head-mounted camera is constantly moving, the hand positions are calculated relative
        to a reference point. In this case, the top-left corner of the top-left AprilTag that is
        affixed to the apparatus frame is chosen. It was chosen because it is considered least likely
        to be covered during the trials (assuming the participants use their right index finger).

        Since the participant's hands are often off camera while the AprilTags are shown on the phone,
        this method trims the hand landmark and reference position data to begin when the last AprilTag
        disappears. Furthermore, the data is checked for frame where the hands or reference position is
        not tracked, and interpolates the positions for these frames. This assumption is reasonable when
        only a few consecutive frames are lost, due to the smoothness/continuity of video position in
        time.
        """

        itr_lists = zip(self.hand_landmark_pos, self.reference_pos, self.block_times, self.last_apriltag_frame)
        for lm_pos, ref_pos, time_vec, ind0 in itr_lists:
            # Interpolate the missing reference AprilTag positions
            ref_x, ref_y = self.interpolate_pos(time_vec, ref_pos[REFERENCE_TAG_ID])
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for lm, lbl in zip(self.tracked_landmark_ids, self.tracked_landmark_names):
                # Interpolate and calculate hand landmark positions, relative to the reference
                lm_x, lm_y = self.interpolate_pos(time_vec, lm_pos[lm])
                lm_pos[lm] = np.stack((lm_x[ind0:] - ref_x[ind0:], lm_y[ind0:] - ref_y[ind0:]))
                ax.plot(time_vec[ind0:], lm_pos[lm][0, :], label=f"X: {lbl}")
                ax.plot(time_vec[ind0:], lm_pos[lm][1, :], label=f"Y: {lbl}")
            ax.legend()
            if show_plot:
                plt.show()


@dataclass
class BlockVariables:
    block_id: int = -1
    frame_0: int = -1
    frame_1: int = -1
    frame_cnt: int = -1
    time: np.ndarray = None
    block_video: cv2.VideoWriter = None

    apriltag_n: int = -1
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
        participant_id (int): unique participant identifier
        session_id (int): session identifier
        save_video (bool): whether to save the video data as individual blocks
        show_video (bool): whether to stream the video during processing
        show_plots (bool): whether to display plots as visual checks on the process
        video_capture: (cv2.VideoCapture): mp4 video to be processed
        pixel_width (int): video width, in pixels
        pixel_height (int): video height, in pixels
        fps (float): video frames rate (frames per second)
        fourcc_codec (str): four character code (fourcc) of the mp4 video file
        n_frames (int): number of frames in the video
        diode_df_blocks (list[pd.DataFrame]): light diode sensor data, collected from the phone screen
        event_onsets (list[np.ndarray]): event onset times for each block
        block_durations (list[float]): duration of each block
        video_time (np.ndarray): video frame timestamps
        apriltag_detector (Detector): AprilTag detection object
        tracked_landmarks (dict[str, int]): hand landmark names (keys) and their mediapipe IDs (values)
        hands_model (mp.hands.Hands): mediopipe hand tracking object
    """

    def __init__(self, config: PipelineConfig):
        """"
        Parameters
            config (PipelineConfig)
                participant_id (int): unique participant identifier
                session_id (int): session identifier
                tracked_hand_landmarks (list[int]): all mediapipe hand landmark indices to be tracked
                save_video (bool): whether to save the video data as individual blocks
                show_video (bool): whether to stream the video during processing
                visual_plot_check (bool): whether to display plots as visual checks on the process
                apriltag_kwargs (dict): key word arguments for the AprilTag detector class
                mediapipe_kwargs (dict): key word arguments for hand tracker class
                max_hands (int): the maximum number of hands for the hand tracker to track
        """

        # Processing parameters
        self.participant_id: int = config.participant_id
        self.session_id: str = config.session_id
        self.save_video: bool = config.save_video
        self.show_video: bool = config.show_video
        self.show_plots: bool = config.visual_plot_check

        # Initialize a video capture object and store its parameters
        video_path = f"data/participants/P{config.participant_id}/{config.session_id}/" \
                     f"P{config.participant_id}_{config.session_id}.mp4"
        video_capture = cv2.VideoCapture(video_path)
        self.video_capture: cv2.VideoCapture = video_capture
        self.pixel_width: int = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.pixel_height: int = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps: float = video_capture.get(cv2.CAP_PROP_FPS)
        self.fourcc_codec: int = cv2.VideoWriter_fourcc(*get_fourcc(video_capture))
        self.n_frames: int = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Split the diode data into blocks
        diode_df = preprocess_diode_data(config.participant_id, config.session_id)
        self.diode_df_blocks: list[pd.DataFrame] = separate_diode_blocks(diode_df, HIGH_THRESHOLD, APRILTAG_THRESHOLD)
        event_onset_times, block_duration_times = get_event_times(self.diode_df_blocks, EVENT_THRESHOLD)
        self.event_onsets: list[np.ndarray] = event_onset_times
        self.block_durations: list[float] = block_duration_times

        # Preprocess the gaze data
        video_time = preprocess_gaze_data(config.participant_id, config.session_id, config.visual_plot_check)
        self.video_time: np.ndarray = video_time

        # AprilTag detection variables
        # self.apriltag_detector: Detector = Detector()
        self.apriltag_detector: Detector = Detector(**config.apriltag_kwargs)

        # Hand tracking variables
        self.tracked_landmarks: dict[str, int] = config.tracked_hand_landmarks
        self.hands_model: mp.hands.Hands = mp.hands.Hands(max_num_hands=config.max_hands, **config.mediapipe_kwargs)

    def iterate_video_frames(self) -> SessionData:
        """Stores hand positions for all video frames, and writes block videos with markers.

        Returns
            session_data (SessionData): class object containing all relevant experiment session data
        """

        # Initialize variables for analyzing video
        video_frame_cnt = -1

        # Initialize the classes used to store experimental data
        block_vars = BlockVariables()
        session_data = SessionData(
            self.participant_id,
            self.session_id,
            self.video_time,
            self.tracked_landmarks,
            self.diode_df_blocks,
            self.event_onsets,
            self.block_durations,
        )

        run_time_t0 = time.time()
        while self.video_capture.isOpened():
            # Read the next frame
            ret, frame = self.video_capture.read()
            video_frame_cnt += 1
            block_vars.frame_cnt += 1
            if video_frame_cnt == self.n_frames:
                print("End of video file.")
                break
            elif not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # if video_frame_cnt % 6000 == 0 and video_frame_cnt > 0:
            #     print(f"Current frame: {video_frame_cnt}/{self.n_frames}, runtime: {time.time() - run_time_t0:0.2f}")
            #     break
            # if video_frame_cnt >= 3:
            #     break

            # Convert frame from BGR to grayscale and RGB
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect AprilTags
            tags = self.detect_apriltags_phone(
                frame_grayscale,
                video_frame_cnt,
                block_vars,
                session_data,
            )

            # If in a block
            if block_vars.frame_cnt < (block_vars.frame_1 - block_vars.frame_0):
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
                        session_data.reference_pos[-1][tag.tag_id][:, block_vars.frame_cnt] = xy_corner

                if self.save_video or self.show_video:
                    # Draw hand landmarks and connections in frame
                    self.markup_video_frame(
                        frame_rgb,
                        tracking_results,
                        block_vars,
                        tags,
                    )
                    frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if self.save_video:
                    # Write blocks to new video
                    # frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    if block_vars.block_id >= 0:
                        block_vars.block_video.write(frame_out)
                        # pass

            if self.show_video:
                # Stream the frame (as video)
                # frame_out = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', frame_out)
                if cv2.waitKey(1) == ord('q'):
                    break

        print(f"It took {time.time() - run_time_t0:0.2f} seconds to iterate through the video")
        print(f"Reported frames: {self.n_frames}")
        print(f"Counted frames: {video_frame_cnt}")
        if self.n_frames == video_frame_cnt:
            print('No corrupted frames! Hooray!')
        self.video_capture.release()
        block_vars.block_video.release()
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

        If the end of the 5th AprilTag is detected at approximately the correct time,
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
                    elif block_vars.apriltag_n == N_APRILTAGS_MAX or block_vars.block_id < 0:  # New block of trials
                        if block_vars.block_id >= 0:
                            block_vars.block_video.release()
                            if self.show_plots:
                                self.plot_last_block_data(block_vars, session_data)
                        self.reset_block_data(
                            video_frame_cnt,
                            block_vars,
                            session_data
                        )
                    else:  # Increment AprilTag (never more than 4)
                        block_vars.apriltag_n += 1
                        assert block_vars.apriltag_n <= N_APRILTAGS_MAX
                block_vars.apriltag_detected = True
                block_vars.apriltag_last_frame = video_frame_cnt
                session_data.apriltag123_visible[-1][block_vars.frame_cnt] = 100
            else:
                # No AprilTag detected on phone
                if block_vars.apriltag_n == N_APRILTAGS_MAX and block_vars.apriltag_detected:
                    # If it is the 5th AprilTag, check that the timing is approximately correct
                    tag_time = session_data.video_time[block_vars.frame_0:video_frame_cnt]
                    print(tag_time[-1] - tag_time[0])
                    assert 8.9 < (tag_time[-1] - tag_time[0]) < 9.3
                    block_vars.event_ind = 0
                    session_data.last_apriltag_frame.append(block_vars.frame_cnt)
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
        assert block_vars.block_id <= N_BLOCKS_MAX
        block_vars.apriltag_n = 0
        block_vars.event_ind = -1

        # First (frame_0) and last (frame_1) video frames of block
        block_vars.frame_0 = video_frame_cnt
        block_duration = session_data.block_durations[block_vars.block_id]
        block_end_time = session_data.video_time[block_vars.frame_0] + block_duration
        block_vars.frame_1 = np.where(session_data.video_time > block_end_time)[0][0] - 1
        session_data.block_inds.append((block_vars.frame_0, block_vars.frame_1))
        print(f"First and last block frames are {block_vars.frame_0} and {block_vars.frame_1}")

        # Block time series
        # TODO: make sure the blocks are trimmed properly
        block_vars.time = session_data.video_time[block_vars.frame_0:block_vars.frame_1]
        block_vars.time -= block_vars.time[0]
        session_data.block_times.append(block_vars.time)
        block_len = block_vars.frame_1 - block_vars.frame_0
        session_data.apriltag123_visible.append(np.zeros(block_len))
        session_data.hand_landmark_pos.append({lm: np.zeros((2, block_len)) for lm in session_data.tracked_landmark_ids})
        session_data.reference_pos.append({ref: np.zeros((2, block_len)) for ref in APPARATUS_TAG_IDS})
        block_vars.event_onsets = session_data.event_onsets_blocks[block_vars.block_id]

        # OpenCV video writer, used to create block videos
        block_vars.block_video = cv2.VideoWriter(
            f'data/participants/P{session_data.participant_id}/{session_data.session_id}/block_videos/'
            f'block{block_vars.block_id}.mp4',
            self.fourcc_codec,
            self.fps,
            (self.pixel_width, self.pixel_height)
        )

    def get_hand_position(
            self,
            block_frame_cnt: int,
            tracking_results: NamedTuple,
            session_data: SessionData,
    ) -> None:
        """Gets the positions of all relevant hand landmarks.

        Parameters
            block_frame_cnt (int): the number of the current video frame of the block
            tracking_results (NamedTuple): AprilTag tracking objects
            session_data (SessionData): all variables tracked for the session
        """
        # Get the positions of all relevant hand landmarks
        if tracking_results.multi_hand_landmarks is not None:
            pointing_hand_pos = tracking_results.multi_hand_landmarks[0].landmark
            for landmark_id in session_data.tracked_landmark_ids:
                landmark_pos = pointing_hand_pos[landmark_id]
                # cx, cy = int(landmark_pos.x * self.width), int(landmark_pos.y * self.height)
                cx, cy = landmark_pos.x * self.pixel_width, landmark_pos.y * self.pixel_height
                session_data.hand_landmark_pos[-1][landmark_id][:, block_frame_cnt] = cx, cy
        # else:
        #     for landmark_id in session_data.tracked_landmark_ids:
        #         session_data.hand_landmark_pos[-1][landmark_id][:, frame_id] = None, None

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
                finger_tip_pos = tracking_results.multi_hand_landmarks[0].landmark[INDEX_FINGER_TIP_IDX]
                cx = finger_tip_pos.x * self.pixel_width
                cy = finger_tip_pos.y * self.pixel_height
                r_circle = 15
                cv2.circle(frame_rgb, (int(cx), int(cy) - 50), r_circle, (255, 255, 255), -1)
                block_vars.event_ind += 1

    @staticmethod
    def missed_apriltag_detection(frame_grey: np.ndarray) -> None:
        """Attempts to detect missed AprilTags by increasing the Detector's sharpening parameter.

        Parameters
            frame_grey (np.ndarray): video frame in which to detect AprilTags
        """

        sharp_detector = Detector(decode_sharpening=0.75)
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
        ax.plot(diode_block_data.time, diode_block_data.light_value, label='Diode Value')
        ax.plot(block_time, event_onset_binary, label='Event Video')
        ax.plot(block_time, session_data.apriltag123_visible[-1], label='AprilTag Video')
        ax.legend()
        plt.show()

    @staticmethod
    def draw_hand_connections(frame: np.ndarray, tracking_results: NamedTuple) -> None:
        """Draws the hand landmarks and connections in the current frame.

        Parameters
            frame (np.ndarray): current video frame
            tracking_results (NamedTuple): output object of mediapipe hand tracking
        """

        if tracking_results.multi_hand_landmarks is not None:
            for hand in tracking_results.multi_hand_landmarks:
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
