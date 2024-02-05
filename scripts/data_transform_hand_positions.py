"""This script must be run after the pipeline. The hand tracking positions that come out of the
pipeline are in the pixel frame of reference. Therefore, the x-direction is the horizontal direction
with regards to the video frame, and similarly for the y-direction. Since the participant's head is
constantly moving, and the position between participants and sessions varies greatly, the hand
tracking data must be tranformed into a standard frame of reference, in order to be comparable
between sessions, or even between frames.

This script uses three AprilTags to define a frame of reference that is consistent across all trials.
The standard unit of this frame of reference is the distance between the top-left corners of the two
AprilTags at the top of the experimental apparatus (IDs 40 and 10). Since this distance is not the
same in all video frames, all the position values are scaled to make them consistent and comparable.

There are many video frames where the AprilTags are not on the screen, covered, or undetected for
some other reason. This will be accounted for in the next step. For now, these missing AprilTag
reference positions are simply interpolated using the frame before and after the missed detections."""

import os
import sys
import pickle
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline

sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(0, str(Path(os.path.dirname(__file__)).parent.absolute()))
from utils.calculations import MIN_POSITION, get_basis_vectors
from utils.data_loading import load_pipeline_config, get_files_containing
from utils.pipeline import INDEX_FINGER_TIP_IDX, SessionData, load_session_data


DT_SPEED = 0.001


# TODO: change this to calculate_hand_speeds.py, and add hand speed calculations
class TransformedHandData:
    """This class stores the transformed hand landmark data.

    The hand data is transformed from the video frame of reference to the frame of reference defined
    by the top-left corner of the AprilTags on the experimental apparatus, with IDs: 10 (y-dir basis),
    20 or 30 (x-dir basis), and 40 (origin). This transformation includes a scaling factor, such that
    data from all sessions, blocks, and time points are comparable.

    Attributes
        participant_id (str): unique participant identifier
        session_id (str): session identifier
        hand_landmarks (dict[int, str]): names (values) of landmarks associated with each ID (keys)
        time (list[np.ndarray]): time data for each block
        hand_position (list[dict[str, np.ndarray]]): transformed hand landmark position data
        ref_pos (list[dict[str, np.ndarray]]): transformed AprilTag reference position data

    """

    def __init__(self, session_data: SessionData):
        self.participant_id: str = session_data.participant_id
        self.session_id: str = session_data.session_id
        self.hand_landmarks: dict[str, int] = session_data.tracked_landmarks
        self.time_video: list[np.ndarray] = []
        self.time_interpolated: list[np.ndarray] = []
        self.hand_pos: list[dict[int, np.ndarray]] = []
        self.hand_pos_interpolated: list[dict[int, np.ndarray]] = []
        self.hand_speed: list[dict[int, np.ndarray]] = []
        self.ref_pos: list[dict[int, np.ndarray]] = []

    def transform_hand_positions(
            self,
            session_data_class: SessionData,
            reference_scale_matrix: np.ndarray,
            plot_landmarks: list[int] = None,
    ) -> None:
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
        time. Trials with an unacceptable number of consecutive missed AprilTag detections will be
        filtered at a later date.

        The same AprilTags cannot always be used as references. Therefore, the AprilTag IDs used to
        define the coordinates of a reference frame are provided in the order [origin, v1, v2].

        Parameters
            session_data_class (SessionData): class containing the pipeline session data
            reference_scale_matrix (np.ndarray): matrix used to scale the transformed positions
            plot_landmarks (dict[str, int]): hand landmarks to plot
        """

        itr_lists = zip(
            session_data_class.hand_landmark_pos_abs,
            session_data_class.reference_pos_abs,
            session_data_class.block_times,
            session_data_class.first_trial_frame.values(),
        )
        for block, (lm_pos, ref_pos, time_vec, ind0) in enumerate(itr_lists):
            # Determine which AprilTag is the best reference
            reference_tag_id = session_data_class.apparatus_tag_ids[0]

            # Interpolate the missing reference AprilTag positions
            interp_ref_pos = {}
            for tag_id in session_data_class.apparatus_tag_ids:
                interp_ref_pos[tag_id] = interpolate_pos(time_vec, ref_pos[tag_id])[:, ind0:]

            # Rotate the frame to be square to the apparatus' AprilTags
            position_shape = interp_ref_pos[reference_tag_id].shape
            transformed_lm_pos = {lm: np.zeros(position_shape) for lm in session_data_class.tracked_landmarks.values()}
            rotated_ref_pos = {tag: np.zeros(position_shape) for tag in session_data_class.apparatus_tag_ids}
            for i in range(position_shape[1]):
                # Calculate the transformation matrix for each frame
                transformation_matrix = get_transformation_matrix(
                    interp_ref_pos,
                    i,
                    session_data_class.apparatus_tag_ids,
                    reference_scale_matrix,
                )

                # Transform hand landmarks
                for lbl, lm in session_data_class.tracked_landmarks.items():
                    # Interpolate the hand landmark positions
                    lm_xy = interpolate_pos(time_vec, lm_pos[lm])[:, ind0:]
                    # Translate the coordinates to the origin of the reference frame
                    rel_xy = lm_xy - interp_ref_pos[reference_tag_id]
                    # Transform the coordinates into the reference frame
                    transformed_lm_pos[lm][:, i] = transformation_matrix @ rel_xy[:, i]

                # Transform reference AprilTags
                for tag_id in session_data_class.apparatus_tag_ids:
                    rotated_ref_pos[tag_id][:, i] = transformation_matrix @ interp_ref_pos[tag_id][:, i]

            if plot_landmarks is not None:
                # TODO: update plot_landmarks such that it pulls the landmark names from session_data_class
                landmark_names_ids = {name: lm for name, lm in self.hand_landmarks.items() if lm in plot_landmarks}
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                for landmark_name, landmark_id in landmark_names_ids.items():
                    ax.plot(time_vec[ind0:], transformed_lm_pos[landmark_id][0, :], label=f"X: {landmark_name}")
                    ax.plot(time_vec[ind0:], transformed_lm_pos[landmark_id][1, :], label=f"Y: {landmark_name}")
                ax.set_title(f"{self.participant_id}-{self.session_id}-Block: {block} (Ref Tag: #{reference_tag_id})")
                ax.legend()
                plt.show()
                plt.close()
            else:
                print(f"{self.participant_id}-{self.session_id}-Block: {block} (Ref Tag: #{reference_tag_id})")

            self.time_video.append(time_vec[ind0:])
            self.hand_pos.append(transformed_lm_pos)
            self.ref_pos.append(rotated_ref_pos)

    def calculate_hand_speeds(self, session_data: SessionData) -> None:
        # Calculate the index fingertip speed
        for block_time, hand_pos, onset_times in zip(self.time_video, self.hand_pos, session_data.event_onsets_blocks):
            # Interpolate the time to be used in the spline (video resolution is too low for differentiation)
            time_interpolated = np.arange(block_time[0], block_time[-1], DT_SPEED)

            # Determine the smoothing factor
            x_tip, y_tip = hand_pos[INDEX_FINGER_TIP_IDX][0, :], hand_pos[INDEX_FINGER_TIP_IDX][1, :]
            for s in np.arange(3000, 6001, 300):
                x_tip_spline = cubic_spline_filter(block_time, x_tip, time_interpolated, s)
                y_tip_spline = cubic_spline_filter(block_time, y_tip, time_interpolated, s)

            for landmark_names, landmark_ids in self.hand_landmarks.items():
                # x_pos = hand_pos[lm_ind][0, :]
                # filtered_spline = cubic_spline_filter(block_time, x_pos, filtered_time, postprocess_config.smoothing_factor)
                # nan_mask = np.isnan(filtered_spline)
                # filtered_time = filtered_time[~nan_mask]
                # filtered_spline = filtered_spline[~nan_mask]
                # filtered_vel = np.diff(filtered_spline) / np.diff(filtered_time)
                fig, ax = plt.subplots(2, 1, figsize=(10, 10))
                ax[0].scatter(block_time, x_pos, label="X")
                # ax[0].plot(filtered_time, filtered_spline, label=f"Cubic Spline")
                # ax[0].vlines(onset_times, min(x_pos), max(x_pos), 'r')
                # # ax[0].set_ylim([min(x_pos), max(x_pos)])
                # ax[0].set_xlabel("Time")
                # ax[0].set_ylabel("Transformed Index Finger Tip Position")
                # ax[1].plot(filtered_time[:-1], filtered_vel, label=f"dx/dt")
                # ax[1].vlines(onset_times, min(filtered_vel), max(filtered_vel), 'r')
                # print(f"Min and max speed {min(filtered_vel)} and {max(filtered_vel)}")
                # ax[0].legend()
                # plt.show()
                for s in [1000, 3000, 5000, 7000]:
                    filtered_spline = cubic_spline_filter(block_time, x_pos, filtered_time, s)
                    nan_mask = np.isnan(filtered_spline)
                    filtered_time = filtered_time[~nan_mask]
                    filtered_spline = filtered_spline[~nan_mask]
                    filtered_vel = np.diff(filtered_spline) / np.diff(filtered_time)
                    ax[0].plot(filtered_time, filtered_spline, label=f"Smoothing: {s}")
                    ax[0].vlines(onset_times, min(x_pos), max(x_pos), 'r')
                    ax[1].plot(filtered_time[:-1], filtered_vel, label=f"Smoothing: {s}")
                    ax[1].vlines(onset_times, min(filtered_vel), max(filtered_vel), 'r')
                ax[0].set_ylabel("Index Fingertip Position", fontsize=16)
                ax[0].set_title("Cubic Spline Fitting of Hand Positions", fontsize=20)
                ax[0].set_xlim([65.8, 67.9])
                ax[0].set_ylim([min(x_pos), max(x_pos)])
                ax[0].legend(loc=1)
                ax[1].set_xlabel("Time")
                ax[1].set_ylabel("Index Fingertip Speed", fontsize=16)
                ax[1].set_xlim([65.8, 67.9])
                ax[1].set_ylim([min(filtered_vel), max(filtered_vel)])
                ax[1].legend(loc=1)
                plt.show()


def interpolate_pos(time_data: np.ndarray, position_data: np.ndarray) -> np.ndarray:
    """Interpolates the hand landmark and reference position data.

    Parameters
        time_data (np.ndarray): the timestamp for each block frame
        position_data (np.ndarray): the x-y position for each block frame

    Returns
        np.ndarray: the interpolated x- and y-values of the data
    """

    if np.sum(position_data) == 0:
        print("No position data")
        return position_data
    else:
        interp_mask = (position_data[0, :] > MIN_POSITION) & (position_data[1, :] > MIN_POSITION)
        x_interp = np.interp(time_data, time_data[interp_mask], position_data[0, :][interp_mask])
        y_interp = np.interp(time_data, time_data[interp_mask], position_data[1, :][interp_mask])

        return np.stack((x_interp, y_interp))


def get_transformation_matrix(
        reference_tag_pos: dict[int, np.ndarray],
        index: int,
        reference_tag_ids: list[int],
        reference_scale_matrix: np.ndarray,
) -> np.ndarray:
    """Calculates a transformation matrix, given a set of basis coordinates and scaling matrix.

    The transformation accounts for both the conversion from the pixel frame of reference to the apparatus
    frame of reference, as well as scaling by a global scaling factor that is consistent across all
    participants, sessions, blocks and trials. The global scaling factor allows an equivalent comparison
    of speed, even though the absolute (pixel) dimensions change, due to differences in the placement of
    the head mounted camera, relative to the experimental apparatus. The reference distance for scaling is
    the distance between the top-left corners of the top two AprilTags (IDs 40 and 10).

    Parameters
        reference_coordinates (dict[int, np.ndarray]): AprilTag reference coordinates (top-left corner)
        index (int): video frame index to calculate for
        reference_tag_ids (list[int]): AprilTag IDs used as reference points (origin, y-dir, x-dir)
        reference_scale_matrix (np.ndarray): reference scaling matrix

    Returns
        (np.ndarray): matrix that performs a basis transformation and scaling when multiplied on the left
    """

    basis_v1, basis_v2 = get_basis_vectors(reference_tag_pos, index, reference_tag_ids)
    rotation_matrix = np.stack((basis_v1, basis_v2)).T

    return np.linalg.inv(rotation_matrix @ reference_scale_matrix)


def get_scaling_matrix(
            reference_coordinates: dict[int, np.ndarray],
            index: int,
            reference_tag_ids: list[int],
) -> np.ndarray:
    """Calculates a scaling matrix, given a set of basis vectors.

    Parameters
        reference_coordinates (dict[int, np.ndarray]): top-left corner coordinates of the three reference AprilTags
        index (int): video frame index to calculate for
        reference_tag_ids (list[int]): AprilTag IDs used as reference points (origin, y-dir, x-dir)

    Returns
        (np.ndarray): scaling matrix for a given set of basis vectors
    """

    basis_v1, basis_v2 = get_basis_vectors(reference_coordinates, index, reference_tag_ids)

    return np.diag([1 / np.linalg.norm(basis_v1), 1 / np.linalg.norm(basis_v2)])


def load_transformed_hand(participant_id: str, session_id: str) -> TransformedHandData | None:
    """Loads the transformed hand data for a particular session.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        (TransformedHandData): class containing the transformed hand data
    """

    hand_pos_path = f"../data/pipeline_data/{participant_id}/{session_id}/"\
                    f"{participant_id}_{session_id}_transformed_hand_data.pkl"
    if os.path.exists(hand_pos_path):
        with open(hand_pos_path, "rb") as f:
            return pickle.load(f)
    else:
        return None


def cubic_spline_filter(
        time_existing: np.ndarray,
        position_existing: np.ndarray,
        time_interpolated: np.ndarray,
        smoothing: int,
) -> np.ndarray:
    """Interpolates and fits a cubic spline to the hand landmark position data.

    Reference: https://docs.scipy.org/doc/scipy/tutorial/interpolate/smoothing_splines.html

    Parameters
        time_existing (np.ndarray): time vector for the existing position data
        position_existing (np.ndarray): position data for the hand landmarks
        time_interpolated (np.ndarray): interpolated time vector over which to fit the spline
        smoothing (int): spline smoothing factor
    """
    tck = splrep(time_existing, position_existing, s=smoothing)
    return BSpline(*tck, extrapolate=False)(time_interpolated)


def calculate_speed(time_data: np.ndarray, position_data: np.ndarray) -> np.ndarray:
    """Calculates speed from position data.

    Parameters
        time_data (np.ndarray): time vector associated with the postion data
        position_data (np.ndarray): single dimension positon data

    Returns
        (np.ndarray): vector of speed at each time point
    """
    return np.diff(position_data) / np.diff(time_data)


def main(plot_landmarks: list[int], overwrite_data: bool):
    """Script that transforms the hand postions to a consistent frame of reference.

    The frame of reference used is the AprilTags set on the corners of the experimental apparatus.

    Parameters
        plot_landmarks (dict[str, int]): names (keys) and IDs (values) of the hand landmarks to plot, if any
    """

    # Load a list of all the pipeline data files
    fpaths, files = get_files_containing("../data/pipeline_data", "pipeline_data.pkl")

    # Load the scaling matrix, or create it, if one does not already exist
    scaling_matrix_fpath = "../data/combined_sessions/scaling_matrix.pkl"
    if os.path.exists(scaling_matrix_fpath):
        with open(scaling_matrix_fpath, "rb") as f:
            scale_matrix = pickle.load(f)
    else:
        with open(os.path.join(fpaths[0], files[0]), "rb") as f:
            reference_data: SessionData = pickle.load(f)
        scale_matrix = get_scaling_matrix(reference_data.reference_pos_abs[0], 0, reference_data.apparatus_tag_ids)
        pathlib.Path(scaling_matrix_fpath).mkdir(parents=True)

    for fpath, file in zip(fpaths, files):
        # Skip data that has already been transformed
        participant_id, session_id = fpath.split("/")[-2:]
        fname = f"{participant_id}_{session_id}_transformed_hand_data.pkl"
        if not overwrite_data and os.path.exists(os.path.join(fpath, fname)):
            print(f"{fname} already exists")
            continue

        # Load the session data
        session_data = load_session_data(participant_id, session_id)

        # Calculate the hand positions relative to the apparatus frame
        hand_data = TransformedHandData(session_data)
        hand_data.transform_hand_positions(session_data, scale_matrix, plot_landmarks)

        # Calculate the speed of each landmark from the position data

        # # Save the transformed hand data
        # with open(os.path.join(fpath, fname), "wb") as f:
        #     pickle.dump(hand_data, f)


if __name__ == "__main__":
    # Get the pipeline config files
    pipeline_config = load_pipeline_config()

    # Define an argparser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--overwrite",
        action="store_true",
        help="If True, overwrite the existing data"
    )
    # TODO: add arg for landmarks to plot (must be int)
    #  - use the hand_landmarks_tracked variable in session_data to get the name of the landmark
    parser.add_argument(
        "-lm", "--landmark_ids",
        type=int,
        nargs="*",
        choices=list(pipeline_config.tracked_hand_landmarks.values()),
        default=[],
        help="IDs of hand landmark positions to plot after transformation"
    )
    args = parser.parse_args()

    # main({"Index Tip": 8})
    main(args.landmark_ids, args.overwrite)
