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

The last step in this script is to fit a cubic spline to the transformed hand position data, and
calculate the landmark speeds. The cubic spline is chosen because it is guaranteed to be twice
differentiable. Speeds are calculate for the x and y directions, as well as the combined total.

There are many video frames where the AprilTags are not on the screen, covered, or undetected for
some other reason. This will be accounted for in the next step. For now, these missing AprilTag
reference positions are simply interpolated using the frame before and after the missed detections."""

import os
import sys
import cv2
import pickle
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, BSpline

sys.path.insert(0, os.path.abspath(".."))
from utils.calculations import MIN_POSITION, get_basis_vectors, calculate_time_derivative
from utils.data_loading import load_pipeline_config, get_files_containing, load_block_video_mp4
from utils.pipeline import SessionData, load_session_data, INDEX_FINGER_TIP_ID


DT_SPEED = 0.001
SMOOTHING = 4500
BASIS_RATIO = 1 / 0.94


# TODO: Refactor transform_hand_positions function (put plotting in its own module)
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
        self.hand_position: list[dict[int, np.ndarray]] = []
        self.ref_position: list[dict[int, np.ndarray]] = []
        self.time_interpolated: list[np.ndarray] = []
        self.hand_pos_interpolated: list[dict[int, np.ndarray]] = []
        self.hand_speed: list[dict[int, np.ndarray]] = []

    def transform_hand_positions(
            self,
            session_data: SessionData,
            reference_scale_matrix: np.ndarray,
            plot_landmarks: list[int] = None,
            plot_vectors: bool = False,
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
            session_data (SessionData): class containing the pipeline session data
            reference_scale_matrix (np.ndarray): matrix used to scale the transformed positions
            plot_landmarks (dict[str, int]): hand landmarks to plot
        """

        itr_lists = zip(
            session_data.hand_landmark_pos_abs,
            session_data.reference_pos_abs,
            session_data.block_times,
            session_data.first_trial_frame.values(),
        )
        for block, (lm_pos, ref_pos, time_vec, ind0) in enumerate(itr_lists):
            # Determine which AprilTag is the best reference
            reference_tag_id = session_data.apparatus_tag_ids[0]

            # Plot the frame with the transformed vectors (for visual check)
            if plot_vectors:
                # Define the position of the index fingertip
                tip_pos = session_data.hand_landmark_pos_abs[block][INDEX_FINGER_TIP_ID]
                tip_pos_trans = np.zeros_like(tip_pos)
                tip_pos_rel = tip_pos - ref_pos[reference_tag_id]

                # Initialize the transformed reference positions
                ref_pos_trans = {tag: np.zeros_like(tip_pos) for tag in session_data.apparatus_tag_ids}
                origin_id, v2_id, v1_id = session_data.apparatus_tag_ids

                # Load the block video
                vcap = load_block_video_mp4(session_data.participant_id, session_data.session_id, block)
                width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                i = 0
                while vcap.isOpened():
                    ret, frame = vcap.read()
                    if i >= ind0 and time_vec[i] > 21:
                        print(f"\nFrame {i + 1}, Block time: {time_vec[i]:0.3f}")
                        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                        ax.set_xlim([-1, width])
                        ax.set_ylim([height, -1])
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ax.imshow(frame_rgb)

                        # Calculate the transformation matrix for each frame
                        basis_v1, basis_v2 = get_basis_vectors(ref_pos, i, session_data.apparatus_tag_ids)
                        rotation_matrix = np.stack((basis_v1, basis_v2)).T
                        transformation_matrix = np.linalg.inv(rotation_matrix @ reference_scale_matrix)

                        # Transform hand landmark (finger tip) and references
                        tip_pos_trans[:, i] = transformation_matrix @ tip_pos_rel[:, i]
                        print(f"Tip position (x, y) = ({tip_pos_rel[0, i]:0.3f}, {tip_pos_rel[1, i]:0.3f})")
                        print(f"Tip position (x, y) = ({tip_pos_trans[0, i]:0.3f},"
                              f" {tip_pos_trans[1, i]:0.3f}) (transformed)")
                        for tag_id in session_data.apparatus_tag_ids:
                            ref_pos_trans[tag_id][:, i] = transformation_matrix @ ref_pos[tag_id][:, i]
                        v1_norm = np.linalg.norm(ref_pos[v1_id][:, i] - ref_pos[origin_id][:, i])
                        v2_norm = np.linalg.norm(ref_pos[v2_id][:, i] - ref_pos[origin_id][:, i])
                        v1_norm_trans = np.linalg.norm(ref_pos_trans[v1_id][:, i] - ref_pos_trans[origin_id][:, i])
                        v2_norm_trans = np.linalg.norm(ref_pos_trans[v2_id][:, i] - ref_pos_trans[origin_id][:, i])
                        print(f"Basis v1 norm {v1_norm:0.3f}")
                        print(f"Basis v2 norm {v2_norm:0.3f}")
                        print(f"Basis v1 scaled norm {v1_norm_trans:0.3f}")
                        print(f"Basis v2 scaled norm {v2_norm_trans:0.3f}")
                        print(f"Reference vector length ratio before and after scaling: "
                              f"{v2_norm / v1_norm:0.2f} and {v2_norm_trans /  v1_norm_trans:0.2f} respectively")

                        # Reference frame and true fingertip vector
                        origin_x, origin_y = ref_pos[origin_id][:, i]
                        ax.plot([origin_x, ref_pos[v1_id][0, i]], [origin_y, ref_pos[v1_id][1, i]], "crimson", lw=2.5)
                        ax.plot([origin_x, ref_pos[v2_id][0, i]], [origin_y, ref_pos[v2_id][1, i]], "crimson", lw=2.5,
                                label="Reference Frame")
                        ax.plot([origin_x, tip_pos[0, i]], [origin_y, tip_pos[1, i]], "crimson", ls="--", lw=2.5,
                                label="Fingertip Position (Reference Frame)")

                        # # Transformed frame and fingertip vector, translated to image origin
                        # ax.plot([0, ref_pos_trans[v1_id][0, i] - ref_pos_trans[origin_id][0, i]],
                        #         [0, ref_pos_trans[v1_id][1, i] - ref_pos_trans[origin_id][1, i]],
                        #         "dodgerblue", lw=2.5)
                        # ax.plot([0, ref_pos_trans[v2_id][0, i] - ref_pos_trans[origin_id][0, i]],
                        #         [0, ref_pos_trans[v2_id][1, i] - ref_pos_trans[origin_id][1, i]],
                        #         "dodgerblue", lw=2.5, label="Transformed Frame")
                        # ax.plot([0, tip_pos[0, i] - ref_pos[origin_id][0, i]],
                        #         [0, tip_pos[1, i] - ref_pos[origin_id][1, i]], "r--", lw=2.5)
                        # ax.plot([0, tip_pos_trans[0, i]], [0, tip_pos_trans[1, i]],
                        #         "dodgerblue", ls="--", lw=2.5, label="Fingertip Position (Transformed Frame)")

                        # Transformed frame and fingertip vector, translated to origin of reference frame
                        ax.plot([origin_x, ref_pos_trans[v1_id][0, i] - ref_pos_trans[origin_id][0, i] + origin_x],
                                [origin_y, ref_pos_trans[v1_id][1, i] - ref_pos_trans[origin_id][1, i] + origin_y],
                                "dodgerblue", lw=2.5)
                        ax.plot([origin_x, ref_pos_trans[v2_id][0, i] - ref_pos_trans[origin_id][0, i] + origin_x],
                                [origin_y, ref_pos_trans[v2_id][1, i] - ref_pos_trans[origin_id][1, i] + origin_y],
                                "dodgerblue", lw=2.5, label="Transformed Frame")
                        # ax.plot([0, tip_pos[0, i] - ref_pos[origin_id][0, i]],
                        #         [0, tip_pos[1, i] - ref_pos[origin_id][1, i]], "g--", lw=3)
                        ax.plot([origin_x, tip_pos_trans[0, i] + origin_x],
                                [origin_y, tip_pos_trans[1, i] + origin_y],
                                "dodgerblue", lw=2.5, ls="--", label="Fingertip Position (Transformed Frame)")

                        ax.axis("off")
                        ax.set_title(f"{session_data.participant_id}-{session_data.session_id}-Block {block}, "
                                     f"Video Time: {time_vec[i]:0.2f}s (Video Frame {i + 1})", fontsize=19)
                        ax.legend(loc=0, framealpha=0.9)
                        plt.show()
                        # fig.savefig(
                        #     f"../images/plots/{session_data.participant_id}-{session_data.session_id}-Block{block}-"
                        #     f"Frame{i}_hand_position_transformation.png", dpi=fig.dpi, bbox_inches="tight"
                        # )

                    i += 1

            else:
                # Interpolate the missing reference AprilTag positions
                interp_ref_pos = {}
                for tag_id in session_data.apparatus_tag_ids:
                    interp_ref_pos[tag_id] = interpolate_pos(time_vec, ref_pos[tag_id])[:, ind0:]

                # Interpolate the hand landmark coordinates and translate them to the origin of the reference frame
                transformed_lm_pos = {}
                for lbl, lm in session_data.tracked_landmarks.items():
                    lm_xy = interpolate_pos(time_vec, lm_pos[lm])[:, ind0:]
                    transformed_lm_pos[lm] = lm_xy - interp_ref_pos[reference_tag_id]

                # Rotate the frame to be square to the apparatus' AprilTags
                position_shape = interp_ref_pos[reference_tag_id].shape
                # transformed_lm_pos = {lm: np.zeros(position_shape) for lm in session_data.tracked_landmarks.values()}
                rotated_ref_pos = {tag: np.zeros(position_shape) for tag in session_data.apparatus_tag_ids}

                for i in range(position_shape[1]):
                    # Calculate the transformation matrix for each frame
                    transformation_matrix = get_transformation_matrix(
                        interp_ref_pos,
                        i,
                        session_data.apparatus_tag_ids,
                        reference_scale_matrix,
                    )

                    # Transform hand landmarks
                    for lbl, lm in session_data.tracked_landmarks.items():
                        # Transform the coordinates into the reference frame
                        transformed_lm_pos[lm][:, i] = transformation_matrix @ transformed_lm_pos[lm][:, i]

                    # Transform reference AprilTags
                    for tag_id in session_data.apparatus_tag_ids:
                        rotated_ref_pos[tag_id][:, i] = transformation_matrix @ interp_ref_pos[tag_id][:, i]

                if plot_landmarks is not None:
                    landmark_names_ids = {name: lm for name, lm in self.hand_landmarks.items() if lm in plot_landmarks}
                    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                    for landmark_name, landmark_id in landmark_names_ids.items():
                        ax.plot(time_vec[ind0:], transformed_lm_pos[landmark_id][0, :], label=f"X: {landmark_name}")
                        ax.plot(time_vec[ind0:], transformed_lm_pos[landmark_id][1, :], label=f"Y: {landmark_name}")
                    ax.set_title(f"{self.participant_id}-{self.session_id}, Block {block}")
                    ax.legend()
                    plt.show()
                    plt.close()
                print(f"{self.participant_id}-{self.session_id}, Block {block}")

                self.time_video.append(time_vec[ind0:])
                self.hand_position.append(transformed_lm_pos)
                self.ref_position.append(rotated_ref_pos)

    def calculate_hand_speeds(self, session_data: SessionData) -> None:
        """Calculates the speed of each hand landmark.

        Parameters
            session_data (SessionData): class containing the pipeline session data
        """

        # Calculate the index fingertip speed
        block_itrs = zip(self.time_video, self.hand_position, session_data.event_onsets_blocks)
        for block_time, hand_pos, onset_times in block_itrs:
            # Interpolate the time to be used in the spline (video resolution is too low for differentiation)
            time_interpolated = np.arange(block_time[0], block_time[-1], DT_SPEED)

            lm_pos_interpolated = {}
            lm_hand_speed = {}
            # lm_hand_acc = {}
            for landmark_name, landmark_id in self.hand_landmarks.items():
                # Separate the x and y position data into 1D vectors
                x_tip, y_tip = hand_pos[landmark_id][0, :], hand_pos[landmark_id][1, :]

                # Fit a cubic spline to the position data
                x_spline = cubic_spline_filter(block_time, x_tip, time_interpolated, SMOOTHING)
                y_spline = cubic_spline_filter(block_time, y_tip, time_interpolated, SMOOTHING)
                nan_mask = np.isnan(x_spline)
                time_interpolated = time_interpolated[~nan_mask]
                x_spline = x_spline[~nan_mask]
                y_spline = y_spline[~nan_mask]
                lm_pos_interpolated[landmark_id] = np.stack((x_spline, y_spline), axis=0)

                # Calculate the x, y and combined speeds
                x_speed = calculate_time_derivative(time_interpolated, x_spline)
                y_speed = calculate_time_derivative(time_interpolated, y_spline)
                total_speed = np.sqrt(x_speed ** 2 + y_speed ** 2)
                lm_hand_speed[landmark_id] = np.stack((x_speed, y_speed, total_speed), axis=0)

            self.time_interpolated.append(time_interpolated)
            self.hand_pos_interpolated.append(lm_pos_interpolated)
            self.hand_speed.append(lm_hand_speed)


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

    In the true apparatus setup, the ratio of lengths between the v1 basis and the v2 basis is ~0.94.
    This value is approximate, and was derived from the image of the experimental setup. A scaling
    factor is included in the transformation matrix, to account for this difference of basis lengths.

    The v1 basis is defined as the horizontal distance between the top-left corners of the top two
    apparatus AprilTags. The v2 basis is defined as the vertical distance between the top-left corners
    of the two left-most (right-most if the participant is left handed) apparatus AprilTags.

    Parameters
        reference_coordinates (dict[int, np.ndarray]): top-left corner coordinates of the three reference AprilTags
        index (int): video frame index to calculate for
        reference_tag_ids (list[int]): AprilTag IDs used as reference points (origin, y-dir, x-dir)

    Returns
        (np.ndarray): scaling matrix for a given set of basis vectors
    """

    basis_v1, basis_v2 = get_basis_vectors(reference_coordinates, index, reference_tag_ids)

    return np.diag([1 / np.linalg.norm(basis_v1), BASIS_RATIO / np.linalg.norm(basis_v1)])


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

    Returns
        (np.ndarray): filtered and interpolated position data
    """

    tck = splrep(time_existing, position_existing, s=smoothing)
    return BSpline(*tck, extrapolate=False)(time_interpolated)


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


def main(plot_landmarks: list[int], plot_vectors: bool, overwrite_data: bool):
    """Script that transforms the hand postions to a consistent frame of reference.

    The frame of reference used is the AprilTags set on the corners of the experimental apparatus.

    Parameters
        plot_landmarks (dict[str, int]): names (keys) and IDs (values) of the hand landmarks to plot, if any
        plot_vectors (bool): if True, plot the transformed vectors for each video frame
        overwrite_data (bool): if True, overwrite the current data
    """

    # Load a list of all the pipeline data files
    fpaths, files = get_files_containing("../data/pipeline_data", "pipeline_data.pkl")

    # Load the scaling matrix, or create it, if one does not already exist
    scaling_matrix_fpath = "../data/combined_sessions"
    scaling_matrix_fname = "scaling_matrix.pkl"
    if os.path.exists(os.path.join(scaling_matrix_fpath, scaling_matrix_fname)):
        with open(os.path.join(scaling_matrix_fpath, scaling_matrix_fname), "rb") as f:
            scale_matrix = pickle.load(f)
    else:
        with open(os.path.join(fpaths[0], files[0]), "rb") as f:
            reference_data: SessionData = pickle.load(f)
        scale_matrix = get_scaling_matrix(reference_data.reference_pos_abs[0], 0, reference_data.apparatus_tag_ids)
        pathlib.Path(scaling_matrix_fpath).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(scaling_matrix_fpath, scaling_matrix_fname), "wb") as f_scaling:
            pickle.dump(scale_matrix, f_scaling)
    print(f"Scaling matrix:\n{scale_matrix}")

    for fpath, file in zip(fpaths, files):
        # Skip data that has already been transformed
        participant_id, session_id = fpath.split("/")[-2:]
        fname = f"{participant_id}_{session_id}_transformed_hand_data.pkl"
        if not overwrite_data and not plot_vectors and os.path.exists(os.path.join(fpath, fname)):
            print(f"{fname} already exists")
            continue

        # Load the session data
        session_data = load_session_data(participant_id, session_id)

        # Calculate the hand positions relative to the apparatus frame
        hand_data = TransformedHandData(session_data)
        hand_data.transform_hand_positions(session_data, scale_matrix, plot_landmarks, plot_vectors)

        # Calculate the speed of each landmark from the position data
        hand_data.calculate_hand_speeds(session_data)

        if overwrite_data:
            # Save the transformed hand data
            with open(os.path.join(fpath, fname), "wb") as f:
                pickle.dump(hand_data, f)


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
    parser.add_argument(
        "-p", "--plot_vectors",
        action="store_true",
        help="If True, plot the transformed vectors for each video frame"
    )
    parser.add_argument(
        "-lm", "--landmark_ids",
        type=int,
        nargs="*",
        choices=list(pipeline_config.tracked_hand_landmarks.values()),
        help="IDs of hand landmark positions to plot after transformation"
    )
    args = parser.parse_args()

    main(args.landmark_ids, args.plot_vectors, args.overwrite)
