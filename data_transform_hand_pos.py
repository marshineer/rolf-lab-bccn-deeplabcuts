import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import get_files_containing, get_basis_vectors
from utils_pipeline import SessionData, load_session_data


MIN_POSITION = 0.1


class TransformedHandData:
    """This class stores the transformed hand landmark data.

    The hand data is transformed from the video frame of reference to the frame of reference defined
    by the AprilTags on the experimental apparatus, with IDs: 10 (y-dir basis), 20 or 30 (x-dir basis),
    and 40 (origin). This transformation includes a scaling factor, such that data from all sessions,
    blocks, and time points are comparable.

    The hand landmark speeds are calculated in both the x- and y-directions, as well as the combined
    magnitude. Reference position data is included for checking later.

    Attributes
        hand_landmarks (dict[int, str]): names (values) of landmarks associated with each ID (keys)
        time (list[np.ndarray]): time data for each block
        hand_position (list[dict[str, np.ndarray]]): transformed hand landmark position data
        hand_speed (list[dict[str, np.ndarray]]): hand landmark speed data for each block and landmark

    """

    def __init__(self, hand_landmarks):
        self.hand_landmarks: dict[int, str] = hand_landmarks
        self.time: list[np.ndarray] = []
        self.hand_position: list[dict[str, np.ndarray]] = []
        self.hand_speed: list[dict[str, np.ndarray]] = []
        self.ref_pos: list[dict[str, np.ndarray]] = []

    def calculate_hand_speeds(self) -> None:
        """Calculates the x, y and total speed of the transformed hand position data."""

        for block_time, block_pos in zip(self.time, self.hand_position):
            lm_speed_dict = {}
            for lm, pos in block_pos.items():
                lm_speed_dict[lm] = np.zeros((3, block_time.size - 1))
                for i in range(pos.shape[0]):
                    lm_speed_dict[lm][i, :] = calculate_speed(pos[i, :], block_time)
                lm_speed_dict[lm][-1, :] = np.sqrt(lm_speed_dict[lm][0, :] ** 2 + lm_speed_dict[lm][1, :] ** 2)
            self.hand_speed.append(lm_speed_dict)


def transform_hand_positions(
        output_data: TransformedHandData,
        session_data_class: SessionData,
        reference_scale_matrix: np.ndarray,
        plot_landmarks: dict[str, int] = None,
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
    time.

    The same AprilTags cannot always be used as references. Therefore, the AprilTag IDs used to
    define the coordinates of a reference frame are provided in the order [origin, v1, v2].

    Parameters
        output_data (PostProcessedData): dataclass in which post-processed data is stored
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
        tranformed_lm_pos = {lm: np.zeros(position_shape) for lm in session_data_class.tracked_landmark_ids}
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
            for lm, lbl in zip(session_data_class.tracked_landmark_ids, session_data_class.tracked_landmark_names):
                # Interpolate the hand landmark positions
                lm_xy = interpolate_pos(time_vec, lm_pos[lm])[:, ind0:]
                # Translate the coordinates to the origin of the reference frame
                rel_xy = lm_xy - interp_ref_pos[reference_tag_id]
                # Transform the coordinates into the reference frame
                tranformed_lm_pos[lm][:, i] = transformation_matrix @ rel_xy[:, i]

            # Transform reference AprilTags
            for tag_id in session_data_class.apparatus_tag_ids:
                rotated_ref_pos[tag_id][:, i] = transformation_matrix @ interp_ref_pos[tag_id][:, i]

        if plot_landmarks is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            for landmark_name, landmark_id in plot_landmarks.items():
                ax.plot(time_vec[ind0:], tranformed_lm_pos[landmark_id][0, :], label=f"X: {landmark_name}")
                ax.plot(time_vec[ind0:], tranformed_lm_pos[landmark_id][1, :], label=f"Y: {landmark_name}")
            ax.set_title(f"{participant_id}-{session_id}-Block: {block} (Ref Tag: #{reference_tag_id})")
            ax.legend()
            plt.show()
            plt.close()
        else:
            print(f"{participant_id}-{session_id}-Block: {block} (Ref Tag: #{reference_tag_id})")

        output_data.time.append(time_vec[ind0:])
        output_data.hand_position.append(tranformed_lm_pos)
        output_data.ref_pos.append(rotated_ref_pos)


def interpolate_pos(time_data: np.ndarray, position_data: np.ndarray) -> np.ndarray:
    """Interpolates the hand landmark and reference position data.

    Parameters
        time_data (np.ndarray): the timestamp for each block frame
        position_data (np.ndarray): the x-y position for each block frame

    Returns
        np.ndarray: the interpolated x- and y-values of the data
    """

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


def calculate_speed(time: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Calculates speed from position data.

    Parameters
        time (np.ndarray): time vector associated with the postion data
        position (np.ndarray): single dimension positon data

    Returns
        (np.ndarray): vector of speed at each time point
    """

    return np.diff(position) / np.diff(time)


if __name__ == "__main__":
    # Initialize the variables used for all participants and sessions
    with open("data/pipeline_data/P17/A1/P17_A1_pipeline_data.pkl", "rb") as f:
        reference_data: SessionData = pickle.load(f)
    scale_matrix = get_scaling_matrix(reference_data.reference_pos_abs[0], 0, reference_data.apparatus_tag_ids)

    fpaths, files = get_files_containing("data/pipeline_data", "pipeline_data.pkl")
    for fpath, file in zip(fpaths, files):
        # Load the session data
        participant_id, session_id = fpath.split("/")[-2:]
        session_data = load_session_data(participant_id, session_id)

        # Calculate the hand positions relative to the apparatus frame
        tracked_landmarks = {lm_id: lm for lm_id, lm in zip(session_data.tracked_landmark_ids,
                                                            session_data.tracked_landmark_names)}
        hand_data = TransformedHandData(tracked_landmarks)
        transform_hand_positions(
            hand_data,
            session_data,
            scale_matrix,
            # {"Index Tip": 8},
        )

        # Calculate the x, y and total speed of the hand landmarks
        hand_data.calculate_hand_speeds()

        # Save the transformed hand data
        fname = f"{participant_id}_{session_id}_transformed_hand_data.pkl"
        with open(os.path.join(fpath, fname), "wb") as f:
            pickle.dump(hand_data, f)
