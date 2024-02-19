import os
import sys
import cv2
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(".."))
from utils.calculations import get_basis_vectors
from utils.data_loading import load_block_video_mp4
from utils.pipeline import load_session_data, INDEX_FINGER_TIP_ID


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
    parser.add_argument(
        "frame",
        type=int,
        required=True,
        help="Video frame"
    )
    parser.add_argument(
        "-s", "--save_first_frame",
        action="store_true",
    )
    args = parser.parse_args()

    # Load the session data
    session_data = load_session_data(args.participant, args.session)
    time_vec = session_data.block_times[args.block]

    # Load the scaling matrix
    with open("../data/combined_sessions/scaling_matrix.pkl", "rb") as f:
        scale_matrix = pickle.load(f)

    # Define the reference postions
    ref_pos = session_data.reference_pos_abs[args.block]
    reference_tag_id = session_data.apparatus_tag_ids[0]

    # Define the position of the index fingertip
    tip_pos = session_data.hand_landmark_pos_abs[args.block][INDEX_FINGER_TIP_ID]
    tip_pos_trans = np.zeros_like(tip_pos)
    tip_pos_rel = tip_pos - ref_pos[reference_tag_id]

    # Initialize the transformed reference positions
    ref_pos_trans = {tag: np.zeros_like(tip_pos) for tag in session_data.apparatus_tag_ids}
    origin_id, v2_id, v1_id = session_data.apparatus_tag_ids

    # Load the block video
    vcap = load_block_video_mp4(session_data.participant_id, session_data.session_id, args.block)
    width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    i = 0
    while vcap.isOpened():
        ret, frame = vcap.read()
        if i >= args.frame:
            print(f"\nFrame {i + 1}, Block time: {time_vec[i]:0.3f}")
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.set_xlim([-1, width])
            ax.set_ylim([height, -1])
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ax.imshow(frame_rgb)

            # Calculate the transformation matrix for each frame
            basis_v1, basis_v2 = get_basis_vectors(ref_pos, i, session_data.apparatus_tag_ids)
            rotation_matrix = np.stack((basis_v1, basis_v2)).T
            transformation_matrix = np.linalg.inv(rotation_matrix @ scale_matrix)

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
            ax.set_title(f"{session_data.participant_id}-{session_data.session_id}-Block {args.block}, "
                         f"Video Time: {time_vec[i]:0.2f}s (Video Frame {i + 1})", fontsize=19)
            ax.legend(loc=0, framealpha=0.9)
            plt.show()

            if args.save_first_frame and i == args.frame:
                fig.savefig(
                    f"../images/plots/{args.participant}-{args.session}-Block{args.block}-"
                    f"Frame{args.frame}_hand_position_transformation.png", dpi=fig.dpi, bbox_inches="tight"
                )

        i += 1

    vcap.release()
    cv2.destroyAllWindows()
