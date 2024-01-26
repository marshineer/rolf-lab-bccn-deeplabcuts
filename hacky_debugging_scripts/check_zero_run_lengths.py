"""This script checks how many runs of zeros there are above a certain threshold (MAX_ZERO_RUN).
I used it to determine where interpolation would be valid and to just get an idea of how much data
is likely unusable, if I excluded entire blocks, rather than trying to exclude individual trials."""

import os
import pickle
from utils.calculations import checK_apriltag_gaps
from utils.data_loading import get_files_containing, load_postprocessing_config


fname_config = "config_post.json"
if __name__ == "__main__":
    total_blocks = 0
    skipped_blocks = 0

    fpaths, files = get_files_containing("../data/pipeline_data", "pipeline_data.pkl")
    for fpath, file in zip(fpaths, files):
        # Load the session data
        print(f"Checking: '{file}'")
        participant_id, session_id = fpath.split("/")[-2:]
        with open(os.path.join(fpath, file), "rb") as f:
            session_data_class = pickle.load(f)

        # Load the postprocessing config file
        postprocess_config = load_postprocessing_config(fpath)

        for i, block_ref_pos in enumerate(session_data_class.reference_pos_abs):
            total_blocks += 1
            print(f"Block {i}")
            if checK_apriltag_gaps(block_ref_pos):
                skipped_blocks += 1
                postprocess_config.skip_blocks.append(i)
        postprocess_config.skip_blocks = list(set(postprocess_config.skip_blocks))
        print()

    print(f"Skipped {skipped_blocks}/{total_blocks} blocks")
