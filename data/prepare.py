import os
import numpy as np
import json
from utils import download_dataset, char_handler

dataset = "github"
streaming_samples = 5000
overwrite = True  # overwrite the dataset if exists
# ---------------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), dataset)
    os.makedirs(data_dir, exist_ok=True)
    input_file_path = os.path.join(data_dir, dataset)

    # -------------------------------------------------------------
    if not os.path.exists(input_file_path) or overwrite:
        download_dataset(dataset, streaming_samples, input_file_path)

    with open(input_file_path, "r", encoding="utf-8") as f:
        data = f.read()

    # char
    ids, meta = char_handler(data)

    # export to bin files
    ids = np.array(ids, dtype=np.uint16)
    ids.tofile(os.path.join(data_dir, f"char.bin"))

    with open(os.path.join(data_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4)
