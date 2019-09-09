import os
import logging
import tqdm

import numpy as np


file_names = {"X_train": "X_train.npy",
              "X_test": "X_test.npy",
              "y_train": "y_train.npy",
              "y_test": "y_test.npy"
              }


def load_processed_data(file_path=os.path.join(os.getcwd(), "data/processed/"),
                        file_names=file_names,
                        augmented_suffix="_augmented",
                        blurred_suffix="_blurred",
                        scaled_suffix="_scaled"
                        ):
    """
    loads processed data into key value pairs in a dictionary with keys matching the keys in file_names and values as
    the data

    Args:
        file_path: path where the processed data is stored
        file_names: key value pairs of the file to load and its file name
        augmented_suffix: load augmented data with this suffix
        blurred_suffix: load blurred data with this suffix
        scaled_suffix: load scaled data with this suffix

    Returns:
        key value pairs of the name of the data as the key and the data itself as the value

    """

    logger = logging.getLogger(__name__)
    data = []
    for key in tqdm.tqdm(file_names, desc=f"loading files from {file_path}"):

        file_name = file_names[key]

        if augmented_suffix is not "":
            file_name = file_name.replace(".", augmented_suffix + ".")

        if blurred_suffix is not "":
            file_name = file_name.replace(".", blurred_suffix + ".")

        if scaled_suffix is not "":
            file_name = file_name.replace(".", scaled_suffix + ".")

        file_to_load = os.path.join(file_path, file_name)
        logger.info(f"loading {key} in file {file_to_load}")

        try:
            data.append(np.load(file_to_load))
        except IOError:
            logging.ERROR(f"{file_to_load}: Error in loading, is path correct?")
            logging.WARNING(f"Returning empty array for {key}")
            data.append(np.empty())

    return dict([(key, value) for key, value in zip(file_names.keys(), data)])

