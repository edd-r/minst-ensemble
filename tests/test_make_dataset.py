import os
import pytest
import logging

import numpy as np

# Paramaters for parameetrized tests

# check if these files exist
files = [("X_train.npy"),
         ("X_test.npy"),
         ("y_train.npy"),
         ("y_test.npy")
         ]

observation_files = ("X_train.npy", "X_test.npy")
obs_file_shapes = ((60000,28,28), (10000,28,28))
label_files = ("y_train.npy", "y_test.npy")

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("file_name", files)
def test_file_exists(file_name,
                     file_location=os.path.join(os.getcwd(), "data/raw/")
                     ):

    file_path = os.path.join(file_location, file_name)
    logger.info(f"checking {file_path} exists")
    assert os.path.exists(file_path)


@pytest.mark.parametrize("obs_file, label_file",
                         [(obs, labels) for obs, labels in zip(observation_files, label_files)]
                         )
def test_observations_have_labels(obs_file,
                                  label_file,
                                  file_location=os.path.join(os.getcwd(), "data/raw/")
                                 ):

    obs_path = os.path.join(file_location, obs_file)
    label_path = os.path.join(file_location, label_file)
    obs_array = np.load(obs_path)
    label_array = np.load(label_path)
    logger.info(f"\n number of labels: {label_array.shape[0]} \n number of observations: {obs_array.shape[0]}")
    assert label_array.shape[0] == obs_array.shape[0]

@pytest.mark.parametrize("obs_file, expected_shape",
                         [(file, shape) for file, shape in zip(observation_files, obs_file_shapes)]
                         )
def test_observations_have_correct_shape(obs_file,
                                         expected_shape,
                                         file_location=os.path.join(os.getcwd(), "data/raw/")
                                         ):
    obs_path = obs_path = os.path.join(file_location, obs_file)
    obs_array = np.load(obs_path)
    logger.info(f"{obs_file} shape is {obs_array.shape}, expected {expected_shape}")
    assert obs_array.shape == expected_shape
