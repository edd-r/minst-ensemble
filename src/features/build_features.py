import os
import logging
import click

import numpy as np

from tqdm import tqdm

from scipy.ndimage.interpolation import shift
from scipy.ndimage import gaussian_filter

from dotenv import find_dotenv, load_dotenv

from pathlib import Path


raw_file_names = {"X_train": "X_train.npy",
                  "X_test": "X_test.npy",
                  "y_train": "y_train.npy",
                  "y_test": "y_test.npy"
                  }

# TODO: get these click arguments working to pass variables from terminal


@click.command()
@click.argument("raw_data_location",
                default=os.path.join(os.getcwd(), "data/raw/"),
                type=click.Path()
                )
@click.argument("output_location",
                default=os.path.join(os.getcwd(), "data/processed/"),
                type=click.Path()
                )
@click.argument("file_names",
                default=raw_file_names
                )
@click.argument("pixel_shift",
                default=5,
                type=int
                )
@click.argument("pixel_increments",
                default=1,
                type=int
                )
@click.argument("blur_sigma",
                default=1,
                type=float)
@click.option("--overwrite/--no-overwrite",
              default=False)
def main(raw_data_location,
         output_location,
         file_names,
         pixel_shift,
         pixel_increments,
         blur_sigma,
         overwrite
         ):

    logger = logging.getLogger(__name__)
    file_suffix = ""  # what we append to the saved file names

    logger.info("Loading raw data data")
    data_dict = get_all_data(file_names=file_names,
                             directory=raw_data_location
                             )

    # do we have values to created shifted images for data augmentation?

    if bool(pixel_shift) | bool(pixel_increments):
        logger.info("Augmenting data")
        file_suffix = file_suffix + "_augmented"

        try:
            # augment training data
            X_train_augmented, y_train_augmented = augment_images(data_dict["X_train"],
                                                                  data_dict["y_train"],
                                                                  pixels_to_shift=pixel_shift,
                                                                  times_to_shift=pixel_increments
                                                                  )
            # concatenate these
            data_dict["X_train"] = np.concatenate([data_dict["X_train"], X_train_augmented])
            data_dict["y_train"] = np.concatenate([data_dict["y_train"], y_train_augmented])

        except KeyError:
            logger.warning("X train or y train data not found, skipping augmentation")

    else:
        logger.info("No data augmentation applied"
                    "set pixel_shift and pixel increments to positive integers to augment data if required")

    # do we blur the images
    if blur_sigma > 0:
        file_suffix = file_suffix + "_blurred"

        try:
            logger.info("Blurring training data")
            data_dict["X_train"] = blur_data(data_dict["X_train"], blur_sigma)
        except KeyError:
            logger.warning("X_train data not found, skipping blur")

    else:
        logger.info("No blurring applied. If required; to apply blur set blur sigma to > 0")

    logger.info("Flattening arrays for machine learning models")
    data_dict = flatten_data(data_dict)

    logger.info(f"writing data to {output_location}")
    file_locations, saved, replaced = save_processed_data(data_dict, output_location, file_suffix, overwrite)

    for key, save, replace in zip(data_dict, saved, replaced):
        file_name = key + file_suffix
        if save and replace:
            logger.info(f"{file_name} replaced old file in {output_location}")
        elif not save:
            logger.info(f"{file_name} file exists in {output_location}"
                        "so not saved. Set --overwrite flag to replace old file")
        else:
            logger.info(f"{file_name} saved in {output_location}")

    return file_locations


def get_all_data(file_names,
                 directory=os.path.join(os.getcwd(), "data/raw/")
                 ):
    """
    load all the data for processing

    Args:
        directory: file path where data is stored
        file_names: iterable of file names to load in directory

    Returns:
           dictionary of file names as the key and the numpy data as the value of all files in file_names found in
           directory
    """

    logger = logging.getLogger(__name__)
    paths_to_data = [os.path.join(directory, file_names[key]) for key in file_names]

    data = []
    ioerror_files = []

    for file_path, name in zip(paths_to_data, file_names):

        try:
            data.append(np.load(file_path))
        except IOError:
            logger.error(f"{file_path} not found, skipping.")
            ioerror_files.append(name)

    files_to_return = list(set(file_names.keys())-set(ioerror_files))

    # converting to sets messes up order, so we sort, might be fixed with python 3.6+
    files_to_return = sorted(set(files_to_return), key=list(file_names.keys()).index)

    data_loaded = dict(zip(files_to_return, data))

    return data_loaded


def shift_image(image, pixels_to_shift):

    """
    shifts image by a specifed number of pixels in four directions, up, down, left and right
    Args:
        image:
        pixels_to_shift:

    Returns: numpy array of the four shifted images

    """

    # shift right
    right_image = shift(image,
                        [0, pixels_to_shift],
                        cval=0,
                        mode="constant"
                        )
    # shift_left
    left_image = shift(image,
                       [0, -pixels_to_shift],
                       cval=0,
                       mode="constant"
                       )
    # shift_up
    up_image = shift(image,
                     [-pixels_to_shift, 0],
                     cval=0,
                     mode="constant"
                     )
    # shift_down
    down_image = shift(image,
                       [pixels_to_shift, 0],
                       cval=0,
                       mode="constant"
                       )

    return np.array([up_image, down_image, left_image, right_image])


def augment_images(x_data,
                   y_data,
                   pixels_to_shift,
                   times_to_shift
                   ):

    """
    Creates augmented image data for classification by shifting image up, down, left and right a specified number of
    times in given increments.

    Args:
        x_data: images to shift
        y_data: label of the images to shift
        pixels_to_shift: the increment to shift in pixels
        times_to_shift: number of increments to shift image in each direction

    Returns: augmented images and their class labels

    """

    x_augmented = []
    y_augmented = []

    total_for_tqdm = min(len(x_data), len(y_data))
    for X, y in tqdm(zip(x_data, y_data), total=total_for_tqdm, desc="Augmenting data"):

        # perform times_to_shift shifts in pixels_to_shift increments in all directions for the image

        augmented_data = [shift_image(X, pixels_to_shift*n)
                          for n in range(1, times_to_shift+1)
                          ]

        augmented_data = np.concatenate(augmented_data, axis=0)

        # add this to the augmented dataset

        x_augmented.append(augmented_data)
        y_augmented.append(np.full(augmented_data.shape[0], fill_value=y))

    return np.concatenate(x_augmented), np.concatenate(y_augmented)


def blur_data(X_data, sigma):
    """
    Apply gaussian blur with standard deviation sigma across all axes for each 2D image in the iterable X_data
    Args:
        X_data: contains images to blur
        sigma: standard deviation of the blur applied across all axes

    Returns:
        numpy array containing blurred images from X

    """

    blurred_images = np.array([gaussian_filter(image, sigma)
                               for image in tqdm(X_data, desc="Applying gaussian blur")
                               ])

    return blurred_images


def flatten_data(data_dict, feature_keys=("X_train", "X_test")):
    """
    Flattens training and test images from 2D array to 1D for developing ML models

    Args:
        data_dict: contains the data to flatten as numpy arrays
        feature_keys: iterable containing the keys in data_dict for the data to flatten

    Returns:
        data_dict with flattened images
    """

    for key in feature_keys:
        try:
            # Flatten images from 2D array to 1D array
            data_dict[key] = data_dict[key].reshape(data_dict[key].shape[0], -1)
        except KeyError:
            logging.ERROR("key: {key} not found in data, cannot flatten."
                          "Set feature_keys to correct keys for images in data_dict ")

    return data_dict


def save_processed_data(data_dict, save_location, file_suffix="", overwrite=False):

    """
    saves data in data_dict to path in save_location as .npy files

    Args:
        data_dict: data to save as values in dictionary
        save_location: path to save values in data_dict
        file_suffix: to append to key in data_dict giving the name of the .npy file
        overwrite: flag to avoid accidently overwriting of data. Default is False

    Returns:
        lists containing file names saved, whether the data was saved or not - depending on overwrite flag,
            if and existing file was replaced as a consequence of overwrite set to True

    """
    logger = logging.getLogger(__name__)
    saved = []
    replaced = []
    files = []

    for key in data_dict:
        file_name = key + file_suffix
        file_path = os.path.join(save_location, file_name)
        files.append(file_path)

        # behaviour depends on if file exists and what the overwrite flag is set to
        if os.path.exists(file_path+".npy") & overwrite:
            logger.warning(f"{file_path} exists, overwriting as --overwrite flag is set")
            np.save(file_path, data_dict[key])
            saved.append(True)
            replaced.append(True)

        elif os.path.exists(file_path+".npy") & (not overwrite):
            logger.warning(f"{file_path} exists, not overwriting as --no-overwrite flag or no flag is set")
            saved.append(False)
            replaced.append(False)
        else:
            logger.info(f"Saving {file_name} as {file_path}")
            np.save(file_path, data_dict[key])
            saved.append(True)
            replaced.append(False)

    logger.info("done")

    return files, saved, replaced


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

# not used in this stub but often useful for finding various files
project_dir = Path(__file__).resolve().parents[2]

# find .env automagically by walking up directories until it's found, then
# load up the .env entries as environment variables
load_dotenv(find_dotenv())

main()
