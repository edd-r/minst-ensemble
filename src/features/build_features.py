import os
import logging
import click

import numpy as np

from tqdm import tqdm
from scipy.ndimage.interpolation import shift

raw_file_names = {"X_train": "X_train.npy",
                  "X_test": "X_test.npy",
                  "y_train": "y_train.npy",
                  "y_test": "y_test.npy"
                  }


@click.argument("--raw_data_location",
                default=os.path.join(os.getcwd(), "data/raw/"),
                type=click.Path()
                )
@click.argument("file_names",
                default=raw_file_names
                )
@click.argument("pixel_shift",
                default=0,
                type=int
                )
@click.argument("pixel_increments",
                 default=0,
                 type=int
                )
def main(__raw_data_location,
         file_names,
         pixel_shift,
         pixel_increments,
         blur
         ):

    logger =logging.getLogger(__name__)

    logger.info("Loading raw data data")
    data_dict = get_all_data(file_names=file_names,
                              directory=__raw_data_location
                             )

    # do we have values to created shifted images for data augmentation?
    if bool(pixel_shift) | bool(pixel_increments):
        logger.info("Augmenting data")

        try:
            #augment
            X_train_augmented, y_train_augmented = augment_images(data_dict["X_train"],
                                                                  data_dict["y_train"],
                                                                  pixels_to_shift=pixel_shift,
                                                                  times_to_shift=pixel_increments
                                                                  )
            # concatenate these
            data_dict["X_train"] = np.concatenate([data_dict["X_train"], X_train_augmented])
            data_dict["y_train"] = np.concatenate([data_dict["y_train"], y_train_augmented])

        except KeyError:
            logger.warn("X train or y train data not found, skipping augmentation")

    else:
        logger.info("No data augmentation applied"
                    "set pixel_shift and pixel increments to positive integers to augment data if required")




    return None


def get_all_data(file_names,
                 directory=os.path.join(os.getcwd(),"data/raw/")
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
    files_to_return=sorted(set(files_to_return), key=list(file_names.keys()).index)

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
                     [0,pixels_to_shift],
                     cval=0,
                     mode="constant"
                     )
    # shift_left
    left_image = shift(image,
                       [0,-pixels_to_shift],
                       cval=0,
                       mode="constant"
                      )
    # shift_up
    up_image = shift(image,
                       [-pixels_to_shift,0],
                       cval=0,
                       mode="constant"
                      )
    # shift_down
    down_image = shift(image,
                       [pixels_to_shift,0],
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
    for X, y in tqdm(zip(x_data, y_data), desc="Augmenting data"):

        # perform times_to_shift shifts in pixels_to_shift increments in all directions for the image

        augmented_data = [shift_image(X, pixels_to_shift*n)
                          for n in range(1, times_to_shift+1)
                         ]

        augmented_data = np.concatenate(augmented_data, axis=0)

        # add this to the augmented dataset

        x_augmented.append(augmented_data)
        y_augmented.append(np.full(augmented_data.shape[0], fill_value=y))

    return np.concatenate(x_augmented), np.concatenate(y_augmented)



def blur_data(data):

    return data


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

