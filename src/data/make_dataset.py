# -*- coding: utf-8 -*-
import struct
import click
import logging
import tempfile
import os
import gzip
import array

import numpy as np

from pathlib import Path

from urllib.request import urlretrieve
from urllib.parse import urljoin

from dotenv import find_dotenv, load_dotenv

from scipy.ndimage.interpolation import shift

data_url = 'http://yann.lecun.com/exdb/mnist/'
temp_dir = tempfile.gettempdir()

files_to_download = {"X_train": 'train-images-idx3-ubyte.gz',
                     "X_test": 't10k-images-idx3-ubyte.gz',
                     "y_train": 'train-labels-idx1-ubyte.gz',
                     "y_test": 't10k-labels-idx1-ubyte.gz',
                     }

class DecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


class DownloadError(RuntimeError):
    """Raised when downloading fails"""
    pass

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    downloaded_files = [download_data(files_to_download[key])
                        for key in files_to_download
                        ]


def download_data(file_name,
             download_url=data_url,
             target_dir=temp_dir,
             logger=logging.getLogger(__name__)
             ):

    """
    downloads file_name from data_url into target_directory

    Args:
        file_name: file to download from download_url
        download_url: Address to download data from
        target_dir: directory to store data
        logger: logger from module logging

    Returns:
        (string): name and location of file downloaded on successful download

    """

    file_to_download = os.path.join(target_dir, file_name)
    file_url = urljoin(download_url, file_name)

    logger.info(f"Downloading: {file_name} from: {download_url}")

    try:
        urlretrieve(file_url, file_to_download)
        logging.info("success!")
        return file_to_download
    except:
        logger.info("download unsuccessful")
        raise DownloadError(f"{file_url} not downloaded")


def open_and_parse_data(file_location, logger=logging.getLogger(__name__)):

    logger.info(f"unpacking {file_location}")

    if os.path.splitext(file_location)[1] == ".gz":
        with gzip.open(file_location) as file:
            return parse_data(file)

    else:
        with open(file_location) as file:
            return parse_data(file)


    def parse_data(file,
                   logger=logging.getLogger(__name__)
                   ):
        """
        Check data integrity and return data if it appears correct

        Args:
            file: file to parse
            logger: logger from logging module

        Returns:
            (array): numpy array containing data
        """

        data_types_dict = {0x08: 'B',  # unsigned byte
                           0x09: 'b',  # signed byte
                           0x0b: 'h',  # short (2 bytes)
                           0x0c: 'i',  # int (4 bytes)
                           0x0d: 'f',  # float (4 bytes)
                           0x0e: 'd'  # double (8 bytes))
                           }
        logger.info("Begin parsing...")

        header = file.read(4)

        # go through some checks for data integrity
        if len(header) != 4:
            raise DecodeError("Invalid file; empty or does not contain header")

        zeros, data_type, n_dims = struct.unpack('>HBB', header)

        if zeros != 0:
            raise DecodeError("Invalid file; must start with two zero bytes")

        try:
            data_type = data_types_dict[data_type]
        except KeyError:
            raise DecodeError(f"Unknown data type:{data_type} in file")

        # get the data
        data_dims = struct.unpack('>' + 'I' * n_dims, file.read(4*n_dims))
        data = array.array(data_type, file.read())

        # check size of data compared to expected size
        if len(data) != np.prod(data_dims):
            raise DecodeError(f"File has incorrect number of elements. Expected: {len(data)} Found: {np.prod(data_dims)}")

        # format as numpy array and return
        data = np.array(data).reshape(data_dims)

        return data


def download_and_parse(file_name,
                       download_url=data_url,
                       target_dir = temp_dir,
                       shift_images = 0,
                       shift_times = 0,
                       blur=True
                      ):
    """
    Download and parse data
    Args:
        file_name: file to download
        download_url: location of file
        target_dir: directory to download to
        augment: apply data augmentation
        blur: apply blur to data
    Returns:
        (array) numpy array containing data
    """

    file_location = download_data(file_name,
                                  download_url=download_url,
                                  target_dir=target_dir)

    data = open_and_parse_data(file_location)



    return data

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

    return np.array([up_image,down_image,left_image,right_image])


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
    for X, y in zip(x_data,y_data):

        # perform times_to_shift shifts in pixels_to_shift increments in all directions for the image

        augmented_data = [shift_image(X, pixels_to_shift*n)
                          for n in range(1, times_to_shift+1)
                         ]
        augmented_data = np.concatenate(augmented_data, axis=0)

        # add this to the augmented dataset
        x_augmented.append(augmented_data)
        y_augmented = np.full(augmented_data.shape[0], fill_value=y)

    return x_augmented, y_augmented






def blur_data(data):

    return data

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
