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
@click.argument('--data_url',
                default= 'http://yann.lecun.com/exdb/mnist/'
                )
@click.argument('--output_filepath',
                default=os.path.join(os.getcwd(), "data/raw/"),
                type=click.Path()
                )
@click.option("--overwrite/--no-overwrite",
              default=False)
def main(__data_url, __output_filepath, overwrite):

    """ Runs data processing scripts to download data to /raw, ready for processing. Does not overwrite existing files
    by default. Set overwrite flag to do this (carefully)

    Args:
        --data_url: url to download data from
        --output_filepath: path to save the downloaded data, by default this is the sub-directory raw/ in the current
            working directory
        --overwrite: flag to overwrite existing data, default is false to prevent accidental re-write

    Returns:
        list of files downloaded and their locations
    """

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    downloaded_files = [download_data(files_to_download[key], download_url=__data_url)
                        for key in files_to_download
                        ]

    data_arrays = [open_and_parse_data(file)
                   for file in downloaded_files]

    locations, saved, replaced = ([],[],[])

    if overwrite:
        logger.info(f"Overwriting as flag is {overwrite}")
    else:
        logger.info(f"Not overwriting as flag is {overwrite}")

    for data, key in zip(data_arrays, files_to_download):
        location, save, replace = save_data_array(data,
                                                  file_name=key,
                                                  save_location=__output_filepath,
                                                  overwrite=overwrite)
        locations.append(location)
        saved.append(save)
        replaced.append(replaced)

    logger.info("complete, files downloaded are...")

    for key, location, save, replace in zip(files_to_download, locations, saved, replaced):

        if save and replace:
            logger.info(f"{key} replaced old file in {location}")
        elif not save:
            logger.info(f"{key} file exists in {location} so not saved. Set --overwrite flag to replace old file")
        else:
            logger.info(f"{key} saved in {location}")

    return locations



def download_data(file_name,
             download_url,
             target_dir=temp_dir
             ):

    """
    downloads file_name from data_url into target_directory

    Args:
        file_name: file to download from download_url
        download_url: Address to download data from
        target_dir: directory to store data

    Returns:
        (string): name and location of file downloaded on successful download

    """
    logger = logging.getLogger(__name__)

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


def open_and_parse_data(file_location):

    logger = logging.getLogger(__name__)
    logger.info(f"unpacking {file_location}")

    if os.path.splitext(file_location)[1] == ".gz":
        with gzip.open(file_location) as file:
            return parse_data(file)

    else:
        with open(file_location) as file:
            return parse_data(file)


def parse_data(file):
    """
    Check data integrity and return data if it appears correct

    Args:
        file: file to parse

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
    logger = logging.getLogger(__name__)
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

    logger.info(("...parse complete"))

    return data


def save_data_array(data,
                    file_name,
                    save_location,
                    overwrite=False
                    ):
    """
    Saves downloaded data as binary (.npy) file
    Args:
        data: numpy array containing data to save
        file_name: name of the file
        save_location: path to save
        overwrite: boolean to overwrite any files, by default this if false

    Returns:

    """
    logger = logging.getLogger(__name__)
    file_path = os.path.join(save_location, file_name)

    # behaviour depends on if file exists and what the overwrite flag is set to
    if os.path.exists(file_path+".npy") & overwrite:
        logger.warn(f"{file_path} exists, overwriting as --overwrite flag is set")
        np.save(file_path, data)
        saved = True
        replaced = True
    elif os.path.exists(file_path+".npy") & (not overwrite):
        logger.warn(f"{file_path} exists, not overwriting as --no-overwrite flag or no flag is set")
        saved = False
        replaced = False
    else:
        logger.info(f"Saving {file_name} as {file_path}")
        np.save(file_path, data)
        saved = True
        replaced = False

    logger.info("done")
    return file_path, saved, replaced

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
