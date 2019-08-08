# -*- coding: utf-8 -*-
import click
import logging
import tempfile
import os
import gzip

from pathlib import Path

from urllib.request import urlretrieve
from urllib.parse import urljoin

from dotenv import find_dotenv, load_dotenv

data_url = 'http://yann.lecun.com/exdb/mnist/'
temp_dir = tempfile.gettempdir()

files_to_download = {"X_train": 'train-images-idx3-ubyte.gz',
                     "X_test": 't10k-images-idx3-ubyte.gz',
                     "y_train": 'train-labels-idx1-ubyte.gz',
                     "y_test": 't10k-labels-idx1-ubyte.gz',
                     }


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())


class DecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass

class DownloadError(RuntimeError):
    """Raised when downloading fails"""
    pass

def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    downloaded_files = [get_data(files_to_download[key])
                        for key in files_to_download
                        ]



def get_data(file_name,
             download_url=data_url,
             target_dir=temp_dir,
             logger=logger
             ):

    """
    Args:
        file_name: file to download from data_url
        download_url: Address to download data from
        target_dir: directory to store data
        logger: logger from module logging

    Returns: name and location of file downloaded on successful download, NoneType on
        unsuccessful download

    """

    file_to_download = os.path.join(target_dir, file_name)
    file_url = urljoin(data_url, file_name)

    logger.info(f"Downloading: {file_name} from: {download_url}")

    try:
        urlretrieve(file_url, file_to_download)
        logging.info("success!")
        return file_to_download
    except:
        logger.info("download unsuccessful")
        raise DownloadError(f"{file_url} not downloaded")


def open_and_parse_data(file_location, logger=logger):

    logger.info(f"unpacking {file_location}")

    if os.path.splitext(file_location)[1]==".gz":
        with gzip.open(file_location) as file:
            return parse_data(file)

    else:
        with open(file_location) as file:
            return parse_data(file)


def parse_data(file, logger=logger):

    logger.info("begining parsing...")

    header = file.read(4)

    if len(header) !=4: 




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
