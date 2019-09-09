
import os
import sys
import tqdm
import logging

import pandas as pd
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import kneed as kn
import sklearn.decomposition as decomp

# add project modules to the path
path_to_module = os.path.abspath(os.path.join(os.getcwd(), "..", "src/"))
sys.path.append(path_to_module)

import src.models.train_model as train_model


def average_image_by_class(X,
                           y,
                           image_size=(28, 28)):
    """
    creates average image from images in X for each unique label in y
    Args:
        X: array of flattened images
        y: label for each image
        image_size: unflattened size of images in X

    Returns: dictionary containing key, value pairs of label and the average image

    """
    mean_images = []
    for label in tqdm.tqdm(np.unique(y),
                           desc="Creating mean images for each class label"
                           ):

        label_mask = y == label
        x_label = X[label_mask]
        image = x_label.mean(axis=0)
        image = image.reshape(image_size)
        mean_images.append(image)

    return dict([(label, image) for label, image in zip(np.unique(y), mean_images)])


def plot_images(images_dict, plotting=True):

    fig, axs = plt.subplots(len(images_dict), sharex=True, sharey=True)

    for key, fig in zip(images_dict, range(len(images_dict))):
        image = images_dict[key]
        axs[fig].imshow(image)

    if plotting:
        plt.plot()
    return fig, axs


def compare_images(image_dict_list, image_categories=None):

    logger = logging.getLogger(__name__)

    if image_categories is None:
        image_categories = [f"column {n+1}" for n in range(len(image_dict_list))]

    # get all the keys for each of our dictionaries in list
    keys = [list(dictionary.keys()) for dictionary in image_dict_list]

    # same order
    for key in keys:
        key.sort()

    # are the keys for all our dictionaries the same
    if not all([a == b for a, b in it.combinations(keys, 2)]):
        logger.error("Passed dictionaries don't have the same keys, cannot compare")
        return None
    else:
        key_list = keys[0]  # we have established all lists in keys are identical, so take the first

    fig, axes = plt.subplots(nrows=len(key_list),
                             ncols=len(image_dict_list),
                             figsize=(10, 10))
    fig.subplots_adjust(wspace=0, hspace=0)

    for key_index, image_index in it.product(range(len(key_list)), range(len(image_dict_list))):

        logger.info(f"plotting at {key_index}:{image_index}")
        axes[key_index, image_index].imshow(image_dict_list[image_index][key_list[key_index]])

        # set labels for top and leftmost images
        if key_index == 0:
            axes[key_index, image_index].set_xlabel(image_categories[image_index])
            axes[key_index, image_index].xaxis.set_label_position("top")

        if image_index == 0:
            axes[key_index, image_index].set_ylabel(f"label: {key_list[key_index]}")

    return fig, axes



