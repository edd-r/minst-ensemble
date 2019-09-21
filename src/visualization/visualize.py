
import os
import sys
import tqdm
import logging

import itertools as it
import numpy as np
import matplotlib.pyplot as plt

# add project modules to the path
path_to_module = os.path.abspath(os.path.join(os.getcwd(), "..", "src/"))
sys.path.append(path_to_module)



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

    """
    shows the mean class images on a single plot
    Args:
        images_dict: key value pairs of each class (key) and value (mean image)
        plotting: plot inline or not

    Returns: matplotlib figure and axes containing the mean images by class

    """

    fig, axs = plt.subplots(len(images_dict), sharex=True, sharey=True)

    for key, fig in zip(images_dict, range(len(images_dict))):
        image = images_dict[key]
        axs[fig].imshow(image)

    if plotting:
        plt.plot()

    return fig, axs


def compare_images(image_dict_list, image_categories=None):

    """
    Creates plot of mean images by class and processing method for visual comparisons
    Args:
        image_dict_list: list of dictionaries, each containing key value pairs of images by class for the different
            pre-processing method
        image_categories: list of strings corrisponding to classes in image_dict_list

    Returns: matplotlib figure and axes objects of the created plot

    """
    logger = logging.getLogger(__name__)

    # number the classes if image_categories has not been passed
    if image_categories is None:
        image_categories = [f"column {n+1}"
                            for n in range(len(image_dict_list))]

    # get all the keys for each of our dictionaries in list
    keys = [list(dictionary.keys())
            for dictionary in image_dict_list]

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

    for key_index, image_index in tqdm.tqdm_notebook(it.product(range(len(key_list)), range(len(image_dict_list))),
                                                     desc="plotting"
                                                     ):

        logger.info(f"plotting at {key_index}:{image_index}")
        axes[key_index, image_index].imshow(image_dict_list[image_index][key_list[key_index]])

        # set labels for top and leftmost images
        if key_index == 0:
            axes[key_index, image_index].set_xlabel(image_categories[image_index])
            axes[key_index, image_index].xaxis.set_label_position("top")

        if image_index == 0:
            axes[key_index, image_index].set_ylabel(f"label: {key_list[key_index]}")

    return fig, axes



