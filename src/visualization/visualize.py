
import os
import sys
import tqdm
import logging

import numpy as np
import matplotlib.pyplot as plt

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

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)