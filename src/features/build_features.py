
from scipy.ndimage.interpolation import shift


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