import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_box(ax: plt.Axes, image_shape: tuple[int], box_params: tuple[float] | np.ndarray, text=""):
    """
    params::box_params: (x, y, w, h) is proportion of the image, so we need `image_shape`
        - (x, y) is the center of the box.
        - (w, h) is the width and height of the box.
    params::text: The text display in the upper left of the bounding box.
    """
    params, shape = box_params, image_shape
    x_min = int(shape[1]*(params[0]-params[2]/2))
    y_min = int(shape[0]*(params[1]-params[3]/2))
    w = int(shape[1] * params[2])
    h = int(shape[0] * params[3])
    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.scatter(int(shape[1] * params[0]), int(shape[0] * params[1]), color='yellow', s=50)
    ax.add_patch(rect)
    bbox_props = dict(boxstyle="round, pad=0.2", edgecolor='red', facecolor='red')
    if len(text) != 0:
        ax.text(x_min+2, y_min, text, color='white', backgroundcolor='red', va='bottom', ha='left', fontsize=8, bbox=bbox_props)

def plot_cells(ax: plt.Axes, image_shape: tuple[int], S: int):
    """
    Draw the cells division on the image ax.
    params::S: Split the image into SxS cells.
    """
    x_size, y_size = int(image_shape[1] / S), int(image_shape[0] / S)
    for i in range(1,S):
        ax.plot([i*x_size, i*x_size], [0, image_shape[0]], c='b')
        ax.plot([0, image_shape[1]], [i*y_size, i*y_size], c='b')
