from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_examples_from_dataset(ds: Dataset, idxs: List[int], title: str = "", fig: Figure = None) -> Optional[Figure]:
    """Plot images from the dataset according to $idxs.

    Args:
        ds: torch image Dataset
        idxs: Indices that we want to plot
        title: Title for the plot
        fig: An optional figure to plot on

    Returns:
        Figure is a figure was passed, else it plots the images as described.
    """
    image_datas = [ds[i][0].numpy().swapaxes(0, 2).swapaxes(0, 1) for i in idxs]
    if image_datas[0].dtype == np.float32:
        for i, img in enumerate(image_datas):
            im_max, im_min = max(1.0, img.max()), min(0.0, img.min())
            img = (img - im_min) / (im_max - im_min)
            image_datas[i] = img_as_ubyte(img)
    image_labels = [ds.classes[ds[i][1]] for i in idxs]

    fig_avail = True
    if fig is None:
        fig_avail = False
        fig = plt.figure()
    axes: List[List[Axes]] = fig.subplots(
        len(idxs) // 3 + (1 if len(idxs) % 3 != 0 else 0), min(len(idxs), 3), squeeze=False
    )

    fig.suptitle(title)

    if len(idxs) == 1:
        axes[0][0].imshow(image_datas[0])
        axes[0][0].axis("off")
        (_width, height) = fig.get_size_inches()
        fig.set_size_inches(height, height)

    for j, idx in enumerate(idxs):
        axes[j // 3][j % 3].imshow(image_datas[j])
        axes[j // 3][j % 3].axis("off")
        axes[j // 3][j % 3].set_title(f"{idx} - {image_labels[j]}")
    fig.tight_layout()
    plt.tight_layout()
    if fig_avail:
        return fig
    else:
        plt.show()
        return None
