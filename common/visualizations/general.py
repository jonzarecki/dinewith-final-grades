from typing import List, Tuple

import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from matplotlib.figure import Figure


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: list, figsize: Tuple[int, int] = (10, 8), fontsize: int = 14
) -> Figure:
    """Plots a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Args:
        confusion_matrix: The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
                            Similarly, constructed ndarrays can also be used.
        class_names: An ordered list of class names, in the order they index the given confusion matrix.
        figsize:  A 2-long tuple, the first value determining the horizontal size of the outputted figure,
                    the second determining the vertical size. Defaults to (10,7).
        fontsize: Font size for axes labels. Defaults to 14.

    Returns:
        The resulting confusion matrix figure.
    """
    df_cm = pd.DataFrame(
        confusion_matrix,
        index=class_names,
        columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError as ve:
        raise ValueError("Confusion matrix values must be integers.") from ve
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha="right", fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha="right", fontsize=fontsize)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return fig


def figure_to_image(figure: Figure) -> np.ndarray:
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it.

    The supplied figure is closed and inaccessible after this call.
    Args:
        figure: Figure supplied for plotting

    Returns:
        numpy array with the figure's pixels in it's values
    """
    canvas = figure.canvas
    canvas.draw()
    pil_image = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
    plt.close()
    return np.array(pil_image)


def show_ndarray_in_matplotlib(img: np.ndarray):
    img = Image.fromarray(img, "RGB")
    img.save("my.png")
    img.show()


def plot_hv_multi_line_curve(curves: List[List[float]], title: str, labels: List[str] = ()) -> hv.Curve:
    """Plots a multi line curve with holoviews.

    Args:
        curves:  List of lists (of the same length!) with points on the curves
        title: Plot title
        labels: List of labels for the curves

    Returns:
        a holoviews curve object
    """
    assert all(len(curves[0]) == len(curv) for curv in curves), "all curves should be of the same length " \
                                                                "(have the same x axis)"
    labels = [None] * len(curves) if len(labels) == 0 else labels

    curv_plt = hv.Curve(curves[0], label=labels[0])
    for i, curv in enumerate(curves[1:]):
        curv_plt *= hv.Curve(curv, label=labels[i + 1])

    return curv_plt.opts(fontscale=1, width=500, height=400, title=title, toolbar='above', legend_position='top_left')
