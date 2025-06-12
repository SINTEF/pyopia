# -*- coding: utf-8 -*-
"""
Particle plotting functionality for standardised figures
e.g. image presentation, size distributions, montages etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import pyopia.statistics


def show_image(image, pixel_size):
    """Plots a scaled figure (in mm) of an image

    Parameters
    ----------
    image : float
        Image (usually a corrected image, such as im_corrected)
    pixel_size : float
        the pixel size (um) of the imaging system used
    """
    r, c = np.shape(image[:, :, 0])

    plt.imshow(
        image,
        extent=[0, c * pixel_size / 1000, 0, r * pixel_size / 1000],
        interpolation="nearest",
    )
    plt.xlabel("mm")
    plt.ylabel("mm")

    return


def montage_plot(montage, pixel_size):
    """
    Plots a SilCam particle montage with a 1mm scale reference

    Parameters
    ----------
    montage : uint8
        a montage created with scpp.make_montage
    pixel_size : float
        the pixel size (um) of the imaging system used
    """
    msize = np.shape(montage)[0]
    ex = pixel_size * np.float64(msize) / 1000.0

    ax = plt.gca()
    ax.imshow(montage, extent=[0, ex, 0, ex], cmap="grey")
    ax.set_xticks([1, 2], [])
    ax.set_xticklabels(["    1mm", ""])
    ax.set_yticks([], [])
    ax.xaxis.set_ticks_position("bottom")


def classify_rois(roilist, classifier):
    """Classify list of single-object images

    If true_class is specified, mark ROIs not matching this class in the figure.

    Parameters
    ----------
    roilist: list
        List of ROI images to classify

    classifier: pyopia.classify.Classify
        Used to classify ROIs

    Returns
    -------
    df: pd.DataFrame
        Class probabilities for each item in roifiles
    """

    # Get class labels from classifier
    class_labels = [f"probability_{cl}" for cl in classifier.class_labels]

    # Classify all ROIs
    classify_data = []
    for img in roilist:
        prediction = classifier.proc_predict(img).numpy()
        classify_data.append(prediction)
    df = pd.DataFrame(columns=class_labels, data=classify_data)

    return df


def plot_classified_rois(roilist, df_class_labels, true_class=None):
    """Plot classified single-object images and show them in a figure grid with classification info

    If true_class is specified, mark ROIs not matching this class in the figure.

    Parameters
    ----------
    roilist: list
        List of ROI image to classify

    df_class_labels: pd.DataFrame
        Class label for each image in roilist in a column named "best guess"

    true_class: str
        True class of listed ROIs


    Returns
    -------
    fig: matplotlib figure
    ax: matplotlib axes
    """
    if "best guess" not in df_class_labels:
        raise RuntimeError(
            "df_class_clabels must contain column 'best guess'. "
            "See e.g. pyopia.statistics.add_best_guesses_to_stats"
        )

    # Get class labels and class index for each ROI
    label_maxprob_list = df_class_labels["best guess"].values
    class_maxprob_list = pd.Categorical(
        df_class_labels["best guess"].values,
        categories=[
            cl
            for cl in df_class_labels.columns
            if cl not in ["best guess", "best guess value"]
        ],
        ordered=True,
    ).codes

    # Set up figure with 15 axes columns
    N = len(roilist)
    ncols = img_per_col = min(N, 15)
    nrows = 1
    if N > img_per_col:
        ncols = img_per_col
        nrows = N // ncols + int((N % img_per_col) > 0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1 * ncols, 1 * nrows))

    # Plot ROIs in an ncols x nrows grid
    colors = sns.color_palette()
    for i, (img, ax) in enumerate(zip(roilist, axes.flatten())):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.setp(ax.spines.values(), color=colors[class_maxprob_list[i]], lw=4)
        ax.text(
            0,
            1,
            f"{class_maxprob_list[i]} {label_maxprob_list[i][:4].upper()}",
            ha="left",
            va="top",
            fontsize=8,
            transform=ax.transAxes,
        )
        if true_class and (label_maxprob_list[i] != true_class):
            ax.text(
                0.5,
                0.5,
                "X",
                ha="center",
                va="center",
                fontsize=20,
                color="red",
                alpha=0.5,
                transform=ax.transAxes,
            )

    # Hide non-used axes in the grid
    for ax in axes.flatten()[len(roilist):]:
        ax.set_visible(False)

    fig.patch.set_linewidth(10)
    fig.patch.set_edgecolor("k")

    return fig, ax


def classify_plot_class_rois(class_name, classifier, filelist):
    """Classify single-object (ROI) images and plot images in a grid with best guess class.

    Parameters
    ----------
    class_name: str
        Name of class ROI files belong to (e.g. 'copepod')
    classifier: pyopia.classify.Classify
        PyOPIA classifier instance
    filelist: list
        List of single-object (ROI) files

    Returns
    -------
    df_: pandas.DataFrame
        Classification results for each image
    """
    # Load single-object images (ROIs)
    roilist = [np.float64(plt.imread(f)) / 255.0 for f in filelist]

    # Classify ROIs
    df_ = classify_rois(roilist, classifier)
    df_ = pyopia.statistics.add_best_guesses_to_stats(df_)

    # Remove "_probability" from labels
    df_ = df_.replace("probability_", "", regex=True)
    df_.columns = df_.columns.str.replace("probability_", "", regex=True)

    # Plot
    fig, ax = plot_classified_rois(roilist, df_, true_class=class_name)

    # Print classification info
    num_correct = (df_["best guess"] == class_name).sum()
    num_images = len(roilist)
    frac_class = num_correct / num_images
    print(
        f"Correctly identified {class_name} was {100 * frac_class:.1f}% ({num_correct}/{num_images})"
    )
    fig.suptitle(
        f"Class: {class_name} ({num_correct}/{num_images}, {100 * frac_class:.1f}%)"
    )

    return df_.style.format(precision=1, decimal=".")
