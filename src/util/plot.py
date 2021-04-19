from sklearn.metrics.pairwise import cosine_similarity
import itertools
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import confusion_matrix


def get_intra(matrix: np.ndarray, labels: pd.Series) -> Tuple[List[List[float]], List[str]]:
    """ Assumes that sample is already ordered by labels. """
    first_idx = 0
    final_idx = 0
    classes = list(labels.unique())
    vc = labels.value_counts()
    result = []
    for c in classes:
        final_idx += vc[c]
        subset = matrix[first_idx:final_idx, first_idx:final_idx]
        subset = np.triu(subset)
        subset = [i for i in subset.flatten().tolist() if i != 0.0]
        result.append(subset)
        first_idx = final_idx
    return result, classes


def get_inter(matrix: np.ndarray, labels: pd.Series) -> Tuple[List[List[float]], List[str]]:
    """ Assumes that sample is already ordered by labels. """
    first_idx = 0
    final_idx = 0
    classes = list(labels.unique())
    vc = labels.value_counts()
    result = []
    for c in classes:
        final_idx += vc[c]
        subset = np.delete(matrix[first_idx:final_idx, :], range(first_idx, final_idx), axis=1)
        result.append(subset.flatten().tolist())
        first_idx = final_idx
    return result, classes


def compute_distance(ref_embeddings):
    '''
    Computes sum of distances between all classes embeddings on our reference test image:
        d(0,1) + d(0,2) + ... + d(0,9) + d(1,2) + d(1,3) + ... d(8,9)
        A good model should have a large distance between all theses embeddings
    Returns:
        array of shape (nb_classes,nb_classes)
    '''
    return cosine_similarity(ref_embeddings)


def plot_distance_boxplot(matrix: np.ndarray, classes: pd.Series, epoch: str, step: str, distance_type: str,
                          figsize: Tuple = (16, 9)) -> Figure:
    if distance_type == "intra":
        data, class_ticks = get_intra(matrix, classes)
    else:
        data, class_ticks = get_inter(matrix, classes)

    fig, ax = plt.subplots(figsize=figsize)
    if step is None:
        ax.set_title(f'Evaluating {distance_type} embeddings distance from each other at epoch {epoch}')
    else:
        ax.set_title(f'Evaluating {distance_type} embeddings distance from each other at epoch {epoch}, step {step} ')
    plt.xlabel('Classes')
    plt.ylabel('Distance')
    ax.boxplot(data, showfliers=False, showbox=True)
    locs, labels = plt.xticks()
    plt.xticks(locs, class_ticks, rotation=90)
    plt.ylim(-0.05, 1.05)
    plt.show()
    plt.close()

    return fig


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + .02,
                point['y'],
                str(point['val']),
                {'color': 'black',
                 'weight': 'light',
                 'size': 6,
                 })


def plot_density(name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, label: str,
                 figsize: Tuple = (12, 5)) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    plot_df = pd.concat([train[label], val[label], test[label]], axis=1)
    plot_df.columns = [f"train (#samples: {len(train)})",
                       f"val (#samples: {len(val)})",
                       f"test (#samples: {len(test)})"]
    plot_df.plot.density(ax=ax)
    plt.ylabel("Relative frequency")
    plt.xlabel(f"{label}")
    plt.title(f"{name}: density of '{label}' over data splits")
    plt.show()
    plt.close()

    return fig


def plot_embeddings(epoch: str, step: str, df: pd.DataFrame, figsize=(30, 17)) -> Figure:
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(
        x="umap-one", y="umap-two",
        hue="label",
        style="truth_matching",
        data=df,
        legend="full",
        ax=ax
    )
    if step is None:
        plt.title(f" 2D umap embeddings - {epoch}")
    else:
        plt.title(f" 2D umap embeddings - {epoch}_{step}")
    label_point(df["umap-one"], df["umap-two"], df['token'], plt.gca())
    plt.close()

    return fig


def create_distance_plots(path: str, df: pd.DataFrame, embeddings: np.ndarray, epoch: str, step: str) -> None:
    distances = compute_distance(embeddings)
    fig_intra = plot_distance_boxplot(distances, df["label"], epoch, step, "intra", figsize=(16, 9))
    if step is None:
        fig_intra.savefig(path + f"intra_class_distances__{epoch}")
    else:
        fig_intra.savefig(path + f"intra_class_distances__{epoch}_{step}")

    fig_inter = plot_distance_boxplot(distances, df["label"], epoch, step, "inter", figsize=(16, 9))
    if step is None:
        fig_inter.savefig(path + f"inter_class_distances__{epoch}")
    else:
        fig_inter.savefig(path + f"inter_class_distances__{epoch}_{step}")


def create_embeddings_plot(path: str, epoch: str, step: str, df: pd.DataFrame) -> None:
    fig = plot_embeddings(epoch, step, df)
    if step is None:
        fig.savefig(path + f"embeddings__{epoch}")
    else:
        fig.savefig(path + f"embeddings__{epoch}_{step}")


def plot_confusion_matrix(cm: np.ndarray,
                          target_names: List,
                          epoch: str,
                          step: str,
                          title: str = 'Confusion matrix',
                          cmap=None,
                          normalize=False,
                          figsize: Tuple = (12, 9)) -> Figure:
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if step is None:
        plt.title(title + f" (#: {np.sum(cm.astype(int))}) - eval of epoch {epoch}")
    else:
        plt.title(title + f" (#: {np.sum(cm.astype(int))}) - train epoch {epoch} step {step}")
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    if cm.shape[0] <= 20:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.close()

    return fig


def create_confusion_matrix(path: str, classes: List[str], epoch: str, step: str, y_true: np.ndarray,
                            y_pred: np.ndarray) -> None:

    cm = confusion_matrix(y_true, y_pred, classes)
    fig = plot_confusion_matrix(cm, classes, epoch, step)
    if step is None:
        fig.savefig(path + f"confusion_matrix_epoch_{epoch}_evaluation.png")
    else:
        fig.savefig(path + f"confusion_matrix_{epoch}_{step}_train.png")
