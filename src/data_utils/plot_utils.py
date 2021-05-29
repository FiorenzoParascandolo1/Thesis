import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Configuration
fig_size = (9, 3)


def plot_dataframe(data: pd.DataFrame,
                   labels: list = None,
                   v_min: float = -1.96,
                   v_max: float = 1.96,
                   auto_close: bool = True,
                   s: int = 4) -> None:
    """
    Standardize the specified columns of the input dataframe.

    :param data: dataframe to plot.
    :param labels: list of indices representing anomalies.
    :param v_min:
    :param v_max:
    :param auto_close:
    :param s:
    :return:
    """
    if auto_close:
        plt.close('all')
    plt.figure(figsize=fig_size)
    plt.imshow(data.T.iloc[:, :],
               aspect='auto',
               cmap='RdBu',
               vmin=v_min,
               vmax=v_max)
    if labels is not None:
        n_col = len(data.columns)
        lvl = - 0.05 * n_col
        plt.scatter(labels.index, np.ones(len(labels)) * lvl,
                    s=s,
                    color=plt.get_cmap('tab10')(labels))
    plt.tight_layout()
    plt.show()
