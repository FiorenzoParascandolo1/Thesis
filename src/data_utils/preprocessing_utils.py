import pandas as pd
import numpy as np
import torch
from pyts.image import GramianAngularField

MAX_WEEK_DAY = 5
MAX_MONTH_DAY = 31
MAX_MONTH = 12


def period_arc_cos(x):
    """
    Compute the elements required for GADF/GASF transformations

    :param x: time series to be processed.
    :return: the arccos of the rescaled time series.
    """
    return np.arccos(rescaling(x.astype(np.float32)))


def rescaling(x):
    """
    Rescale a time series in [0, 1] range

    :param x: time series to be processed.
    :return: rescaled time series.
    """
    return ((x - max(x)) + (x - min(x))) / (max(x) - min(x) + 1e-5)


def gasf(x):
    """
    The Gramian Angular Field (GAF) imaging is an elegant way to encode time series as images.
    GASF = [cos(θi + θj)]

    :param x: time series to be processed.
    :return: GASF matrix.
    """
    return np.array([[np.cos(i + j) for j in period_arc_cos(x)] for i in period_arc_cos(x)])


def gadf(x):
    """
    The Gramian Angular Field (GAF) imaging is an elegant way to encode time series as images.
    GADF = [sin(θi - θj)]

    :param x: time series to be processed.
    :return: GADF matrix.
    """
    return np.array([[np.sin(i - j) for j in period_arc_cos(x)] for i in period_arc_cos(x)])


def add_features_on_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add time information as columns:
    - WeekDay: [0, 0.2, 0.4, 0.6, 0.8, 1] <-> [Monday, Tuesday, Wednesday, Thursday, Friday]
    - MonthDay: [1 / 31, ..., 1]
    - Month: [1/12, ..., 1]

    :param dataframe: dataframe to be processed.
    :return: processed dataframe.
    """

    dataframe['WeekDay'] = dataframe['Date'].apply(lambda x: (x.weekday() + 1) / MAX_WEEK_DAY)
    dataframe['MonthDay'] = dataframe['Date'].apply(lambda x: x.day / MAX_MONTH_DAY)
    dataframe['Month'] = dataframe['Date'].apply(lambda x: x.month / MAX_MONTH)

    return dataframe


def add_period_return(dataframe: pd.DataFrame,
                      period: int = 1,
                      method: str = "log") -> pd.DataFrame:
    """
    Add the column 'Return' to the input dataframe. It represents.
    Most financial studies involve returns, instead of prices, of assets.
    There are two main advantages of using returns.
    First, for average investors, return of an asset is a complete and scale-free summary of the investment opportunity.
    Second, return series are easier to handle than price series because the former have more attractive statistical
    properties.

    :param dataframe: dataframe to be processed.
    :param period: Period return
    :param method: "linear" or "log"
    :return: processed dataframe.
    """

    if 'Return' not in dataframe.columns:
        dataframe['Return'] = 1
    # Base case
    if period == 0:
        if method == "linear":
            dataframe['Return'] = dataframe['Return'] - 1
        else:
            dataframe['Return'] = np.log(dataframe['Return'])
        return dataframe
    dataframe['Return'] = \
        dataframe['Return'] * dataframe['Close'].shift(periods=period - 1) / dataframe['Close'].shift(periods=period)
    # Recursive call
    return add_period_return(dataframe=dataframe, period=period - 1)


def standardize_dataframe_cols(dataframe: pd.DataFrame,
                               col_names: list = None) -> pd.DataFrame:
    """
    Standardize the specified columns of the input dataframe.

    :param dataframe: dataframe to standardize.
    :param col_names: columns to standardize.
    :return: the standardized dataframe.
    """
    if col_names is None:
        col_names = ["Open", "High", "Low", "Close", "Volume"]

    dataframe[col_names] = (dataframe[col_names] - dataframe[col_names].mean()) / dataframe[col_names].std()

    return dataframe


def normalize_dataframe_cols(dataframe: pd.DataFrame,
                             col_names: list = None) -> pd.DataFrame:
    """
    Standardize the specified columns of the input dataframe.

    :param dataframe: dataframe to normalize.
    :param col_names: columns to normalize.
    :return: the normalized dataframe.
    """
    if col_names is None:
        col_names = ["Open", "High", "Low", "Close", "Volume"]

    dataframe[col_names] = (dataframe[col_names] - dataframe[col_names].min()) / \
                           (dataframe[col_names].max() - dataframe[col_names].min())

    return dataframe


def get_rhombus(h=60, w=60):
    tri_rtc = np.fromfunction(lambda i, j: i >= j, (h // 2, w // 2), dtype=int)
    tri_ltc = np.flip(tri_rtc, axis=1)
    rhombus = np.vstack(
        (np.hstack((tri_ltc, tri_rtc[:, 0:])), np.flip(np.hstack((tri_ltc, tri_rtc[:, 0:])), axis=0)[0:, :]))

    return torch.tensor(rhombus, dtype=torch.float32)


class Rhombus(object):
    def __call__(self, images):
        rhombus = get_rhombus().unsqueeze(dim=0).expand_as(images)
        images = images * rhombus

        return images


class PermuteImages(object):
    def __call__(self, images):
        return images.permute(2, 0, 1)


class StackImages(object):
    def __call__(self, images):
        stacked_images = [torch.cat(images[i], dim=0) for i in range(len(images))]
        up_image = torch.cat([stacked_images[0], stacked_images[1]], dim=1)
        down_image = torch.cat([stacked_images[2], stacked_images[3]], dim=1)
        image = torch.cat([up_image, down_image], dim=2)
        return image


class GADFTransformation(object):
    def __init__(self, periods, pixels):
        self.periods = periods
        self.pixels = pixels
        self.gadf = GramianAngularField(image_size=30, method='difference')

    def __call__(self, images):
        aggregated_images = []

        for period in self.periods:
            series = images[0][-1:0:-period][0:self.pixels]
            images_period = [self.gadf.fit_transform(series[:, j].reshape(1, -1)) for j in range(images[0].shape[1])]
            images_period = [torch.Tensor(image) for image in images_period]

            aggregated_images.append(images_period)

        return aggregated_images
