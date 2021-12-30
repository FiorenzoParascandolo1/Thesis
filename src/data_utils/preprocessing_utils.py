import pandas as pd
import numpy as np
import torch
from pyts.image import GramianAngularField
from datetime import datetime
import matplotlib.pyplot as plt

MAX_WEEK_DAY = 5
MAX_MONTH_DAY = 31
MAX_MONTH = 12
MAX_HOUR = 23
MAX_MINUTE = 55


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

    time_series = dataframe['time'].apply(datetime.fromisoformat)
    dataframe['WeekDay'] = time_series.apply(lambda x: (x.weekday() + 1) / MAX_WEEK_DAY)
    dataframe['MonthDay'] = time_series.apply(lambda x: x.day / MAX_MONTH_DAY)
    dataframe['Month'] = time_series.apply(lambda x: x.month / MAX_MONTH)
    dataframe['Hour'] = time_series.apply(lambda x: x.hour / MAX_HOUR)
    dataframe['Minute'] = time_series.apply(lambda x: x.minute / MAX_MINUTE)

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Add time-feature
    df = add_features_on_time(df)
    df = df[df['volume'] != 0]

    # Select 9:30 <= hours:minute <= 16.00

    """
    df = df[df['Hour'] * MAX_HOUR < 17]
    df = df[df['Hour'] * MAX_HOUR > 9]
    df['Select_1'] = df['Hour'] * MAX_HOUR < 16
    df['Select_2'] = (df['Minute'] * MAX_MINUTE < 5) & (df['Hour'] * MAX_HOUR >= 16)
    df['Select_3'] = df['Select_1'] | df['Select_2']
    df = df[df['Select_3']]
    df['Select_1'] = df['Hour'] * MAX_HOUR >= 10
    df['Select_2'] = (df['Minute'] * MAX_MINUTE >= 30) & (df['Hour'] * MAX_HOUR >= 9)
    df['Select_3'] = df['Select_1'] | df['Select_2']
    df = df[df['Select_3']]
    """

    return df


def get_rhombus(h=60, w=60) -> torch.Tensor:
    """
    Returns a rhombus to get off symmetrical pixels in GAF images.

    :param h: height of the rhombus image.
    :param w: width of the rhombus image.
    :return: rhombus image.
    """
    tri_rtc = np.fromfunction(lambda i, j: i >= j, (h // 2, w // 2), dtype=int)
    tri_ltc = np.flip(tri_rtc, axis=1)
    rhombus = np.vstack((np.hstack((tri_ltc, tri_rtc[:, 0:])),
                         np.flip(np.hstack((tri_ltc, tri_rtc[:, 0:])), axis=0)[0:, :]))

    return torch.tensor(rhombus, dtype=torch.float32)


class Rhombus(object):
    """
    Application-based transformation (multiplication) of each input image by a diamond that can be added in the
    torch-transform pipeline.
    """

    def __call__(self, images):
        """
        :param images: GAF image.
        :return: GAF image rhombus-transformed.
        """

        rhombus = get_rhombus().unsqueeze(dim=0).expand_as(images)
        images = images * rhombus

        return images


class PermuteImages(object):
    """
    Permute tensor images according to PyTorch convention.
    """

    def __call__(self, images: torch.tensor) -> torch.Tensor:
        """
        :param images: GAF image.
        :return: transposed GAF image.
        """
        return images.permute(2, 0, 1)


class StackImages(object):
    """
    Stack sub-images to obtain the final composed image described in:
    https://www.iris.unina.it/retrieve/handle/11588/807057/337910/IEEE_CAA_Journal_of_Automatica_Sinica-3.pdf
    """

    def __call__(self, images: list) -> torch.Tensor:
        """
        :param images: list of GAF images.
        :return: final stacked GAF image.
        """

        # Concatenate the first two sub-images along the columns
        up_image = torch.cat([images[0], images[1]], dim=1)
        # Concatenate the other two sub-images along the columns
        down_image = torch.cat([images[2], images[3]], dim=1)
        # Concatenate 'up' and 'down' images along the rows
        image = torch.cat([up_image, down_image], dim=2)

        return image


class ManageSymmetries(object):
    """
    Solves symmetry of GAF images by representing two periods per image
    """

    def __init__(self,
                 pixels):
        """
        :param pixels: number of pixels per image
        :return:
        """

        """
        Example symmetry_odd for pixels = 5
        1 0 0 0 0 
        1 1 0 0 0 
        1 1 1 0 0 
        1 1 1 1 0 
        1 1 1 1 1
        """
        self.symmetry_odd = torch.tensor(np.concatenate([np.expand_dims(np.fromfunction(lambda i, j: i >= j,
                                                                                        (pixels, pixels),
                                                                                        dtype=int), axis=0)] * 5,
                                                        axis=0)).float()
        """
        Example symmetry_even for pixels = 5
        0 1 1 1 1
        0 0 1 1 1 
        0 0 0 1 1 
        0 0 0 0 1 
        0 0 0 0 0 
        """
        self.symmetry_even = torch.tensor(np.concatenate([np.expand_dims(np.fromfunction(lambda i, j: i < j,
                                                                                         (pixels, pixels), dtype=int),
                                                                         axis=0)] * 5, axis=0)).float()

    def __call__(self,
                 images: list) -> list:
        """
        :param images: list of GAF images.
        :return: list of images preprocessed considering symmetries
        """
        """
        plt.imshow(images[0][2:5, :, :].detach().permute(1, 2, 0).numpy())
        plt.show()
        plt.imshow(self.symmetry_odd[2:5, :, :].detach().permute(1, 2, 0).numpy())
        plt.show()

        plt.imshow(images[1][2:5, :, :].detach().permute(1, 2, 0).numpy())
        plt.show()
        plt.imshow(self.symmetry_even[2:5, :, :].detach().permute(1, 2, 0).numpy())
        plt.show()

        plt.imshow((images[0] * self.symmetry_odd + images[0 + 1] * self.symmetry_even)[2:5, :, :].detach().permute(1, 2, 0).numpy())
        plt.show()
        """
        return [images[i] * self.symmetry_odd + images[i + 1] * self.symmetry_even
                for i in range(0, len(images) - 1, 2)]


class GADFTransformation(object):
    """
    A Gramian angular field is an image obtained from a time series, representing some kind of temporal correlation
    between each pair of values from the time series.
    """

    def __init__(self,
                 periods: list,
                 pixels: int):
        """
        :param periods: list of periods to consider for slicing the time-series.
        :param pixels: number of elements in the sliced time-series to pass to the GAF transformation
        :return:
        """
        self.periods = periods
        self.pixels = pixels
        self.gadf = GramianAngularField(image_size=pixels, method='difference')

    def __call__(self,
                 series: list) -> list:
        """
        :param images: dataframe containing OLHCV features
        :return: list of GAF images for each period
        """
        aggregated_images = []

        # For each period the list of each GAF image (created for each feature) is appended to aggregated_images
        for i in range(len(series)):
            # Apply GADF transformation to each sub-time series for each feature
            images_period = [torch.Tensor(self.gadf.fit_transform(np.array(series[i].iloc[:, j]).reshape(1, -1)))
                             for j in range(series[i].shape[1])]
            aggregated_images.append(images_period)

        # Given n periods, return a list of n-elements. Each element is a list of GADF image
        return [torch.cat(aggregated_images[i], dim=0) for i in range(len(aggregated_images))]


class ManagePeriods(object):
    def __init__(self,
                 periods: list,
                 pixels: int):
        """
        :param periods: list of periods to consider for slicing the time-series.
        :param pixels: number of elements in the sliced time-series to pass to the GAF transformation
        :return:
        """
        self.periods = periods
        self.pixels = pixels

    def manage_periods(self,
                       series: pd.Series,
                       period: int) -> pd.DataFrame:
        """
        :param series: time-series observation.
        :param period: period is used to aggregate time series information in order to create a less granular one
        :return: the new time-series created
        """
        """
        if period == 1:
            return pd.DataFrame(series[-1:0:-period][0:self.pixels]).iloc[::-1]
        """
        close = series[:, 5][-1:0:-period][0:self.pixels].astype(np.float64)
        date = series[:, 0][-1:0:-period][0:self.pixels]
        open = series[:, 1][-period:0:-period][0:self.pixels].astype(np.float64)
        high = pd.Series([max(series[:, 2][(-period * (i + 1)):(-period * i if i != 0 else None)])
                          for i in range(self.pixels)])
        low = pd.Series([min(series[:, 3][(-period * (i + 1)):(-period * i if i != 0 else None)])
                         for i in range(self.pixels)])
        volume = pd.Series([sum(series[:, 4][(-period * (i + 1)):(-period * i if i != 0 else None)])
                            for i in range(self.pixels)])

        df = pd.DataFrame({'Date': date, 'Open': open, 'High': high, 'Low': low, 'Volume': volume,
                           'Close': close}).iloc[::-1]
        df['Date'] = df['Date'].apply(lambda x: datetime.fromisoformat(x))
        return df.set_index('Date')

    def __call__(self,
                 series: pd.Series) -> list:
        """
        :param series: dataframe containing OLHCV features
        :return: list of GAF images for each period
        """

        return [self.manage_periods(series=series, period=period) for period in self.periods]


class IdentityTransformation(object):
    """
    """

    def __init__(self):
        pass

    def __call__(self, x):
        """
        """
        return x
