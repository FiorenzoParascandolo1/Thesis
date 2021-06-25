from src.data_utils.preprocessing_utils import download_data
from src.simulation.simulation import simulation

params = {
    "Dataframe": download_data(ticker="TSLA", period="5min"),
    'Period': 50,
    'Show_every': 50
}


if __name__ == "__main__":
    simulation(params)