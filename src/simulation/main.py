from src.simulation.training import training_loop
import yfinance as yf
import requests

params = {
    # Environment
    'FileName': "EURAUD_M5.csv",
    'EnvType': "stocks-v0",
    'Render': False,
    "Explanations": 10,
    "Pip": 0.000180 / 2,
    "Leverage": False,

    # Environment - Observations
    'Periods': [1, 2, 3, 4],
    'Type': "gadf",
    'Pixels': 30,
    'ManageSymmetries': False,

    # Environment - Wallet
    'WalletFactor': 1000000,
    'BetSizeFactor': 0.34,

    # Policy
    'Architecture': "Vgg",
    'Lr': 1e-4,
    'Epochs': 2,
    'Gamma': 0.99,
    'Lambda': 0.99,
    'LenMemory': 12 * 24,
    'BatchSize': 12,
    'EpsClip': 0.15,
    'ValueLossCoefficient': 0.5,
    'EntropyLossCoefficient': 0.002
    }


def main():
    training_loop(params)


if __name__ == "__main__":
    main()
