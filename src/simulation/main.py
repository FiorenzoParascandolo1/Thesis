from src.simulation.training import training_loop
import yfinance as yf
import requests

params = {
    # Environment
    'FileName': "EURUSD_M5.csv",
    'EnvType': "stocks-v0",
    'Render': False,
    "Explanations": 10,
    "Pip": 0.0000062 / 2,
    "Leverage": 1,

    # Environment - Observations
    'Periods': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pixels': 30,
    'ManageSymmetries': True,

    # Environment - Wallet
    'WalletFactor': 1000000,
    'BetSizeFactor': 0.34,

    # Policy
    'Architecture': "LocallyConnected",
    'Lr': 1e-4,
    'Epochs': 4,
    'Gamma': 0.99,
    'Lambda': 0.99,
    'LenMemory': 570,
    'Horizon': 45,
    'UpdateTimestamp': 90,
    'EpsClip': 0.1,
    'ValueLossCoefficient': 0.5,
    'EntropyLossCoefficient': 0.01
    }


def main():
    """
    df_1 = pd.read_csv("IBM.csv")
    df_2 = pd.read_csv("wallet.csv")
    pd.DataFrame({'Close': list(df_1['close'].iloc[(params['WindowSize'] - 1):(df_1.shape[0])]),
                  'Wallet': df_2['0'].iloc[:]}).to_csv("ibm_close_wallet.csv")
    """
    training_loop(params)
    """
    df_1 = pd.read_csv("IBM.csv")
    df_2 = pd.read_csv("report.csv")
    print(len(list(df_1['close'].iloc[(params['WindowSize']):(df_1.shape[0])])))
    print(len(df_2['WalletSeries'].iloc[:]))
    pd.DataFrame({'Close': list(df_1['close'].iloc[(params['WindowSize'] - 1):(df_1.shape[0] - 1)]),
                  'Wallet': df_2['WalletSeries'].iloc[:]}).to_csv("ibm_close_wallet.csv")
    """


if __name__ == "__main__":
    main()
