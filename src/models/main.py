from src.models.baseline import Baseline
import torch

if __name__ == "__main__":
    model = Baseline(input_size=50, horizon=50, layers=2)
    x = torch.rand(1, 50)
    print(model(x))