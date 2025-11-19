import torch
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    print(f" Using device: {DEVICE}")

    data = pd.read_csv("data/creditcard.csv")
    data.head()

    print("\nClass distribution:")
    print(data['Class'].value_counts(normalize=True).rename('proportion'))
    