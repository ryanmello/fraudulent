import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "main":
    print(f" Using device: {DEVICE}")
