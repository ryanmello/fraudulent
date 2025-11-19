# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions.

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API credentials:**
   - Go to https://www.kaggle.com and log in
   - Navigate to Settings → API → Create New Token
   - Move the downloaded `kaggle.json` to `C:\Users\<YourUsername>\.kaggle\kaggle.json` (Windows)

4. **Download the dataset:**
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
   ```

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

- **Size**: 284,807 transactions
- **Features**: 30 (Time, Amount, V1-V28 anonymized features)
- **Class distribution**: 
  - Legitimate (0): 99.83%
  - Fraudulent (1): 0.17%

## Running

```bash
python fraudlent.py
```

