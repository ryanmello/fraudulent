# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions using multiple classification models with hyperparameter tuning and class imbalance handling.

## Overview

Credit card fraud is a significant problem with highly imbalanced data—fraudulent transactions represent only ~0.17% of all transactions. This project implements a complete ML pipeline that:

- Compares three classification models (Logistic Regression, Random Forest, XGBoost)
- Handles class imbalance using SMOTE (Synthetic Minority Oversampling Technique)
- Performs hyperparameter tuning with RandomizedSearchCV
- Optimizes classification thresholds for maximum F1 score
- Provides comprehensive evaluation metrics and visualizations

## Models & Methodology

### Models
| Model | Description |
|-------|-------------|
| **Logistic Regression** | Baseline linear classifier |
| **Random Forest** | Ensemble of decision trees with hyperparameter tuning |
| **XGBoost** | Gradient boosting with GPU acceleration and hyperparameter tuning |

### Handling Class Imbalance
The dataset is highly imbalanced (99.83% legitimate, 0.17% fraud). We use **SMOTE** to synthetically oversample the minority class during training, integrated into sklearn pipelines to prevent data leakage.

### Evaluation Metrics
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve (more informative for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Classification Report**: Precision, recall, F1 per class

### Visualizations
- Class distribution plot
- Threshold vs F1 score curves
- Feature importance charts
- ROC and Precision-Recall curves
- Confusion matrices

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset:**
   
   Set up Kaggle API credentials:
   - Go to https://www.kaggle.com → Settings → API → Create New Token
   - Move `kaggle.json` to `~/.kaggle/kaggle.json` (or `C:\Users\<Username>\.kaggle\` on Windows)
   
   Then download:
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
   ```

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

| Property | Value |
|----------|-------|
| Transactions | 284,807 |
| Features | 30 (Time, Amount, V1-V28 PCA-transformed) |
| Legitimate (Class 0) | 99.83% |
| Fraudulent (Class 1) | 0.17% |

The V1-V28 features are the result of PCA transformation (original features not provided for confidentiality).

## Project Structure

```
fraudulent/
├── data/
│   └── creditcard.csv    # Dataset (download via Kaggle)
├── fraudulent.py         # Main ML pipeline
├── requirements.txt      # Python dependencies
└── README.md
```

## Running

```bash
python fraudulent.py
```

The script will:
1. Load and display dataset statistics
2. Split data into train/test sets (80/20)
3. Scale the `Amount` and `Time` features
4. Train and tune all three models
5. Evaluate models and find optimal classification thresholds
6. Generate all visualizations
7. Print a summary comparison table

## Configuration

Key parameters can be adjusted at the top of `fraudulent.py`:

```python
RANDOM_STATE = 42      # Reproducibility seed
TEST_SIZE = 0.2        # Test set proportion
SMOTE_RATIO = 1.0      # Target minority/majority ratio after SMOTE
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for XGBoost acceleration)

See `requirements.txt` for full dependency list.
