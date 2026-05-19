# Stock Market Prediction Using Machine Learning

**Rizwan Malik | BSc Computer Science | Birmingham City University**  
**Supervisor: Mohamed Ragab | CMP6200 Final Year Project 2026**

---

## Overview

This project builds and evaluates a machine learning pipeline for predicting stock market closing prices across three publicly available datasets: the **S&P 500**, **Apple Inc. (AAPL)**, and the **NASDAQ Composite**.

Four models are trained and compared head-to-head:

- Linear Regression
- Random Forest
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

Every model is evaluated under **two validation schemes**:

1. A standard 20% chronological holdout split
2. A five-fold walk-forward cross-validation

The central finding is that the evaluation method changes the conclusion entirely. Linear Regression achieved R² of 0.985 on NASDAQ under holdout, but collapsed to a mean R² of −19.05 on SP500 under walk-forward validation. The project demonstrates why single-holdout evaluation is an unreliable basis for comparing models on financial time series.

A **Flask web application** is included, allowing any user to upload their own stock price CSV, run the full model comparison, and view results in a browser — no coding required.

---

## Repository Contents

| File | Description |
|---|---|
| `Stock_Market_App` | Flask web application — run this to use the browser interface |
| `Stock_Market_Notebook.ipynb` | Jupyter Notebook — full analytical pipeline including all models, validation, and results |
| `README.md` | This file |

---

## Key Results

### Holdout Evaluation (20% chronological test set)

| Dataset | Best Model | R² | Worst Model | R² |
|---|---|---|---|---|
| NASDAQ | Linear Regression | 0.985 | Random Forest | 0.775 |
| Apple | Linear Regression | 0.920 | Random Forest | −3.484 |
| SP500 | Linear Regression | 0.678 | Random Forest | −0.279 |

### Walk-Forward Cross-Validation (5-fold, mean R²)

| Dataset | Most Stable Model | Mean R² | Std Dev |
|---|---|---|---|
| SP500 | LSTM | 0.220 | 0.143 |
| Apple | Linear Regression | 0.737 | 0.112 |
| NASDAQ | Linear Regression | 0.729 | 0.423 |

> **Key finding:** Linear Regression appeared to be the best model under holdout but achieved a mean R² of −19.05 on SP500 walk-forward validation, with a standard deviation of 1.94. LSTM was the most temporally stable deep learning model across all conditions.

---

## Flask Web Application

The web app allows you to run the full prediction pipeline through a browser.

### How to run it

**1. Install dependencies**

```bash
pip install flask tensorflow scikit-learn pandas numpy
```

**2. Navigate to the app folder and run**

```bash
python app.py
```

**3. Open your browser and go to**

```
http://localhost:5000
```

**4. Upload a CSV file** containing at least a `Date` column and a `Close` column, configure your settings, and click Run. The app will train all four models and display the results.

### What the app shows you

- Holdout prediction charts for each dataset (actual vs predicted)
- Holdout metrics table: RMSE, MAE, R² for all four models
- Walk-forward cross-validation results with mean and standard deviation
- LSTM and GRU training diagnostics including overfit warnings
- Walk-forward consistency chart showing variance across folds

---

## Jupyter Notebook

The notebook (`Stock_Market_Notebook.ipynb`) contains the full analytical pipeline:

- Data loading and preprocessing for all three datasets
- Sliding window sequence construction (lookback = 20 timesteps)
- Linear Regression and Random Forest baseline training
- Stacked LSTM and GRU model training with EarlyStopping
- Holdout evaluation with RMSE, MAE, and R²
- Five-fold walk-forward cross-validation
- Training diagnostics and overfit detection
- Results visualisation

### How to run the notebook

```bash
pip install jupyter tensorflow scikit-learn pandas numpy matplotlib
jupyter notebook Stock_Market_Notebook.ipynb
```

---

## Technical Details

### Model Architecture (LSTM and GRU)

Both models use an identical stacked architecture to ensure fair comparison:

```
Recurrent Layer 1  →  64 units  (return_sequences=True)
Dropout            →  rate 0.2
Recurrent Layer 2  →  32 units
Dropout            →  rate 0.2
Dense Output       →  1 unit  (linear activation)
Optimiser: Adam  |  Loss: MSE  |  Max epochs: 50  |  EarlyStopping patience: 10
```

### Experiment Configuration

| Parameter | Value |
|---|---|
| Lookback window | 20 timesteps |
| Train / Val / Test split | 70% / 10% / 20% (chronological) |
| Scaler | MinMaxScaler [0,1] — fitted on training data only |
| Walk-forward folds | 5 (expanding window) |
| Batch size | 32 |
| Random seed | 42 |

---

## Requirements

```
Python 3.8+
TensorFlow 2.x
Keras
scikit-learn
pandas
numpy
Flask
matplotlib
```

Install all at once:

```bash
pip install tensorflow scikit-learn pandas numpy flask matplotlib
```

---

## Project Structure

```
rizwxn1.github-io/
│
├── Stock_Market_App          # Flask web application
│   ├── app.py                # Main Flask application
│   ├── templates/            # HTML templates
│   └── uploads/              # Uploaded CSV files (created at runtime)
│
├── Stock_Market_Notebook.ipynb   # Full analytical pipeline
│
└── README.md
```

---

## Limitations

- Models are trained on **univariate closing price data only** — trading volume, technical indicators (RSI, MACD), and news sentiment are not included
- S&P 500 and NASDAQ are positively correlated, limiting cross-market independence
- The Flask app runs locally as a prototype — not deployed to a cloud environment
- Hyperparameters are kept identical across all datasets for comparability, which may disadvantage SP500 specifically

---

## Future Work

- Add multivariate inputs: RSI, MACD, Bollinger Bands, trading volume, news sentiment
- Test Temporal Fusion Transformer for multi-horizon forecasting
- Regime-stratified evaluation across bull, bear, and sideways market windows
- Evaluate directional accuracy and simulated trading profitability
- Cloud deployment with live data feeds and persistent storage

---

## References

- Rouf, N. et al. (2021). Stock Market Prediction Using Machine Learning Techniques: A Decade Survey. *Electronics*, 10(21). doi:10.3390/electronics10212717
- Mintarya, L.N. et al. (2023). Machine learning approaches in stock market prediction: a systematic literature review. *Procedia Computer Science*, 216, pp.96–102.
- Kumar, D., Sarangi, P.K. and Verma, R. (2021). A systematic review of stock market prediction using ML. *Materials Today: Proceedings*, 49(8).
- Parmar, I. et al. (2018). Stock Market Prediction Using Machine Learning. *ICSCCC 2018*.
- Sharma, A., Bhuriya, D. and Singh, U. (2017). Survey of stock market prediction using ML approach. *ICECA 2017*.
- Kaggle (2025). Kaggle: Your home for data science. Available at: https://www.kaggle.com/

---

## Acknowledgements

Thanks to Supervisor Mohamed Ragab for guidance throughout this project.

**Birmingham City University — CMP6200 Individual Honours Project 2026**
