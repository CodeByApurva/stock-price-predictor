# 📈 Stock Price Predictor

A machine-learning powered stock price forecasting tool with an interactive Flask web dashboard. Built as the final project for the **Python for Data Science (PDS)** course.

The project predicts stock **closing prices** using three regression models — **Decision Tree**, **Random Forest**, and **XGBoost** — trained on historical data from **AAPL, AMZN, GOOGL, and MSFT**.

---

## ✨ Features

- **End-to-end ML Pipeline** — data cleaning, missing value handling, outlier capping (IQR), one-hot encoding, and feature engineering all implemented from scratch
- **Three Trained Models** — Decision Tree, Random Forest, and XGBoost regressors with performance comparison
- **Interactive Web Dashboard** — dark-themed Flask UI to upload data, view metrics, make predictions, and explore visualizations
- **Drag & Drop CSV Upload** — upload your dataset and the pipeline runs automatically
- **Live Predictions** — enter previous day's Open/Close + today's Open to get an instant Close price prediction
- **Auto-generated Charts** — performance bar charts, actual vs. predicted scatter plots, and correlation heatmaps

---

## 🛠️ Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Language** | Python 3 |
| **ML / Data** | scikit-learn · XGBoost · Pandas · NumPy |
| **Visualization** | Matplotlib · Seaborn |
| **Web Framework** | Flask · Jinja2 |
| **Frontend** | HTML5 · CSS3 · JavaScript |
| **Development** | Jupyter Notebook · Google Colab |

---

## 📁 Project Structure

```
stock-price-predictor/
├── app.py                          # Flask backend – ML pipeline + routes
├── templates/
│   └── index.html                  # Frontend dashboard (dark-themed UI)
├── static/                         # Static assets (CSS/JS/images)
├── unclean_pds.csv                 # Raw dataset (~28k rows, 4 companies)
├── final-project-devansh-2023800091.ipynb  # Original Jupyter notebook
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+ installed
- `pip` package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/stock-price-predictor.git
   cd stock-price-predictor
   ```

2. **Install dependencies**

   ```bash
   pip install flask pandas numpy scikit-learn xgboost matplotlib seaborn
   ```

3. **Run the app**

   ```bash
   python app.py
   ```

4. **Open the dashboard** — navigate to [http://127.0.0.1:5050](http://127.0.0.1:5050) in your browser

5. **Upload data** — drag & drop `unclean_pds.csv` into the upload zone and the models will train automatically

---

## 📊 How It Works

### Data Pipeline

The raw CSV (`unclean_pds.csv`) contains ~28,000 rows of historical stock data for **AAPL, AMZN, GOOGL, and MSFT** with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, and `company`.

The pipeline performs the following steps:

1. **Data Cleaning** — remove duplicates, parse dates, filter out negative prices
2. **Missing Value Handling** — forward-fill Open from previous Close, back-fill Close from next Open
3. **Outlier Treatment** — cap extreme values using the IQR method (1.5× IQR bounds)
4. **Encoding** — one-hot encode the `company` column
5. **Feature Engineering** — create lagged features (`prev_Open`, `prev_Close`) by shifting values within each company group

### Model Training

Three regression models are trained on an 80/20 chronological split (no data leakage):

| Model | Description |
|-------|-------------|
| **Decision Tree** | Single tree regressor (`max_depth=None`) |
| **Random Forest** | Ensemble of 100 trees |
| **XGBoost** | Gradient-boosted trees (100 estimators) |

**Input Features:** `prev_Open`, `prev_Close`, `Open`
**Target Variable:** `Close`

### Prediction

Given three inputs — the previous day's Open price, previous day's Close price, and today's Open price — the selected model predicts today's **closing price**.

---

## 📈 Model Performance

Results from the test set evaluation:

| Model | MSE (Test) | MAE (Test) | R² Score |
|-------|-----------|-----------|----------|
| Decision Tree | 808.31 | 13.91 | 0.9537 |
| Random Forest | 811.65 | 13.80 | 0.9536 |
| XGBoost | 1331.75 | 18.09 | 0.9238 |

> All three models achieve **R² > 0.92**, with Decision Tree and Random Forest performing nearly identically at **~95% R²**.

---

## 🖥️ Web Dashboard

The Flask frontend provides four sections:

1. **Load Dataset** — drag-and-drop CSV upload that triggers the full ML pipeline
2. **Model Performance** — metric cards displaying MSE, MAE, and R² for all three models
3. **Make a Prediction** — input form to enter stock prices and get an instant Close prediction
4. **Visualizations** — auto-generated charts including performance comparison, actual vs. predicted scatter, and feature correlation heatmap

---

## 📓 Jupyter Notebook

The original analysis    includes:

- Detailed data cleaning with before/after visualizations
- Missing value heatmaps
- Outlier detection box plots
- One-hot encoding analysis
- Standardization histograms
- Feature correlation heatmaps
- Univariate, bivariate, and multivariate analysis
- Model training, validation (TimeSeriesSplit cross-validation), and hyperparameter tuning
- Overfitting vs. underfitting analysis (Train R² vs. Test R²)

---

## 🔮 Future Improvements

- [ ] Add LSTM / neural network models for time-series forecasting
- [ ] Integrate real-time stock data via APIs (e.g., Yahoo Finance)
- [ ] Deploy to a cloud platform (Heroku / Render / AWS)
- [ ] Add more technical indicators as features (RSI, MACD, Bollinger Bands)
- [ ] Implement user authentication and saved prediction history

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Python for Data Science (PDS)** course for the project framework
- [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/) for model implementations
- [Flask](https://flask.palletsprojects.com/) for the web framework
- Historical stock data sourced from Yahoo Finance
