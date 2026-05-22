import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Required for Flask
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)

# Standardize feature names
FEATURES = ["prev_Open", "prev_Close", "Open"]
TARGET = "Close"

# Global state to hold data and charts
STATE = {"df": None, "trained": None, "results": None, "companies": [], "charts": {}}

def _clean_and_pipe(df):
    """Robust pipeline that handles your specific CSV format."""
    # 1. Column Standardization
    df.columns = df.columns.str.strip()
    if "Date" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "Date"})
    
    # Identify company column (your CSV has 'AAPL' in a column named 'company')
    if "company" not in df.columns:
        # Fallback: find the first column with text/strings
        cols = df.select_dtypes(include=['object']).columns
        if len(cols) > 0: df = df.rename(columns={cols[0]: "company"})
    
    # 2. Basic Cleaning
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "company"])
    df = df.sort_values(["company", "Date"])
    
    # 3. Feature Engineering (Crucial for the Predictor)
    for col in ["Open", "Close"]:
        df[f"prev_{col}"] = df.groupby("company")[col].shift(1)
    
    df = df.dropna(subset=FEATURES + [TARGET])
    return df

def make_charts(df, trained, results, X_test, y_test):
    """Generates charts and converts to Base64 for the HTML."""
    charts = {}
    sns.set_style("dark")
    plt.rcParams.update({'text.color': "white", 'axes.labelcolor': "white", 'xtick.color': "white", 'ytick.color': "white"})

    # 1. Performance Bar Chart
    try:
        rdf = pd.DataFrame(results).T
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        rdf[["MSE (Test)", "MAE (Test)"]].plot(kind='bar', ax=ax, color=["#38bdf8", "#818cf8"])
        plt.xticks(rotation=0)
        charts["perf"] = _fig_to_b64(fig)
    except Exception as e: print(f"Chart Error (Perf): {e}")

    # 2. Prediction Scatter
    try:
        y_pred = trained["Random Forest"].predict(X_test)
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#1e293b")
        ax.set_facecolor("#1e293b")
        ax.scatter(y_test, y_pred, alpha=0.3, color="#38bdf8")
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--", color="#f472b6")
        charts["pred"] = _fig_to_b64(fig)
    except Exception as e: print(f"Chart Error (Pred): {e}")

    # 3. Heatmap
    try:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor="#1e293b")
        # Only correlate the columns we actually have
        corr_cols = [c for c in ["Open", "Close", "High", "Low", "prev_Open", "prev_Close"] if c in df.columns]
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        charts["corr"] = _fig_to_b64(fig)
    except Exception as e: print(f"Chart Error (Corr): {e}")

    return charts

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _train(df):
    X, y = df[FEATURES], df[TARGET]
    idx = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]
    
    models = {
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }
    
    trained, results = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "MSE (Test)": round(mean_squared_error(y_test, pred), 4),
            "MAE (Test)": round(mean_absolute_error(y_test, pred), 4),
            "R² Score": round(r2_score(y_test, pred), 4)
        }
        trained[name] = model
    
    charts = make_charts(df, trained, results, X_test, y_test)
    return trained, results, charts

@app.route("/")
def index():
    return render_template("index.html", **STATE, loaded=(STATE["df"] is not None))

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f: return jsonify({"error": "No file"}), 400
    
    raw = pd.read_csv(f)
    df = _clean_and_pipe(raw)
    trained, results, charts = _train(df)
    
    STATE.update({
        "df": df, "trained": trained, "results": results, "charts": charts,
        "companies": sorted(df["company"].unique().tolist())
    })
    return jsonify({"ok": True})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model = STATE["trained"].get(data["model"])
    X_val = pd.DataFrame([[float(data["prev_open"]), float(data["prev_close"]), float(data["open"])]], columns=FEATURES)
    pred = model.predict(X_val)[0]
    return jsonify({"predicted_close": round(float(pred), 2), "model": data["model"]})

if __name__ == "__main__":
    app.run(debug=True, port=5050)