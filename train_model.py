"""Download, clean, train, and save the GharValue AI Linear Regression model."""
from pathlib import Path
import json
import re
import zipfile
import warnings
import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Real Estate Data V21.csv"
ARCHIVE = ROOT / "data" / "real_estate_7_cities.zip"
MODEL = ROOT / "models" / "gharvalue_linear_regression.joblib"
METRICS = ROOT / "models" / "project_summary.json"
URL = "https://www.kaggle.com/api/v1/datasets/download/rakkesharv/real-estate-data-from-7-indian-cities"
SOURCE = "https://www.kaggle.com/datasets/rakkesharv/real-estate-data-from-7-indian-cities"

def download_dataset():
    DATA.parent.mkdir(exist_ok=True)
    if DATA.exists() and DATA.stat().st_size > 1000:
        return
    print("Downloading the public Indian real-estate listings dataset...")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    ARCHIVE.write_bytes(response.content)
    with zipfile.ZipFile(ARCHIVE) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as source_file, DATA.open("wb") as destination:
            destination.write(source_file.read())

def first_number(value):
    """Use the first number in ranges such as '2100 - 2850' sq ft."""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan

def price_to_inr(value):
    """Convert source prices marked in INR lakh or crore to rupees once."""
    text = str(value).replace(",", "").lower()
    number = first_number(text)
    if pd.isna(number):
        return np.nan
    if "cr" in text or "crore" in text:
        return number * 10_000_000
    if "l" in text or "lakh" in text:
        return number * 100_000
    return np.nan

def source_city(location):
    text = str(location).lower()
    if "mumbai" in text:
        return "Mumbai"
    if "pune" in text:
        return "Pune"
    if "bangalore" in text or "bengaluru" in text:
        return "Bengaluru"
    return None

def locality_from_location(location, city):
    parts = [part.strip() for part in str(location).split(",") if part.strip()]
    kept = [part for part in parts if part.lower() not in {city.lower(), "bangalore", "bengaluru"}]
    return ", ".join(kept) if kept else str(location).strip()

def clean_data(raw):
    original_rows = len(raw)
    frame = raw[["Property Title", "Price", "Location", "Total_Area", "Baths"]].copy()
    frame["city"] = frame["Location"].map(source_city)
    frame = frame[frame.city.notna()].copy()
    frame["locality"] = [locality_from_location(location, city) for location, city in zip(frame["Location"], frame.city)]
    # City is selected in the UI to filter localities. The model's location feature is city + locality.
    frame["location"] = frame.city + " | " + frame.locality
    frame["bhk"] = frame["Property Title"].map(first_number)
    frame["area_sqft"] = frame["Total_Area"].map(first_number)
    frame["bathrooms"] = pd.to_numeric(frame["Baths"], errors="coerce")
    frame["price_inr"] = frame["Price"].map(price_to_inr)
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates()
    frame = frame.dropna(subset=["location", "bhk", "area_sqft", "bathrooms", "price_inr"])
    # Basic plausibility cleaning; no values are invented or imputed for training rows.
    frame = frame[(frame.area_sqft >= 300) & (frame.area_sqft <= 20000) &
                  (frame.bhk.between(1, 10)) & (frame.bathrooms.between(1, 12)) &
                  (frame.price_inr > 0)]
    frame = frame[["city", "locality", "location", "area_sqft", "bhk", "bathrooms", "price_inr"]]
    return frame, {"original_rows": original_rows, "duplicate_rows_removed": duplicates,
                   "rows_after_cleaning": len(frame), "missing_values_raw": int(raw.isna().sum().sum())}

def gradient_descent_summary(frame):
    """One-feature area/price example solely for the gradient-descent practical."""
    sample = frame[["area_sqft", "price_inr"]].sample(min(1000, len(frame)), random_state=42)
    x = sample.area_sqft.to_numpy(dtype=float)
    y = sample.price_inr.to_numpy(dtype=float)
    x = (x - x.mean()) / x.std()
    y = y / 100000.0  # lakhs during numerical optimization only
    weight, bias, learning_rate = 0.0, 0.0, 0.03
    costs = []
    for _ in range(200):
        predicted = weight * x + bias
        error = predicted - y
        costs.append(float(np.mean(error ** 2) / 2))
        weight -= learning_rate * float(np.mean(error * x))
        bias -= learning_rate * float(np.mean(error))
    points = [{"iteration": i + 1, "cost": round(c, 4)} for i, c in enumerate(costs) if i % 5 == 0 or i == 199]
    return {"initial_cost": round(costs[0], 4), "final_cost": round(costs[-1], 4), "points": points}

def practical_summaries(frame):
    """Small educational demonstrations; none are used to predict a price."""
    subset = frame.sample(min(3000, len(frame)), random_state=42).copy()
    features = subset[["area_sqft", "bhk", "bathrooms"]]
    target = (subset.price_inr >= subset.price_inr.median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=.2, random_state=42, stratify=target)
    results = {}
    for name, estimator in [("Logistic Regression", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500))])),
                            ("Decision Tree Classification", DecisionTreeClassifier(max_depth=5, random_state=42)),
                            ("KNN Classification", Pipeline([("scale", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=7))]))]:
        estimator.fit(X_train, y_train)
        results[name] = {"task": "Classifies whether price is at/above this sample's median", "accuracy": round(float(accuracy_score(y_test, estimator.predict(X_test))), 3)}
    return results

def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    download_dataset()
    raw = pd.read_csv(DATA)
    frame, cleaning = clean_data(raw)
    X = frame[["location", "area_sqft", "bhk", "bathrooms"]]
    y = frame.price_inr
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    preprocessor = ColumnTransformer([
        ("location", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), ["location"]),
        ("numeric", StandardScaler(), ["area_sqft", "bhk", "bathrooms"]),
    ])
    model = Pipeline([("preprocessor", preprocessor), ("linear_regression", LinearRegression())])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {"mse": float(mean_squared_error(y_test, predictions)), "mae": float(mean_absolute_error(y_test, predictions)),
               "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))), "r2": float(r2_score(y_test, predictions))}
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL)
    cities = {city: sorted(group.locality.dropna().unique().tolist()) for city, group in frame.groupby("city")}
    summary = {"dataset": {"name": "Real Estate Data from 7 Indian Cities", "source": SOURCE, "license": "Kaggle dataset terms", "price_unit": "INR lakh/crore labels", "download_mirror": URL},
               "cleaning": cleaning, "metrics": metrics, "locations": sorted(frame.location.unique().tolist()),
               "cities": cities,
               # Linear Regression can extrapolate below zero. These floors are derived
               # from the lowest 1% of genuine listings in each supported city.
               "city_price_floors": {city: float(group.price_inr.quantile(.01)) for city, group in frame.groupby("city")},
               "ranges": {c: [float(frame[c].min()), float(frame[c].max())] for c in ["area_sqft", "bhk", "bathrooms"]},
               "gradient_descent": gradient_descent_summary(frame), "practicals": practical_summaries(frame)}
    METRICS.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Trained Linear Regression on {len(frame):,} cleaned records.")
    # Keep console output compatible with Windows terminals that use cp1252.
    print("R2:", round(metrics["r2"], 3), "MAE (INR): {:,.0f}".format(metrics["mae"]))

if __name__ == "__main__":
    main()
