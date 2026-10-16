# GharValue AI — Indian House Price Prediction

Single-page Flask college project using **only Linear Regression** for price prediction. The accompanying compact demonstrations cover preprocessing, Gradient Descent, Logistic Regression, Decision Tree Classification, and KNN Classification.

## Dataset

The app downloads [Real Estate Data from 7 Indian Cities](https://www.kaggle.com/datasets/rakkesharv/real-estate-data-from-7-indian-cities) through its public Kaggle download endpoint. It supplies genuine listings for Bengaluru, Mumbai and Pune with location, price, area, BHK (from property title), and bathrooms. Price labels in lakhs/crores are converted exactly once to INR. The downloaded file is saved in `data/`.


## Run the Flask version on Windows

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5000` in a browser. Training is a one-time operation; predictions load the saved `models/gharvalue_linear_regression.joblib` file.

## What is cleaned

The script detects duplicate and missing values, extracts BHK from `size`, takes the first numeric value of area ranges, removes incomplete and plainly implausible records, one-hot encodes location, and scales numeric inputs. It does not invent training rows. The website displays metrics calculated on the held-out test set.

## Verification

After training, run `python -m unittest discover -s tests`. The tests check that the single page loads and invalid requests are rejected. Use only the fields present in the form; all four correspond to selected dataset features.

Predictions are educational estimates for Bengaluru, Mumbai, and Pune and are not professional valuations.

## Run locally with Streamlit

The Streamlit deployment app is [app_streamlit.py](app_streamlit.py). It loads the
already-trained `models/gharvalue_linear_regression.joblib` artifact and does not
retrain or download data when it starts.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_streamlit.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`).

## Deploy to Streamlit Community Cloud

1. Create a new GitHub repository and push this entire project, including the
   `models/` directory, `data/` directory, `requirements.txt`, and `.streamlit/config.toml`.
   Do not add `models/gharvalue_linear_regression.joblib` to `.gitignore`.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/), sign in with GitHub,
   and select **Create app**.
3. Select the repository and branch, set the main file path to `app_streamlit.py`, and
   click **Deploy**.
4. If deployment reports a missing artifact, confirm both files in `models/` were
   committed and pushed. The model is loaded from that directory at startup; training
   is deliberately not run in the cloud.
