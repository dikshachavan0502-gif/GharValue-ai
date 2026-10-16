import unittest
import json
from app import app

class AppTests(unittest.TestCase):
    def setUp(self): self.client = app.test_client()
    def test_home(self): self.assertEqual(self.client.get('/').status_code, 200)
    def test_invalid_prediction(self): self.assertEqual(self.client.post('/predict', json={}).status_code, 400)
    def test_valid_prediction(self):
        with open("models/project_summary.json", encoding="utf-8") as summary_file:
            summary = json.load(summary_file)
        city = next(iter(summary["cities"]))
        response = self.client.post("/predict", json={
            "city": city,
            "locality": summary["cities"][city][0],
            "area_sqft": 1200,
            "bhk": 2,
            "bathrooms": 2,
        })
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIn("prediction", body)
        self.assertGreater(body["prediction"], 0)

    def test_streamlit_prediction_uses_saved_model(self):
        from app_streamlit import load_assets, predict_price

        model, summary = load_assets()
        city = next(iter(summary["cities"]))
        value = predict_price(model, summary, city, summary["cities"][city][0], 1200, 2, 2)
        self.assertGreater(value, 0)

if __name__ == '__main__': unittest.main()
