"""Smoke tests for the Flask prediction interface."""

import unittest

from Flask.app import app


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page_loads(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"NBA Player Stat Predictor", response.data)

    def test_prediction_requires_matchup_fields(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
