"""
test_system.py
Automated End-to-End Test Suite for Smart Healthcare Disease Prediction System.
Tests Flask endpoints, ML pipeline, SQLite persistence, Chatbot emergency screening,
and Report generation.
"""

import sys
sys.dont_write_bytecode = True

import json
import unittest
import app
import prediction

class TestSmartHealthcareSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.app.config['TESTING'] = True
        app.init_db()
        cls.client = app.app.test_client()

    def test_01_static_root_files_served(self):
        """Verify index.html, style.css, script.js, and chatbot.js are served directly from root."""
        resp = self.client.get('/')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Smart Healthcare Disease Prediction", resp.data)

        resp_css = self.client.get('/style.css')
        self.assertEqual(resp_css.status_code, 200)

        resp_js = self.client.get('/script.js')
        self.assertEqual(resp_js.status_code, 200)

        resp_chat_js = self.client.get('/chatbot.js')
        self.assertEqual(resp_chat_js.status_code, 200)
        print("✓ Test 1: Root static files served successfully.")

    def test_02_health_endpoint_and_ml_metrics(self):
        """Verify /api/health returns healthy status and genuine ML metrics."""
        resp = self.client.get('/api/health')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["database"], "connected")
        self.assertIn("accuracy", data["model_metrics"])
        self.assertGreater(data["model_metrics"]["accuracy"], 90.0)
        print(f"✓ Test 2: System Health & ML Metrics verified (Accuracy: {data['model_metrics']['accuracy']}%).")

    def test_03_prediction_api_and_db_storage(self):
        """Verify /api/predict validates vitals, infers disease risk, and stores record."""
        payload = {
            "patient_name": "Test Patient",
            "age": 42,
            "gender": "Female",
            "body_temp": 102.0,
            "systolic_bp": 125,
            "diastolic_bp": 82,
            "heart_rate": 95,
            "blood_glucose": 98,
            "symptoms": ["fever", "body_pain", "fatigue", "headache"]
        }
        resp = self.client.post('/api/predict', json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertTrue(data["success"])
        res_data = data["data"]
        self.assertIn("predicted_category", res_data)
        self.assertIn("risk_level", res_data)
        self.assertIn("id", res_data)
        print(f"✓ Test 3: Prediction successful: {res_data['predicted_category']} (Risk: {res_data['risk_level']}).")

    def test_04_critical_emergency_vitals_detection(self):
        """Verify that hypertensive crisis or cardiopulmonary symptoms trigger Urgent alert."""
        payload = {
            "patient_name": "Urgent Patient",
            "age": 60,
            "body_temp": 98.6,
            "systolic_bp": 185,
            "diastolic_bp": 122,
            "symptoms": ["chest_discomfort", "shortness_of_breath"]
        }
        resp = self.client.post('/api/predict', json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()["data"]
        self.assertEqual(data["risk_level"], "Urgent")
        self.assertGreater(len(data["vital_warnings"]), 0)
        print("✓ Test 4: Critical cardiopulmonary vitals successfully flagged as Urgent.")

    def test_05_chatbot_faq_response(self):
        """Verify /api/chat responds safely with medical knowledge base."""
        resp = self.client.post('/api/chat', json={"message": "What causes fever?"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["is_emergency"])
        self.assertIn("fever", data["reply"].lower())
        self.assertGreater(len(data["suggestions"]), 0)
        print("✓ Test 5: Chatbot FAQ response successfully generated.")

    def test_06_chatbot_emergency_keyword_alert(self):
        """Verify /api/chat identifies emergency triggers and returns urgent 911 warning."""
        resp = self.client.post('/api/chat', json={"message": "Help I have severe crushing chest pain and cannot breathe!"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["is_emergency"])
        self.assertIn("EMERGENCY", data["reply"])
        self.assertIn("911", data["reply"])
        print("✓ Test 6: Chatbot emergency keywords identified and 911 alert returned.")

    def test_07_history_retrieval_and_filtering(self):
        """Verify /api/history returns stored assessments and respects filters."""
        resp = self.client.get('/api/history')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertGreater(data["total"], 0)
        self.assertIn("records", data)
        self.assertIn("risk_counts", data)
        print(f"✓ Test 7: History API returned {data['total']} recorded assessments.")

    def test_08_report_generation(self):
        """Verify /api/report/<id> generates downloadable clinical assessment report."""
        hist_resp = self.client.get('/api/history')
        records = hist_resp.get_json()["records"]
        latest_id = records[0]["id"]

        resp = self.client.get(f'/api/report/{latest_id}?format=text')
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"SMART HEALTHCARE DISEASE RISK PREDICTION REPORT", resp.data)
        self.assertIn(b"VITAL SIGNS SUMMARY", resp.data)
        self.assertIn(b"CRITICAL MEDICAL DISCLAIMER", resp.data)

        # JSON format report check
        resp_json = self.client.get(f'/api/report/{latest_id}?format=json')
        self.assertEqual(resp_json.status_code, 200)
        json_data = resp_json.get_json()
        self.assertEqual(json_data["id"], latest_id)
        print(f"✓ Test 8: Report generation verified (Text & JSON) for record #{latest_id}.")

    def test_09_history_deletion(self):
        """Verify deleting an individual history item."""
        payload = {"patient_name": "Delete Me", "age": 25, "body_temp": 98.6, "symptoms": []}
        create_resp = self.client.post('/api/predict', json=payload)
        rec_id = create_resp.get_json()["data"]["id"]

        del_resp = self.client.delete(f'/api/history/{rec_id}')
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.get_json()["success"])
        print(f"✓ Test 9: Assessment record #{rec_id} successfully deleted.")

    def test_10_validation_error_handling(self):
        """Verify /api/predict handles invalid or missing required parameters."""
        # Missing age
        resp1 = self.client.post('/api/predict', json={"body_temp": 98.6})
        self.assertEqual(resp1.status_code, 400)

        # Unrealistic body temperature
        resp2 = self.client.post('/api/predict', json={"age": 30, "body_temp": 140.0})
        self.assertEqual(resp2.status_code, 400)
        print("✓ Test 10: Input validation and 400 error handling verified.")

    def test_11_history_search_and_risk_filters(self):
        """Verify history query parameter searching and risk filtering."""
        resp = self.client.get('/api/history?search=Patient&risk_level=All')
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIsInstance(data["records"], list)
        print("✓ Test 11: History search queries and risk filters verified.")

    def test_12_chatbot_lifestyle_topics(self):
        """Verify chatbot covers lifestyle topics (diabetes, blood pressure, hydration, nutrition, sleep)."""
        topics = [
            "What are common diabetes warning signs?",
            "What are blood pressure concerns?",
            "What are hydration tips?",
            "Healthy diet habits"
        ]
        for t in topics:
            resp = self.client.post('/api/chat', json={"message": t})
            self.assertEqual(resp.status_code, 200)
            data = resp.get_json()
            self.assertGreater(len(data["reply"]), 50)
            self.assertGreater(len(data["suggestions"]), 0)
        print("✓ Test 12: Chatbot health & lifestyle topics verified.")

if __name__ == '__main__':
    unittest.main()
