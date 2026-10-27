"""Kiểm thử Hợp đồng API Mô hình – Dịch vụ (Contract Tests v1.0).

Kiểm tra tính tuân thủ nghiêm ngặt của phản hồi JSON v1.0 từ dịch vụ FastAPI
và giao diện BaseCatClassifier của mô hình.

Chạy kiểm thử:
    python -m unittest api/test_contract.py
    hoặc: pytest api/test_contract.py
"""
import io
import unittest
from PIL import Image

from fastapi.testclient import TestClient
from api.main import app, MAX_BYTES
from api.temp_model import BaseCatClassifier, MockCatClassifier, DEFAULT_BREEDS


class TestAPIContractV1(unittest.TestCase):
    """Bộ kiểm thử hợp đồng chuẩn v1.0 cho CatBreedID API."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Tạo một ảnh RGB mẫu hợp lệ trong bộ nhớ
        img = Image.new("RGB", (256, 256), color=(200, 100, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.valid_jpeg_bytes = buf.getvalue()

    def test_01_health_endpoint_contract(self):
        """Kiểm tra hợp đồng endpoint GET /health."""
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("api_version"), "1.0")
        self.assertIn("is_mock", data)
        self.assertEqual(data.get("num_classes"), 12)
        self.assertIsInstance(data.get("classes"), list)
        self.assertEqual(len(data["classes"]), 12)
        self.assertEqual(data["classes"], DEFAULT_BREEDS)

    def test_02_predict_valid_image_contract(self):
        """Kiểm tra hợp đồng JSON v1.0 khi gửi ảnh hợp lệ."""
        files = {"file": ("test_cat.jpg", self.valid_jpeg_bytes, "image/jpeg")}
        res = self.client.post("/predict", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # 1. api_version
        self.assertEqual(data.get("api_version"), "1.0")

        # 2. status & top_label
        self.assertIn(data.get("status"), ["accepted", "rejected"])
        if data["status"] == "accepted":
            self.assertIsInstance(data.get("top_label"), str)
            self.assertIn(data["top_label"], DEFAULT_BREEDS)
        else:
            self.assertIsNone(data.get("top_label"))

        # 3. top5
        top5 = data.get("top5")
        self.assertIsInstance(top5, list)
        self.assertEqual(len(top5), 5)
        prev_prob = 1.0
        for item in top5:
            self.assertIn("label", item)
            self.assertIn("prob", item)
            self.assertIsInstance(item["label"], str)
            self.assertIsInstance(item["prob"], (int, float))
            self.assertGreaterEqual(item["prob"], 0.0)
            self.assertLessEqual(item["prob"], 1.0)
            self.assertLessEqual(item["prob"], prev_prob + 1e-4)  # Đã sắp xếp giảm dần
            prev_prob = item["prob"]

        # 4. ood (phương pháp chính)
        ood = data.get("ood")
        self.assertIsInstance(ood, dict)
        for key in ["method", "score", "threshold", "accept"]:
            self.assertIn(key, ood)
        self.assertIsInstance(ood["accept"], bool)
        self.assertIsInstance(ood["score"], (int, float))
        self.assertIsInstance(ood["threshold"], (int, float))

        # 5. ood_all (5 phương pháp cho chế độ nâng cao)
        ood_all = data.get("ood_all")
        if ood_all is not None:
            self.assertIsInstance(ood_all, list)
            self.assertEqual(len(ood_all), 5)
            for m in ood_all:
                for k in ["method", "score", "threshold", "accept"]:
                    self.assertIn(k, m)
                self.assertIsInstance(m["accept"], bool)

        # 6. model metadata
        model = data.get("model")
        self.assertIsInstance(model, dict)
        for k in ["backbone", "mode", "seed", "checkpoint_id"]:
            self.assertIn(k, model)

        # 7. latency_ms & heatmap
        self.assertIn("latency_ms", data)
        self.assertIsInstance(data["latency_ms"], int)
        self.assertGreater(data["latency_ms"], 0)
        self.assertIn("heatmap", data)

    def test_03_predict_file_too_large(self):
        """Kiểm tra mã lỗi 413 khi gửi ảnh vượt quá 8 MB."""
        large_bytes = b"0" * (MAX_BYTES + 1024)
        files = {"file": ("large.jpg", large_bytes, "image/jpeg")}
        res = self.client.post("/predict", files=files)
        self.assertEqual(res.status_code, 413)
        self.assertIn("quá lớn", res.json().get("detail", ""))

    def test_04_predict_unsupported_media_type(self):
        """Kiểm tra mã lỗi 415 khi gửi file không phải JPG hoặc PNG."""
        files = {"file": ("document.pdf", b"%PDF-1.4...", "application/pdf")}
        res = self.client.post("/predict", files=files)
        self.assertEqual(res.status_code, 415)
        self.assertIn("Chỉ chấp nhận ảnh JPG hoặc PNG", res.json().get("detail", ""))

    def test_05_predict_corrupted_image(self):
        """Kiểm tra mã lỗi 400 khi nội dung file hỏng không load được qua Pillow."""
        corrupted_bytes = b"NOT_A_VALID_JPEG_HEADER_123456789"
        files = {"file": ("corrupted.jpg", corrupted_bytes, "image/jpeg")}
        res = self.client.post("/predict", files=files)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Không đọc được ảnh", res.json().get("detail", ""))

    def test_06_model_interface_architecture(self):
        """Kiểm tra tính tuân thủ của Lớp giao diện mô hình BaseCatClassifier."""
        model = MockCatClassifier()
        self.assertIsInstance(model, BaseCatClassifier)
        self.assertEqual(len(model.classes), 12)

        # Test predict_raw trả về (logits, features)
        img = Image.new("RGB", (224, 224))
        logits, features = model.predict_raw(img)
        self.assertEqual(logits.shape, (12,))
        self.assertEqual(features.shape, (128,))

        # Test predict trả về đủ trường JSON v1.0
        result = model.predict(img)
        self.assertEqual(result["api_version"], "1.0")
        self.assertIn("ood_all", result)
        self.assertEqual(len(result["ood_all"]), 5)


if __name__ == "__main__":
    unittest.main()
