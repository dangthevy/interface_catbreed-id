"""Mô hình tạm và Giao diện lập trình Mô hình (Tuần 3 - API v1.0).

Định nghĩa lớp cơ sở BaseCatClassifier làm hợp đồng giữa Mô hình và Dịch vụ.
Khi nhóm bàn giao checkpoint thật ở Tuần 12, chỉ cần tạo lớp kế thừa
BaseCatClassifier và thay thế thể hiện (instance) trong main.py.
"""
from abc import ABC, abstractmethod
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# 12 giống mèo Oxford-IIIT Pet Dataset theo thứ tự cố định
DEFAULT_BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
]


class BaseCatClassifier(ABC):
    """Giao diện trừu tượng (Interface Contract) giữa Mô hình và Dịch vụ API.
    
    Tất cả các mô hình triển khai (mô hình tạm, mô hình MobileNetV2, EfficientNet-B0)
    đều phải tuân thủ nghiêm ngặt giao diện này.
    """

    def __init__(self, classes: Optional[List[str]] = None, msp_threshold: float = 0.50):
        self.classes = classes or list(DEFAULT_BREEDS)
        self.msp_threshold = msp_threshold

    @abstractmethod
    def preprocess(self, img: Image.Image) -> Any:
        """Tiền xử lý ảnh đầu vào: Resize 256, CenterCrop 224 và chuẩn hóa."""
        pass

    @abstractmethod
    def predict_raw(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Trích xuất logits phân loại và đặc trưng (features) trước lớp fully-connected.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - logits: mảng 1D độ dài num_classes (dành cho MSP, MaxLogit, Energy)
                - features: mảng 1D kích thước D (dành cho Mahalanobis, KNN distance)
        """
        pass

    @abstractmethod
    def predict(self, img: Image.Image) -> Dict[str, Any]:
        """Thực hiện suy luận phân loại và tính điểm OOD, trả về Dict chuẩn JSON v1.0."""
        pass


class MockCatClassifier(BaseCatClassifier):
    """Mô hình tạm thời cho Tuần 2 & 3.
    
    Sinh kết quả giả lập nhưng ổn định và nhất quán dựa trên mã băm SHA-256 của ảnh.
    Cung cấp đầy đủ logits, features, top-5 xác suất và điểm 5 phương pháp OOD.
    """

    def __init__(self, classes: Optional[List[str]] = None, msp_threshold: float = 0.50):
        super().__init__(classes=classes, msp_threshold=msp_threshold)
        self.backbone = "MobileNetV2 (mock)"
        self.mode = "feature_extractor"
        self.checkpoint_id = "mock-checkpoint-v1.0"

    def preprocess(self, img: Image.Image) -> Image.Image:
        """Resize cạnh ngắn về 256 rồi CenterCrop 224 (khớp tiêu chuẩn tuần 4)."""
        img = img.convert("RGB")
        w, h = img.size
        s = 256 / min(w, h)
        img = img.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
        w, h = img.size
        left, top = (w - 224) // 2, (h - 224) // 2
        return img.crop((left, top, left + 224, top + 224))

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict_raw(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo logits và feature vector giả lập nhất quán theo SHA-256 nội dung ảnh."""
        x = self.preprocess(img)
        seed = int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # Logits cho 12 giống mèo
        logits = rng.normal(0.0, 1.0, len(self.classes))
        logits[rng.integers(len(self.classes))] += rng.uniform(1.0, 5.0)

        # Features giả lập (D = 128 chiều)
        features = rng.normal(0.0, 1.0, 128)
        return logits, features

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        logits, features = self.predict_raw(img)
        x = self.preprocess(img)
        seed = int(hashlib.sha256(x.tobytes()).hexdigest()[:8], 16)

        probs = self._softmax(logits)
        order = np.argsort(-probs)[:5]
        top5 = [{"label": self.classes[i], "prob": round(float(probs[i]), 4)} for i in order]

        msp_score = float(probs.max())
        accept = msp_score >= self.msp_threshold

        # 5 phương pháp OOD với quy ước chuẩn hóa: ĐIỂM CÀNG CAO CÀNG GIỐNG MÈO IN-DISTRIBUTION
        ood_main = {
            "method": "MSP (tạm)",
            "score": round(msp_score, 4),
            "threshold": self.msp_threshold,
            "accept": accept,
        }

        max_logit = float(logits.max())
        energy_score = float(np.log(np.sum(np.exp(logits))))
        # Mahalanobis & KNN mô phỏng khoảng cách âm (khoảng cách càng nhỏ -> điểm âm càng gần 0 -> càng in-distribution)
        mahalanobis_score = round(float(msp_score * 3.0 - 1.5), 2)
        knn_score = round(float(msp_score * 2.0 - 1.2), 2)

        ood_all = [
            ood_main,
            {"method": "MaxLogit", "score": round(max_logit, 2), "threshold": 5.1, "accept": max_logit >= 5.1},
            {"method": "Energy Score", "score": round(energy_score, 2), "threshold": 6.0, "accept": energy_score >= 6.0},
            {"method": "Mahalanobis (Feature space)", "score": mahalanobis_score, "threshold": 0.0, "accept": mahalanobis_score >= 0.0},
            {"method": "KNN Distance (k=10)", "score": knn_score, "threshold": 0.0, "accept": knn_score >= 0.0},
        ]

        return {
            "api_version": "1.0",
            "is_mock": True,
            "status": "accepted" if accept else "rejected",
            "top_label": top5[0]["label"] if accept else None,
            "top5": top5,
            "ood": ood_main,
            "ood_all": ood_all,
            "model": {
                "backbone": self.backbone,
                "mode": self.mode,
                "seed": seed,
                "checkpoint_id": self.checkpoint_id,
            },
            "heatmap": None,  # Tuần 12: Grad-CAM base64 PNG
        }


# Tương thích ngược với mã cũ
default_model = MockCatClassifier()
BREEDS = default_model.classes


def preprocess(img: Image.Image) -> Image.Image:
    return default_model.preprocess(img)


def predict(img: Image.Image) -> Dict[str, Any]:
    return default_model.predict(img)
