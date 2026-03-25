"""
CloudPPEDetector — Roboflow Serverless API, direct violation classification.

Model: construction-site-safety/27  (roboflow-universe-projects, 81.4% mAP)
Direct output classes:
  Violations : NO-Hardhat, NO-Safety Vest, NO-Mask
  Compliant  : Hardhat, Safety Vest, Mask
  Neutral    : Person, Safety Cone, machinery, vehicle

No spatial/IoU logic needed — the model directly labels each detection
as a violation or compliant item. This is the correct architecture.
"""

import os
from typing import List, Dict

from inference_sdk import InferenceHTTPClient

from app.utils.violation_definitions import get_violation_info, is_violation, HIDDEN_CLASSES


class CloudPPEDetector:
    """
    PPE detector using Roboflow's Serverless Inference API.
    Set ROBOFLOW_API_KEY as an environment variable before running.
    """

    def __init__(self, confidence: float = 0.35):
        api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ROBOFLOW_API_KEY environment variable is not set. "
                "Add it to your .env file."
            )
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        self.model_id = "construction-site-safety/27"
        self.confidence = confidence

    def detect(self, image_path: str) -> List[Dict]:
        """
        Send image to Roboflow API and return structured detections.

        Returns list of dicts with keys:
            class, label, confidence, bbox [x1,y1,x2,y2],
            severity, corrective_action, is_violation
        """
        api_response = self.client.infer(image_path, model_id=self.model_id)

        if "predictions" not in api_response:
            return []

        detections = []

        for pred in api_response["predictions"]:
            if pred["confidence"] < self.confidence:
                continue
            if pred["class"] in HIDDEN_CLASSES:
                continue

            x1 = int(pred["x"] - pred["width"] / 2)
            y1 = int(pred["y"] - pred["height"] / 2)
            x2 = int(pred["x"] + pred["width"] / 2)
            y2 = int(pred["y"] + pred["height"] / 2)

            class_name = pred["class"]
            info = get_violation_info(class_name)

            detections.append({
                "class":            class_name,
                "label":            info["label"],
                "confidence":       round(pred["confidence"], 4),
                "bbox":             [x1, y1, x2, y2],
                "severity":         info["severity"],
                "corrective_action": info["corrective_action"],
                "is_violation":     is_violation(class_name),
            })

        # Violations first, then by confidence descending
        detections.sort(key=lambda d: (not d["is_violation"], -d["confidence"]))
        return detections

    def detect_violations_only(self, image_path: str) -> List[Dict]:
        return [d for d in self.detect(image_path) if d["is_violation"]]

    @property
    def model_path(self) -> str:
        return self.model_id


# Alias so any import of PPEDetector still works
PPEDetector = CloudPPEDetector
