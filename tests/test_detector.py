"""
Tests for PPEDetector and supporting utilities.
These tests do NOT require a GPU — they mock or use minimal inference.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from app.utils.violation_definitions import (
    get_violation_info,
    is_violation,
    VIOLATION_DEFINITIONS,
)
from app.utils.image_annotator import annotate_image, pil_to_bytes


# ── violation_definitions tests ───────────────────────────────────────────────

class TestViolationDefinitions:
    def test_known_violation_classes_are_violations(self):
        for cls in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
            assert is_violation(cls), f"{cls} should be a violation"

    def test_safe_classes_are_not_violations(self):
        for cls in ["Hardhat", "Safety Vest", "Mask", "Person", "Safety Cone", "vehicle"]:
            assert not is_violation(cls), f"{cls} should not be a violation"

    def test_unknown_no_prefix_is_violation(self):
        assert is_violation("NO-Gloves")

    def test_get_violation_info_known(self):
        info = get_violation_info("NO-Hardhat")
        assert info["label"] == "Missing Hard Hat"
        assert info["severity"] == "High"
        assert "corrective_action" in info

    def test_get_violation_info_fallback(self):
        info = get_violation_info("no-unknown-ppe")
        assert "label" in info
        assert "severity" in info
        assert "corrective_action" in info

    def test_all_violation_definitions_have_required_keys(self):
        for cls, info in VIOLATION_DEFINITIONS.items():
            assert "label" in info, f"Missing 'label' for {cls}"
            assert "severity" in info, f"Missing 'severity' for {cls}"
            assert "corrective_action" in info, f"Missing 'corrective_action' for {cls}"
            assert info["severity"] in ("High", "Medium", "Low"), \
                f"Invalid severity for {cls}: {info['severity']}"


# ── image_annotator tests ─────────────────────────────────────────────────────

class TestImageAnnotator:
    def _sample_image(self, w=640, h=480) -> Image.Image:
        return Image.new("RGB", (w, h), color=(100, 100, 100))

    def _sample_detection(self) -> dict:
        return {
            "class": "no-helmet",
            "label": "Missing Hard Hat",
            "confidence": 0.87,
            "bbox": [50, 50, 200, 250],
            "severity": "High",
            "corrective_action": "Wear a hard hat.",
            "is_violation": True,
        }

    def test_annotate_returns_pil_image(self):
        img = self._sample_image()
        result = annotate_image(img, [self._sample_detection()])
        assert isinstance(result, Image.Image)

    def test_annotate_does_not_modify_original(self):
        img = self._sample_image()
        original_pixels = list(img.getdata())
        annotate_image(img, [self._sample_detection()])
        assert list(img.getdata()) == original_pixels

    def test_annotate_empty_detections(self):
        img = self._sample_image()
        result = annotate_image(img, [])
        assert result.size == img.size

    def test_annotate_safe_detection(self):
        img = self._sample_image()
        safe_det = {
            "class": "helmet",
            "label": "Hard Hat",
            "confidence": 0.91,
            "bbox": [10, 10, 100, 100],
            "severity": "Low",
            "corrective_action": "",
            "is_violation": False,
        }
        result = annotate_image(img, [safe_det])
        assert isinstance(result, Image.Image)

    def test_pil_to_bytes_returns_bytes(self):
        img = self._sample_image()
        data = pil_to_bytes(img)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_annotate_multiple_detections(self):
        img = self._sample_image()
        dets = [
            {**self._sample_detection(), "bbox": [10, 10, 100, 100]},
            {**self._sample_detection(), "class": "no-vest", "label": "Missing Vest",
             "bbox": [200, 50, 350, 300], "severity": "High"},
        ]
        result = annotate_image(img, dets)
        assert isinstance(result, Image.Image)


# ── PPEDetector tests (mocked) ────────────────────────────────────────────────

class TestPPEDetector:
    def test_detect_violations_only_filters_correctly(self):
        """Verify detect_violations_only returns only violations."""
        from model.detector import PPEDetector

        detector = PPEDetector()

        mixed = [
            {"class": "NO-Hardhat", "label": "Missing Hard Hat", "confidence": 0.9,
             "bbox": [0, 0, 100, 100], "severity": "High",
             "corrective_action": "Wear a hat.", "is_violation": True},
            {"class": "Hardhat", "label": "Hard Hat", "confidence": 0.85,
             "bbox": [200, 0, 300, 100], "severity": "Low",
             "corrective_action": "", "is_violation": False},
        ]

        with patch.object(detector, "detect", return_value=mixed):
            violations = detector.detect_violations_only("fake_path.jpg")

        assert len(violations) == 1
        assert violations[0]["class"] == "NO-Hardhat"

    def test_model_path_property(self):
        from model.detector import PPEDetector
        detector = PPEDetector()
        assert isinstance(detector.model_path, str)
        assert len(detector.model_path) > 0
