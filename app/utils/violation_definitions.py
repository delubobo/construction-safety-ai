"""
Maps YOLOv8 class labels from the PPE detection model to
human-readable descriptions and corrective actions.

Class names match the Roboflow Construction Site Safety dataset:
  universe.roboflow.com/roboflow-100/construction-safety-gsnvb

Violation classes:  NO-Hardhat, NO-Safety Vest, NO-Mask
Safe/neutral classes: Hardhat, Safety Vest, Mask, Person,
                      Safety Cone, machinery, vehicle
"""

# Classes that represent a PPE violation (exact Roboflow class names)
VIOLATION_CLASSES = {
    "NO-Hardhat",
    "NO-Safety Vest",
}

# Classes that are compliant or neutral — not flagged
SAFE_CLASSES = {
    "Hardhat",
    "Safety Vest",
    "Person",
    "Safety Cone",
    "machinery",
    "vehicle",
}

# Classes to suppress entirely — never shown in UI, annotations, or reports
HIDDEN_CLASSES = {
    "Mask",
    "NO-Mask",
}

VIOLATION_DEFINITIONS = {
    "NO-Hardhat": {
        "label": "Missing Hard Hat",
        "severity": "High",
        "corrective_action": (
            "Worker must immediately don an ANSI/ISEA Z89.1 Class E hard hat. "
            "Stop work until compliant. Issue written warning per safety plan."
        ),
    },
    "NO-Safety Vest": {
        "label": "Missing Hi-Vis Vest",
        "severity": "High",
        "corrective_action": (
            "Worker must put on ANSI/ISEA 107 Class 2 or 3 high-visibility vest. "
            "Required at all times in active work zones."
        ),
    }
}

SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def get_violation_info(class_name: str) -> dict:
    """Return definition dict for a violation class, or a generic fallback."""
    if class_name in VIOLATION_DEFINITIONS:
        return VIOLATION_DEFINITIONS[class_name]
    # Fallback for any unknown violation class
    return {
        "label": class_name.replace("-", " ").title(),
        "severity": "Medium",
        "corrective_action": "Review PPE requirements with site safety officer.",
    }


def is_violation(class_name: str) -> bool:
    """Return True if the detected class represents a PPE violation."""
    if class_name in SAFE_CLASSES:
        return False
    if class_name in VIOLATION_CLASSES:
        return True
    # Heuristic: any remaining "NO-" class is a violation
    if class_name.startswith("NO-"):
        return True
    return False
