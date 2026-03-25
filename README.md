# Construction Safety AI

**Live demo:** *(link added after HF Spaces deploy)*

A computer vision tool that automatically detects PPE (Personal Protective Equipment) violations on construction sites using YOLOv8.

## What it does

Upload a construction site photo and the app:
1. Runs YOLOv8 object detection to identify workers and their PPE
2. Flags violations (missing hard hat, vest, gloves, etc.) with bounding boxes
3. Generates a downloadable PDF safety inspection report with annotated photo and corrective actions
4. Logs every session to SQLite and shows a violation trend chart over time

## Tech stack

| Layer | Choice |
|---|---|
| CV Model | YOLOv8 (ultralytics) |
| App | Streamlit |
| PDF | ReportLab |
| Storage | SQLite + SQLAlchemy |
| Hosting | Hugging Face Spaces |

## Research Context

This project was built as a portfolio demonstration of AI-assisted safety compliance for large-scale construction projects. On a real data center construction site, safety officers inspect hundreds of workers daily across multiple active zones. This tool automates the detection step — the officer uploads a drone or CCTV frame, the model flags violations in seconds, and a legally-defensible PDF report is generated automatically.

The workflow maps directly to OSHA 29 CFR 1926 Subpart E (PPE requirements) and mirrors what enterprise safety platforms like Smartvid.io and Intenseye provide at enterprise pricing. This implementation demonstrates the same core capability using open-source tools.

## Run locally

```bash
py -3.12 -m venv venv
./venv/Scripts/activate          # Windows (Git Bash)
# source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
streamlit run app/main.py
```

## Project structure

```
construction-safety-ai/
├── model/
│   ├── detector.py          # PPEDetector class (YOLOv8 wrapper)
│   ├── report_builder.py    # ReportLab PDF generator
│   └── weights/             # .pt model files (gitignored)
├── app/
│   ├── main.py              # Streamlit UI
│   ├── database.py          # SQLAlchemy session logging
│   └── utils/
│       ├── image_annotator.py       # PIL bounding box overlay
│       └── violation_definitions.py # Class → corrective action map
├── data/sample_images/      # Test photos (gitignored)
├── tests/
│   └── test_detector.py
├── app.py                   # HF Spaces entry point
└── requirements.txt
```
