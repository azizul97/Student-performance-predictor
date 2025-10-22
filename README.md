# Predicting Student Performance

Simple data analytics & machine learning project to predict whether a student is a high performer
(average score >= 70) from demographic/background features.

## Files
- `datastudents_performance.csv.csv` — dataset (must be in same folder)
- `predict_student_performance.py` — main script (run it)
- `requirements.txt` — Python dependencies
- `outputs/` — model and artifacts produced after running the script

## How to run
1. Create a virtual environment (recommended):
   - Windows:
     ```powershell
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

2. Install dependencies:
```bash
pip install -r requirements.txt
