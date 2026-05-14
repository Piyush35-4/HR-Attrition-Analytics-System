"""
app.py
-------
Flask web application for HR Attrition Prediction.

Routes:
  GET  /           → Home page with employee input form
  POST /predict    → Receives form data, runs prediction, shows result page
  GET  /analytics  → Shows EDA charts in the browser

How to run:
  python app.py

Then visit: http://127.0.0.1:5000
"""

import os
import sys

# Allow Python to find our src/ package
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, request, redirect, url_for
from src.predict import predict_attrition

# ─────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'hr_attrition_secret_key'

# Folder where EDA/training charts are saved
OUTPUTS_FOLDER = os.path.join(os.path.dirname(__file__), 'outputs')


# ─────────────────────────────────────────────
# Helper: Parse and validate form input
# ─────────────────────────────────────────────

def parse_form_input(form):
    """
    Extract and type-cast form values from the Flask request.

    We need to ensure numerical fields are integers and text fields
    are strings before passing to the preprocessing module.

    Args:
        form (ImmutableMultiDict): request.form from Flask

    Returns:
        dict: Cleaned input ready for predict_attrition()
        list: Validation error messages (empty if all OK)
    """
    errors = []
    data = {}

    # ── Required numerical fields ────────────────────────────────
    int_fields = {
        'Age':                    (18, 65),
        'MonthlyIncome':          (1000, 100000),
        'JobSatisfaction':        (1, 4),
        'YearsAtCompany':         (0, 40),
        'WorkLifeBalance':        (1, 4),
        'EnvironmentSatisfaction':(1, 4),
        'DistanceFromHome':       (1, 29),
        'NumCompaniesWorked':     (0, 9),
        'TotalWorkingYears':      (0, 40),
        'PercentSalaryHike':      (11, 25),
        'TrainingTimesLastYear':  (0, 6),
        'YearsInCurrentRole':     (0, 18),
        'YearsSinceLastPromotion':(0, 15),
        'YearsWithCurrManager':   (0, 17),
    }

    for field, (lo, hi) in int_fields.items():
        raw = form.get(field, '').strip()
        if not raw:
            errors.append(f"{field} is required.")
            data[field] = lo  # Default fallback
        else:
            try:
                val = int(raw)
                if not (lo <= val <= hi):
                    errors.append(f"{field} must be between {lo} and {hi}.")
                data[field] = val
            except ValueError:
                errors.append(f"{field} must be a whole number.")
                data[field] = lo

    # ── Required categorical fields ──────────────────────────────
    str_fields = ['Department', 'OverTime', 'BusinessTravel',
                  'Gender', 'MaritalStatus', 'JobRole', 'EducationField']
    for field in str_fields:
        val = form.get(field, '').strip()
        if not val:
            errors.append(f"{field} is required.")
        data[field] = val

    # ── Fields with defaults (not in form but needed by model) ───
    data.setdefault('Education', 3)
    data.setdefault('HourlyRate', 65)
    data.setdefault('DailyRate', 800)
    data.setdefault('MonthlyRate', 14000)
    data.setdefault('JobInvolvement', 3)
    data.setdefault('JobLevel', 2)
    data.setdefault('PerformanceRating', 3)
    data.setdefault('RelationshipSatisfaction', 3)
    data.setdefault('StockOptionLevel', 1)

    return data, errors


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    """
    Home page: Renders the employee input form.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction route:
      1. Parse and validate form data
      2. Run the ML model prediction
      3. Render the result page
    """
    form_data, errors = parse_form_input(request.form)

    if errors:
        # Re-render the form with validation errors
        return render_template('index.html', errors=errors, form_data=form_data)

    try:
        result = predict_attrition(form_data)
        return render_template('result.html', result=result, input_data=form_data)

    except FileNotFoundError as e:
        error_msg = str(e)
        return render_template('index.html', errors=[error_msg], form_data=form_data)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return render_template('index.html',
                               errors=[f"An unexpected error occurred: {str(e)}"],
                               form_data=form_data)


@app.route('/analytics')
def analytics():
    """
    Analytics page-->Displays EDA charts saved in outputs/.
    """
    chart_files = []
    chart_info = {
        'attrition_distribution.png': 'Attrition Distribution',
        'age_vs_attrition.png':       'Age vs Attrition',
        'income_vs_attrition.png':    'Monthly Income vs Attrition',
        'overtime_vs_attrition.png':  'Overtime vs Attrition',
        'department_attrition.png':   'Department-wise Attrition',
        'correlation_heatmap.png':    'Correlation Heatmap',
        'feature_importance.png':     'Feature Importance (Random Forest)',
        'model_comparison.png':       'Model Comparison',
    }

    for filename, title in chart_info.items():
        full_path = os.path.join(OUTPUTS_FOLDER, filename)
        if os.path.exists(full_path):
            chart_files.append({'filename': filename, 'title': title})

    return render_template('analytics.html', charts=chart_files)


@app.route('/outputs/<filename>')
def serve_output(filename):
    """
    Serve chart images from the outputs/ folder.
    Flask's send_from_directory is used for security.
    """
    from flask import send_from_directory
    return send_from_directory(OUTPUTS_FOLDER, filename)



if __name__ == '__main__':
    print("\n" + "="*55)
    print("  HR Attrition Analytics & Prediction System")
    print("  Flask Web App — http://127.0.0.1:5000")
    print("="*55 + "\n")

    # Check that models exist before starting
    model_path = os.path.join('models', 'random_forest.pkl')
    if not os.path.exists(model_path):
        print("[WARNING] Trained model not found!")
        print("Please run 'python main.py' first to train the models.\n")

    app.run(debug=True, host='127.0.0.1', port=5000)
