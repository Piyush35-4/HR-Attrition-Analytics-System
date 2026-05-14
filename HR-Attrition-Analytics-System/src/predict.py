#This module is imported by app.py.

import os
import joblib

from src.preprocessing import preprocess_single_input


# Constants
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# using Random Forest as the best model. 
BEST_MODEL_NAME = 'logistic_regression'


def load_best_model():
    path = os.path.join(MODELS_DIR, f'{BEST_MODEL_NAME}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Please run 'python main.py' or 'python src/train_model.py' first."
        )
    model = joblib.load(path)
    return model


def predict_attrition(input_dict):

    #Preprocessing the input
    X_scaled = preprocess_single_input(input_dict)

    model = load_best_model()

    # Get prediction (0 = Stay, 1 = Leave)
    prediction = model.predict(X_scaled)[0]

    # probability scores
    proba = model.predict_proba(X_scaled)[0]  # [prob_stay, prob_leave]
    confidence = round(proba[prediction] * 100, 1)

    # Determine risk level
    leave_probability = round(proba[1] * 100, 1)
    if leave_probability >= 70:
        risk_level = 'High'
    elif leave_probability >= 40:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'

    # res
    if prediction == 1:
        label = 'Likely to Leave'
        prediction_str = 'Yes'
    else:
        label = 'Not Likely to Leave'
        prediction_str = 'No'

    explanation = build_explanation(input_dict, prediction, leave_probability, risk_level)

    return {
        'prediction': prediction_str,
        'label': label,
        'confidence': confidence,
        'leave_probability': leave_probability,
        'explanation': explanation,
        'risk_level': risk_level,
    }


def build_explanation(input_dict, prediction, leave_probability, risk_level):
    """
    Generate a simple, human-readable explanation of why the model made its prediction.

    Returns:
        str: Human-readable explanation
    """
    factors = []

    # Assess key risk factors
    overtime = str(input_dict.get('OverTime', 'No'))
    if overtime.lower() in ['yes', '1']:
        factors.append("works overtime (a strong predictor of burnout)")

    job_sat = int(input_dict.get('JobSatisfaction', 3))
    if job_sat <= 2:
        factors.append("has low job satisfaction")

    income = int(input_dict.get('MonthlyIncome', 5000))
    if income < 3000:
        factors.append("earns a below-average monthly income")

    wlb = int(input_dict.get('WorkLifeBalance', 3))
    if wlb <= 2:
        factors.append("reports poor work-life balance")

    env_sat = int(input_dict.get('EnvironmentSatisfaction', 3))
    if env_sat <= 2:
        factors.append("is dissatisfied with the work environment")

    years = int(input_dict.get('YearsAtCompany', 5))
    if years <= 2:
        factors.append("is relatively new to the company (< 2 years)")

    # Build the explanation text
    if prediction == 1:
        if factors:
            factor_text = ", ".join(factors[:-1])
            if len(factors) > 1:
                factor_text += f", and {factors[-1]}"
            else:
                factor_text = factors[0]
            return (
                f"The model predicts a {risk_level.lower()} risk of attrition ({leave_probability:.0f}% probability). "
                f"Key contributing factors: the employee {factor_text}. "
                "HR should consider a retention conversation."
            )
        else:
            return (
                f"The model predicts a {risk_level.lower()} risk of attrition ({leave_probability:.0f}% probability) "
                "based on the overall profile. Consider monitoring engagement levels."
            )
    else:
        return (
            f"The model predicts this employee is likely to stay (attrition probability: {leave_probability:.0f}%). "
            "No immediate retention risk detected based on the provided data."
        )
