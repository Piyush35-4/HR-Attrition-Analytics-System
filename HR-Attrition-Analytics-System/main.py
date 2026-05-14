import os
import sys

# Make sure Python can find our src/ modules
sys.path.insert(0, os.path.dirname(__file__))


def banner(title):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step_generate_data():
    banner("STEP 1: Generating Dataset")
    data_path = os.path.join('data', 'HR-Attrition-Analytics-System\data\WA_Fn-UseC_-HR-Employee-Attrition.csv')
    if os.path.exists(data_path):
        print(f"[INFO] Dataset already exists at {data_path}. Skipping generation.")
        print("[TIP]  To use the real IBM dataset from Kaggle, place it at:")
        print(f"       {os.path.abspath(data_path)}")


def step_run_eda():
    banner("STEP 2: Exploratory Data Analysis")
    from src.eda import run_eda
    run_eda()


def step_train_models():
    banner("STEP 3: Training ML Models")
    from src.train_model import run_training_pipeline
    run_training_pipeline(tune_hyperparams=False)


def step_final_instructions():
    banner("SETUP COMPLETE ✓")
    print("\nAll models are trained and saved to models/")
    print("All EDA charts are saved to outputs/")
    print("\n" + "─" * 60)
    print("  START THE WEB APP:")
    print("  $ python app.py")
    print("\n  Then open in your browser:")
    print("  http://127.0.0.1:5000")
    print("─" * 60 + "\n")


if __name__ == '__main__':
    step_generate_data()
    step_run_eda()
    step_train_models()
    step_final_instructions()
