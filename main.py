"""
Loan Approval Prediction System - Main Entry Point

Runs the complete pipeline:
1. Generate/Load dataset
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Model Training & Evaluation
5. Sample Predictions
"""

import os
import sys

# Add src directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)


def main():
    """Run the complete Loan Approval Prediction pipeline."""
    print("\n")
    print("=" * 60)
    print("  LOAN APPROVAL PREDICTION SYSTEM")
    print("  Using Machine Learning")
    print("=" * 60)
    
    data_path = os.path.join(project_root, 'data', 'loan_data.csv')
    
    # -- Step 1: Generate Dataset (if not exists) --
    if not os.path.exists(data_path):
        print("\n\n[STEP 1] Generating Dataset...")
        print("-" * 60)
        sys.path.insert(0, os.path.join(project_root, 'data'))
        from generate_dataset import main as gen_main
        gen_main()
    else:
        print(f"\n\n[STEP 1] Dataset found at {data_path}")
    
    # -- Step 2: Data Preprocessing --
    print("\n\n[STEP 2] Data Preprocessing...")
    print("-" * 60)
    from data_preprocessing import full_preprocessing_pipeline
    X_train, X_test, y_train, y_test, scaler, feature_names, df = full_preprocessing_pipeline(data_path)
    
    # -- Step 3: Exploratory Data Analysis --
    print("\n\n[STEP 3] Exploratory Data Analysis...")
    print("-" * 60)
    from eda import run_full_eda
    run_full_eda(data_path)
    
    # -- Step 4: Model Training & Evaluation --
    print("\n\n[STEP 4] Model Training & Evaluation...")
    print("-" * 60)
    from model_training import run_training_pipeline
    results, best_name = run_training_pipeline(X_train, X_test, y_train, y_test, feature_names)
    
    # -- Step 5: Sample Predictions --
    print("\n\n[STEP 5] Sample Predictions...")
    print("-" * 60)
    from predict import run_predictions
    best_model = results[best_name]['model']
    run_predictions(best_model, scaler, feature_names)
    
    # -- Summary --
    output_dir = os.path.join(project_root, 'outputs')
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"\n  EDA plots saved to:      {output_dir}")
    print(f"  Results summary saved to: {output_dir}")
    print(f"  Best Model: {best_name}")
    print(f"  Accuracy: {results[best_name]['accuracy']*100:.2f}%")
    print(f"\n  To swap in Kaggle dataset:")
    print(f"     Replace data/loan_data.csv and re-run this script.\n")


if __name__ == "__main__":
    main()
