import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import os

def evaluate_model():
    processed_dir = "data/processed_data"
    models_dir = "models"
    metrics_dir = "metrics"
    
    # 1. Safety check for required input files
    test_x_path = f"{processed_dir}/X_test_scaled.csv"
    test_y_path = f"{processed_dir}/y_test.csv"
    model_path = f"{models_dir}/gbr_model.pkl"
    
    if not os.path.exists(test_x_path) or not os.path.exists(test_y_path) or not os.path.exists(model_path):
        print("Error: Missing required files for evaluation. Please run training.py first.")
        return
        
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    print("\n--- Starting Final Model Evaluation ---")
    print("Loading trained GBR model and test datasets...")
    
    # Load test data and the final trained model
    X_test_scaled = pd.read_csv(test_x_path)
    y_test = pd.read_csv(test_y_path).values.ravel()
    model = joblib.load(model_path)
    
    print("Predicting silica concentration on unseen test data...")
    # Generate predictions
    predictions = model.predict(X_test_scaled)
    
    # Save predictions directly inside the 'data/' directory (as required by layout)
    pred_df = pd.DataFrame(predictions, columns=["predicted_silica_concentrate"])
    pred_df.to_csv("data/prediction.csv", index=False)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Format scores and round to 4 decimal places
    scores = {
        "MSE": round(float(mse), 4),
        "R2": round(float(r2), 4)
    }

    
    # Save metrics into metrics/scores.json for DagsHub dashboard
    with open(f"{metrics_dir}/scores.json", "w") as f:
        json.dump(scores, f, indent=4)
        
    # 2. English confirmation message displaying the final quality scores
    print("----------------------------------------------------------------")
    print("✓ Step 5/5: Model evaluation completed successfully.")
    print(f"   -> Predictions saved to 'data/prediction.csv'.")
    print(f"   -> Evaluation metrics saved to '{metrics_dir}/scores.json'.")
    print(f"   -> Final Performance - R2-Score: {r2:.4f} | MSE: {mse:.4f}")
    print(f"   -> Well done! Like steak.")
    print("================================================================\n")

if __name__ == "__main__":
    evaluate_model()
