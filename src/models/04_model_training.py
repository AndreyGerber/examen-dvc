import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os

def train_model():
    input_dir = "data/processed_data"
    models_dir = "models"
    
    # 1. Safety check for the input data and best parameters
    train_x_path = f"{input_dir}/X_train_scaled.csv"
    train_y_path = f"{input_dir}/y_train.csv"
    params_path = f"{models_dir}/best_params.pkl"
    
    if not os.path.exists(train_x_path) or not os.path.exists(train_y_path) or not os.path.exists(params_path):
        print(f"Error: Missing required files. Please ensure you have run the normalization and grid_search scripts first.")
        return
    
    print("\n--- Starting Final Model Training (Gradient Boosting Regressor) ---")
    print("Linear regression is too basic for this task, neuronal networks are too complex, GBR should be fine.")
    print("Loading optimized hyperparameters...")
    
    # Load data and the best parameters found by GridSearch
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    best_params = joblib.load(params_path)
    
    # Initialize the GBR model with the exact winning parameters
    model = GradientBoostingRegressor(**best_params, random_state=42)
    
    print("Training the model on the full scaled training dataset...")
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the final trained model (This is critical for your DagsHub submission!)
    joblib.dump(model, f"{models_dir}/gbr_model.pkl")
    
    # 2. English confirmation message
    print("----------------------------------------------------------------")
    print("✓ Step 4/5: Model training completed successfully.")
    print(f"   -> Final model saved as '{models_dir}/gbr_model.pkl'.")
    print("   -> Your model is trained and ready for performance evaluation!\n")

if __name__ == "__main__":
    train_model()
