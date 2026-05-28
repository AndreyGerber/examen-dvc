import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def normalize_data():
    # Adjusted to your correct directory name
    input_dir = "data/processed_data"
    models_dir = "models"
    
    # 1. Check if the required split files exist
    train_path = f"{input_dir}/X_train.csv"
    test_path = f"{input_dir}/X_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Missing input files in '{input_dir}'. Please run split_data.py first.")
        return

    # Read the datasets
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)
    
    # Initialize and fit the StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to keep column names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 2. Check if models directory exists, if not, create it
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Directory '{models_dir}' has been successfully created.")
        
    # Save the scaled datasets and the scaler model
    X_train_scaled_df.to_csv(f"{input_dir}/X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv(f"{input_dir}/X_test_scaled.csv", index=False)
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")
    
    # 3. English confirmation message
    print("✓ Step 2/5: Data normalization completed successfully. Scaled datasets saved.")

if __name__ == "__main__":
    normalize_data()
