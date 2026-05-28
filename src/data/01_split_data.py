import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data():
    raw_data_path = "data/raw_data/raw.csv"
    output_dir = "data/processed_data"
    
    # 1. Check if the raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"Error: No file found at '{raw_data_path}'. Please make sure the file exists.")
        return

    # Read data
    df = pd.read_csv(raw_data_path)
    
    # Remove timestamp column as it is irrelevant for the prediction
    if 'date' in df.columns:
        df = df.drop(columns=['date'])
    
    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Check if output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' did not exist and has been successfully created.")
    
    # Save the 4 datasets into data/processed
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)
    
    # 3. Confirmation message when all steps are completed
    print("✓ Step 1/5: Data splitting completed successfully. All 4 datasets have been saved.")

if __name__ == "__main__":
    split_data()
