import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

def run_grid_search():
    # 1. FIXED: Adjusted to your correct directory name
    input_dir = "data/processed_data"
    models_dir = "models"
    
    # 2. ADDED: Safety check for input files
    train_x_path = f"{input_dir}/X_train_scaled.csv"
    train_y_path = f"{input_dir}/y_train.csv"
    
    if not os.path.exists(train_x_path) or not os.path.exists(train_y_path):
        print(f"Error: Missing input files in '{input_dir}'. Please run a normalization script first.")
        return
    
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    X_train = pd.read_csv(train_x_path)
    y_train = pd.read_csv(train_y_path).values.ravel()
    
    print("\n--- Starting Model Hyperparameter Optimization (GridSearch) ---")
    print("Please wait, the system is testing different parameter combinations using all CPU cores...")
    print('You can grap a coffee while waiting ☕️\n')
    
    # Define model & parameter grid
    gbr = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4]
    }
    
    # Run GridSearch
    grid = GridSearchCV(gbr, param_grid, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    # Save the best parameters
    joblib.dump(grid.best_params_, f"{models_dir}/best_params.pkl")
    
    # 3. FIXED: Output completely in English with the actual results displayed
    print("----------------------------------------------------------------")
    print(f"✓ Step 3/5: GridSearch completed successfully.")
    print(f"   -> Best Parameters found: {grid.best_params_}")
    print("   -> Results have been saved to 'models/best_params.pkl'.\n")
    print("   -> what are you waiting for? Go check if best parameters are there! And then go to your model training!\n")


if __name__ == "__main__":
    run_grid_search()
