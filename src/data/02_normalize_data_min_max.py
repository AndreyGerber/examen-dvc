import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

def normalize_data_min_max():
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
    
    print("\n=========================================================================")
    print("         STARTING ADVANCED DATA AUDIT: OUTLIERS & DIAGNOSTICS            ")
    print("=========================================================================")
    print("\n📖 QUICK STATISTICAL GUIDE (HOW TO READ THIS REPORT):")
    print("-------------------------------------------------------------------------")
    print("1. SKEWNESS (How asymmetrical is the data shape? Looking at absolute value):")
    print("   • 0.0 to 0.5  -> PERFECT [Data is beautifully symmetric and balanced]")
    print("   • 0.5 to 1.0  -> OK      [Slightly tilted to one side, but still fine]")
    print("   • Above 1.0   -> BAD     [Extremely distorted! Bad for MinMaxScaler]")
    print("\n2. KURTOSIS (How sharp or flat is the peak? Looking at absolute value):")
    print("   • 0.0 to 0.5  -> PERFECT [Looks exactly like a normal, smooth bell curve]")
    print("   • 0.5 to 1.0  -> OK      [A bit pointy or flat, but acceptable]")
    print("   • Above 1.0   -> BAD     [Looks like an extreme needle or a flat box]")
    print("=========================================================================")
    
    total_outliers = 0
    total_rows = len(X_train)
    critical_features_count = 0
    
    # Check for outliers, stats, and distribution classifications for each column
    for column in X_train.columns:
        mean_val = X_train[column].mean()
        median_val = X_train[column].median()
        
        # Shape of distribution (Skewness & Kurtosis)
        skewness = X_train[column].skew()
        kurtosis = X_train[column].kurt()  # 0 = normal bell curve
        
        # Classification for Skewness
        abs_skew = abs(skewness)
        if abs_skew <= 0.5:
            skew_status = "🟢 PERFECT (Value is between 0.0 and 0.5. Very safe!)"
        elif abs_skew <= 1.0:
            skew_status = "🟡 OK (Value is between 0.5 and 1.0. Acceptable)"
        else:
            skew_status = "🔴 BAD (Value is ABOVE 1.0. High risk of data distortion for MinMaxScaler!)"
            critical_features_count += 1
            
        # Classification for Kurtosis
        abs_kurt = abs(kurtosis)
        if abs_kurt <= 0.5:
            kurt_status = "🟢 PERFECT (Value is between 0.0 and 0.5. Beautiful bell curve!)"
        elif abs_kurt <= 1.0:
            kurt_status = "🟡 OK (Value is between 0.5 and 1.0. Still fine)"
        else:
            kurt_status = "🔴 BAD (Value is ABOVE 1.0. Peak shape is too extreme!)"
            critical_features_count += 1
        
        # IQR Calculation for Outliers
        Q1 = X_train[column].quantile(0.25)
        Q3 = X_train[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = X_train[(X_train[column] < lower_bound) | (X_train[column] > upper_bound)]
        outlier_count = len(outliers)
        total_outliers += outlier_count
        
        print(f"\n📊 Feature Analysis: '{column}'")
        print(f"   -> Data Quantity: {outlier_count} outliers found out of {total_rows} rows.")
        print(f"   -> Spread (IQR):  {IQR:.4f} (Middle 50% range of your data)")
        print(f"   -> Center:        Mean: {mean_val:.4f} | Median: {median_val:.4f}")
        print(f"   -> Skewness:      {skewness:.4f} -> {skew_status}")
        print(f"   -> Kurtosis:      {kurtosis:.4f} -> {kurt_status}")
        
        # Worst Outlier check
        if outlier_count > 0:
            max_val = X_train[column].max()
            min_val = X_train[column].min()
            worst_outlier = max_val if abs(max_val - mean_val) > abs(min_val - mean_val) else min_val
            print(f"   -> Worst Outlier: {worst_outlier:.4f} (This is the most extreme value in this column)")
        else:
            print("   -> Worst Outlier: None (No extreme single values found)")
            
    print("\n=========================================================================")
    
    # 2. Strategic MLOps Architectural Advice on Screen
    print("\n--- FINAL ARCHITECTURAL DIAGNOSIS ---")
    if total_outliers > 0 or critical_features_count > 0:
        print(f"[STATUS] Outliers total: {total_outliers}")
        print(f"[STATUS] Deformed/Bad metrics found: {critical_features_count}")
        print("[WARNING] RECOMMENDATION: Because you have 🔴 BAD metrics or active outliers,")
        print("[WARNING]                 using the 'StandardScaler' is strongly advised for this project!")
        print("[WARNING]                 MinMaxScaler will squash your normal data too much.\n")
    else:
        print("[STATUS] Your dataset looks completely balanced, clean, and normal.")
        print("[INFO] RECOMMENDATION: MinMaxScaler is safe, optimal, and ready to use!\n")

    # Proceed with MinMaxScaler application
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    X_train_scaled_df.to_csv(f"{input_dir}/X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv(f"{input_dir}/X_test_scaled.csv", index=False)
    joblib.dump(scaler, f"{models_dir}/scaler.pkl")
    
    print("✓ Step 2/5 (Alternative): Data MinMaxScaler normalization & diagnostic audit completed.")

if __name__ == "__main__":
    normalize_data_min_max()
