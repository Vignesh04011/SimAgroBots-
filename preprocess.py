# ======================
# AGRICULTURAL DATA PREPROCESSING PIPELINE
# ======================
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 1. DATA LOADING
# ----------------------
def load_and_merge_data():
    """Load and merge the two datasets with proper error handling"""
    try:
        # Load both datasets
        dynamic_df = pd.read_excel("dynamic_features.xlsx", engine='openpyxl')
        main_df = pd.read_excel("main_features.xlsx", engine='openpyxl')
        
        print("✅ Files loaded successfully")
        print(f"Main features shape: {main_df.shape}")
        print(f"Dynamic features shape: {dynamic_df.shape}")
        
        # Find potential merge keys (case-insensitive)
        common_cols = list(set(col.lower() for col in dynamic_df.columns) & 
                       set(col.lower() for col in main_df.columns))
        
        if not common_cols:
            print("⚠️ No common columns found - attempting to merge by index")
            if len(dynamic_df) == len(main_df):
                df = pd.concat([main_df, dynamic_df], axis=1)
                print("✅ Merged by row position (same number of rows)")
            else:
                raise ValueError("Cannot merge - different row counts and no common columns")
        else:
            # Find the actual column names (case-sensitive)
            merge_col = None
            for col in common_cols:
                # Find matching case-sensitive column names
                main_col = [c for c in main_df.columns if c.lower() == col][0]
                dynamic_col = [c for c in dynamic_df.columns if c.lower() == col][0]
                
                if main_col in main_df and dynamic_col in dynamic_df:
                    merge_col = (main_col, dynamic_col)
                    break
            
            if merge_col:
                print(f"✅ Merging on columns: {merge_col[0]} (main) and {merge_col[1]} (dynamic)")
                df = pd.merge(main_df, dynamic_df, 
                             left_on=merge_col[0], right_on=merge_col[1], 
                             how='outer')
                df = df.drop(columns=[merge_col[1]])  # Drop duplicate merge column
            else:
                raise ValueError("Could not find matching column names")
        
        # Fix duplicate columns
        duplicate_pairs = [
            ('Rainfall(mm)', 'Rainfall_mm'),
            ('Temperature(C)', 'Temperature_Celsius'),
            ('Fertilizer(g)', 'Fertilizer_Used'),
            ('Irrigation_Used', 'Irrigation_Applied')
        ]
        
        for main_col, alt_col in duplicate_pairs:
            if main_col in df.columns and alt_col in df.columns:
                df[main_col] = df[main_col].fillna(df[alt_col])
                df = df.drop(columns=[alt_col])
        
        return df
    
    except Exception as e:
        print("\n❌ Error loading data:")
        print(f"Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check both files exist in the correct location")
        print("2. Verify the files have at least one common column")
        print("3. Ensure both files have the same number of rows if no common columns")
        print("4. Check for Excel file corruption")
        raise

# ----------------------
# 2. DATA CLEANING
# ----------------------
def clean_data(df):
    """Handle missing values and data quality issues"""
    # Ensure target exists
    if 'Health Score' not in df.columns:
        raise ValueError("Target column 'Health Score' not found in data")
    
    # Define columns by type
    num_cols = [
        'NDVI', 'SAVI', 'Soil_Moisture', 'Humidity', 'Canopy_Coverage',
        'Chlorophyll_Content', 'Leaf_Area_Index', 'Temperature(C)',
        'Rainfall(mm)', 'Fertilizer(g)', 'Crop Height(cm)', 'Sunlight(hrs)',
        'DayInSeason', 'Wind_Speed', 'Organic_Matter', 'Weed_Coverage',
        'Soil_pH', 'Days_to_Harvest'
    ]
    
    cat_cols = ['Crop', 'Region', 'Soil_Type', 'Weather_Condition']
    ordinal_cols = ['Pest Level', 'Crop_Stress_Indicator']
    binary_cols = ['Pest_Hotspots', 'Irrigation_Used']
    
    # Only keep columns that exist in the data
    num_cols = [col for col in num_cols if col in df.columns]
    cat_cols = [col for col in cat_cols if col in df.columns]
    ordinal_cols = [col for col in ordinal_cols if col in df.columns]
    binary_cols = [col for col in binary_cols if col in df.columns]
    
    print("\n🔍 Data cleaning summary:")
    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")
    print(f"Ordinal columns: {ordinal_cols}")
    print(f"Binary columns: {binary_cols}")
    
    # Handle missing values
    print("\n🔄 Handling missing values...")
    if num_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])
        print(f"Imputed {len(num_cols)} numerical columns with median")
    
    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        print(f"Imputed {len(cat_cols)} categorical columns with mode")
    
    if binary_cols:
        for col in binary_cols:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0})
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        print(f"Converted {len(binary_cols)} binary columns to 0/1")
    
    return df, num_cols, cat_cols, ordinal_cols, binary_cols

# ----------------------
# 3. FEATURE ENGINEERING
# ----------------------
def engineer_features(df):
    """Create new features from existing data"""
    print("\n⚙️ Engineering new features...")
    
    # Vegetation health index
    if all(col in df.columns for col in ['NDVI', 'SAVI']):
        df['Vegetation_Health'] = (df['NDVI'] + df['SAVI']) / 2
        print("- Created Vegetation_Health from NDVI and SAVI")
    
    # Pest risk score
    if all(col in df.columns for col in ['Pest_Hotspots', 'Pest Level']):
        df['Pest_Risk'] = df['Pest_Hotspots'] * df['Pest Level']
        print("- Created Pest_Risk from Pest_Hotspots and Pest Level")
    
    # Weather stress indicator
    if 'Weather_Condition' in df.columns:
        df['Weather_Stress'] = df['Weather_Condition'].str.contains(
            'storm|heavy rain|drought', case=False, regex=True).astype(int)
        print("- Created Weather_Stress from Weather_Condition")
    
    return df

# ----------------------
# 4. PREPROCESSING PIPELINE
# ----------------------
def create_preprocessing_pipeline(num_cols, cat_cols, ordinal_cols):
    """Build the complete preprocessing pipeline"""
    transformers = []
    
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
        print(f"\n🔧 Added numerical scaler for {len(num_cols)} features")
    
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(
            drop='first', 
            handle_unknown='ignore',
            sparse_output=False), cat_cols))
        print(f"🔧 Added one-hot encoder for {len(cat_cols)} categorical features")
    
    if ordinal_cols:
        transformers.append(('ordinal', OrdinalEncoder(), ordinal_cols))
        print(f"🔧 Added ordinal encoder for {len(ordinal_cols)} features")
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='passthrough',
        verbose_feature_names_out=False
    )
    
    return preprocessor

# ----------------------
# 5. MAIN EXECUTION
# ----------------------
def main():
    print("\n" + "="*50)
    print("🌱 AGRICULTURAL DATA PREPROCESSING PIPELINE")
    print("="*50 + "\n")
    
    try:
        # 1. Load and merge data
        df = load_and_merge_data()
        
        # 2. Clean data
        df, num_cols, cat_cols, ordinal_cols, binary_cols = clean_data(df)
        
        # 3. Feature engineering
        df = engineer_features(df)
        
        # 4. Prepare for modeling
        X = df.drop('Health Score', axis=1)
        y = df['Health Score']
        
        # Split before preprocessing to avoid leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # 5. Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(num_cols, cat_cols, ordinal_cols)
        
        # 6. Apply transformations
        print("\n🛠️ Applying preprocessing...")
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names
        feature_names = []
        if num_cols:
            feature_names.extend(num_cols)
        if cat_cols:
            cat_encoder = preprocessor.named_transformers_['cat']
            feature_names.extend(cat_encoder.get_feature_names_out(cat_cols))
        if ordinal_cols:
            feature_names.extend(ordinal_cols)
        
        # Add engineered features
        new_features = ['Vegetation_Health', 'Pest_Risk', 'Weather_Stress']
        feature_names.extend([f for f in new_features if f in df.columns])
        
        # 7. Save processed data
        processed_train = pd.DataFrame(X_train_processed, columns=feature_names)
        processed_test = pd.DataFrame(X_test_processed, columns=feature_names)
        
        processed_train['Health_Score'] = y_train.values
        processed_test['Health_Score'] = y_test.values
        
        processed_train.to_csv("processed_train.csv", index=False)
        processed_test.to_csv("processed_test.csv", index=False)
        
        print("\n✅ Preprocessing complete!")
        print(f"Final training set shape: {processed_train.shape}")
        print(f"Final test set shape: {processed_test.shape}")
        print("\nOutput files saved:")
        print("- processed_train.csv")
        print("- processed_test.csv")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()