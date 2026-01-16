# ============================================
# Preprocessing Pipeline for train.csv and test.csv
# This script:
# 1) Loads train.csv
# 2) Drops unwanted columns
# 3) Builds a preprocessing pipeline
# 4) Fits it on training data
# 5) Applies it to test.csv
# 6) Exports processed_test.csv
# ============================================

# Importing necessary libraries
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ============================================
# Load Training Data
# ============================================

train_df = pd.read_csv("Data/train.csv")

print("Training data shape:", train_df.shape)

# ============================================
# Drop Unnecessary Columns
# ============================================

columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
train_df = train_df.drop(columns=columns_to_drop, errors="ignore")

# Target column
target_column = "SalePrice"

X_train = train_df.drop(target_column, axis=1)
y_train = train_df[target_column]

print("Features shape:", X_train.shape)
print("Target shape:", y_train.shape)

# ============================================
# Separate Column Types
# ============================================

categorical_cols = X_train.select_dtypes(include=["object"]).columns
numerical_cols = X_train.select_dtypes(exclude=["object"]).columns

print("Categorical columns:", len(categorical_cols))
print("Numerical columns:", len(numerical_cols))

# ============================================
# Build Pipelines
# ============================================

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# ============================================
# Fit Preprocessor on Training Data
# ============================================

preprocessor.fit(X_train)
print("Preprocessing pipeline fitted on training data.")

# ============================================
# Load Test Data
# ============================================

test_df = pd.read_csv("Data/test.csv")
print("Test data shape (before drop):", test_df.shape)

test_df = test_df.drop(columns=columns_to_drop, errors="ignore")
print("Test data shape (after drop):", test_df.shape)

# ============================================
# Transform Test Data
# ============================================

X_test_processed = preprocessor.transform(test_df)

# ============================================
# Get Feature Names After Encoding
# ============================================

cat_feature_names = preprocessor.named_transformers_["cat"] \
    .named_steps["encoder"] \
    .get_feature_names_out(categorical_cols)

feature_names = list(numerical_cols) + list(cat_feature_names)

# Convert to DataFrame
if hasattr(X_test_processed, "toarray"):
    X_test_processed = X_test_processed.toarray()

X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)

print("Processed test data shape:", X_test_processed.shape)

# ============================================
# Export Processed Test Data
# ============================================

X_test_processed.to_csv("processed_test.csv", index=False)
print("Processed test file saved as processed_test.csv")

# ============================================
# End of Script
# ============================================
