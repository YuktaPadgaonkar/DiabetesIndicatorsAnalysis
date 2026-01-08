import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score
import xgboost as xgb
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DIABETES PREDICTION MODEL - IMPROVED VERSION")
print("=" * 80)

# Load and preprocess data
print("\n[1/7] Loading and preprocessing data...")
df = pd.read_csv("df_cleaned.csv", dtype={'CancerType': str})

# Handle missing values
null_ratio = df.isnull().sum() / len(df)
drop_cols = null_ratio[null_ratio > 0.40].index
df = df.drop(columns=drop_cols).copy()
df = df.dropna().copy()

print(f"   Dataset shape: {df.shape}")
print(f"   Target distribution:\n{df['HasDiabetes'].value_counts()}")

# Ordinal encoding
ordinal_maps = {
    "AgeGroup": ['18–24', '25–34', '35–44', '45–54', '55–64', '65+'],
    "EducationLevel": [
        'Less than High School',
        'High School Graduate',
        'Some College or Technical School',
        'College Graduate'
    ],
    "IncomeCategoryLabel": [
        '< $15,000',
        '$15,000–$24,999',
        '$25,000–$34,999',
        '$35,000–$49,999',
        '$50,000–$99,999',
        '$100,000–$199,999',
        '$200,000 or more'
    ]
}

ordinal_maps = {
    col: order for col, order in ordinal_maps.items() if col in df.columns
}

for col, order in ordinal_maps.items():
    mapping = {cat: i+1 for i, cat in enumerate(order)}
    df[col] = df[col].map(mapping)

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include='number').columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols = [c for c in categorical_cols if c not in ordinal_maps]

# One-hot encoding
encoder = OneHotEncoder(sparse_output=False)
encoded = encoder.fit_transform(df[categorical_cols])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=df.index
)

# Scale numeric features
scaler = MinMaxScaler()
scaled_numeric = scaler.fit_transform(df[numeric_cols])
numeric_df = pd.DataFrame(
    scaled_numeric,
    columns=numeric_cols,
    index=df.index
)

# Combine all features
ordinal_df = df[list(ordinal_maps.keys())]
processed_data = pd.concat([numeric_df, ordinal_df, encoded_df], axis=1)

# Feature engineering - create interaction features
print("\n[2/7] Engineering features...")
features_added = 0

# Check what columns are available
available_cols = processed_data.columns.tolist()

# Try to create Age-BMI interaction
age_col = 'AgeGroup' if 'AgeGroup' in available_cols else None
bmi_cols = [col for col in available_cols if 'BMI' in col.upper()]
if age_col and bmi_cols:
    for bmi_col in bmi_cols[:1]:  # Use first BMI column found
        try:
            processed_data['Age_BMI_Interaction'] = processed_data[age_col] * processed_data[bmi_col]
            features_added += 1
            break
        except:
            pass

# Try to create Income-Education interaction
if 'IncomeCategoryLabel' in available_cols and 'EducationLevel' in available_cols:
    try:
        processed_data['Income_Education_Interaction'] = processed_data['IncomeCategoryLabel'] * processed_data['EducationLevel']
        features_added += 1
    except:
        pass

print(f"   Total features after engineering: {processed_data.shape[1]} ({features_added} new features added)")

# Prepare target and features
target_df = processed_data["HasDiabetes"]
features_df = processed_data.drop('HasDiabetes', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features_df, target_df, test_size=0.2, random_state=42, stratify=target_df
)

print(f"\n[3/7] Handling class imbalance...")
print(f"   Original training set distribution:")
print(f"   Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")

# Try multiple resampling strategies
resampling_methods = {
    "SMOTE": SMOTE(random_state=42, sampling_strategy=0.8),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42, sampling_strategy=0.8),
    "SMOTETomek": SMOTETomek(random_state=42, smote=SMOTE(sampling_strategy=0.8))
}

# Use SMOTETomek for better results (hybrid approach)
best_sampler = resampling_methods["SMOTETomek"]
X_train_res, y_train_res = best_sampler.fit_resample(X_train, y_train)

print(f"   After resampling with SMOTETomek:")
print(f"   Class 0: {(y_train_res == 0).sum()}, Class 1: {(y_train_res == 1).sum()}")

# Define models with class weights and improved parameters
print("\n[4/7] Training models with hyperparameter tuning...")

models = {}

# Logistic Regression with class weight
models["Logistic Regression"] = LogisticRegression(
    max_iter=10000,
    class_weight='balanced',
    C=0.1,
    random_state=42
)

# Random Forest with tuned parameters
models["Random Forest"] = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting with better parameters
models["Gradient Boosting"] = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    subsample=0.8,
    random_state=42
)

# XGBoost - often performs better
models["XGBoost"] = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train_res == 0).sum() / (y_train_res == 1).sum(),
    random_state=42,
    eval_metric='logloss'
)

# LightGBM - fast and efficient
models["LightGBM"] = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42,
    verbose=-1
)

# Train and evaluate models
print("\n[5/7] Training and evaluating individual models...")
print("=" * 80)

results = {}
best_f1 = 0
best_model_name = None
best_model = None

# Convert to numpy arrays for compatibility with all models
X_train_res_array = X_train_res.values if hasattr(X_train_res, 'values') else X_train_res
y_train_res_array = y_train_res.values if hasattr(y_train_res, 'values') else y_train_res
X_test_array = X_test.values if hasattr(X_test, 'values') else X_test
y_test_array = y_test.values if hasattr(y_test, 'values') else y_test

for name, model in models.items():
    print(f"\n{name}")
    print("-" * 80)

    # Train model
    model.fit(X_train_res_array, y_train_res_array)

    # Predictions
    y_pred = model.predict(X_test_array)
    y_pred_proba = model.predict_proba(X_test_array)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test_array, y_pred)
    roc_auc = roc_auc_score(y_test_array, y_pred_proba)
    f1 = f1_score(y_test_array, y_pred, average='weighted')

    # Store results
    results[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    # Track best model by F1 score
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = model

    # Print detailed metrics
    print(classification_report(y_test_array, y_pred))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test_array, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

# Create ensemble model using top performers
print("\n[6/7] Creating ensemble model...")
print("=" * 80)

# Select top 3 models for voting
voting_estimators = [
    ('xgb', models["XGBoost"]),
    ('lgbm', models["LightGBM"]),
    ('gb', models["Gradient Boosting"])
]

ensemble_model = VotingClassifier(
    estimators=voting_estimators,
    voting='soft',
    n_jobs=-1
)

ensemble_model.fit(X_train_res_array, y_train_res_array)
y_pred_ensemble = ensemble_model.predict(X_test_array)
y_pred_proba_ensemble = ensemble_model.predict_proba(X_test_array)[:, 1]

# Evaluate ensemble
print("\nEnsemble Model (Soft Voting)")
print("-" * 80)
print(classification_report(y_test_array, y_pred_ensemble))

accuracy_ensemble = accuracy_score(y_test_array, y_pred_ensemble)
roc_auc_ensemble = roc_auc_score(y_test_array, y_pred_proba_ensemble)
f1_ensemble = f1_score(y_test_array, y_pred_ensemble, average='weighted')

print(f"Accuracy: {accuracy_ensemble:.4f}")
print(f"ROC-AUC Score: {roc_auc_ensemble:.4f}")
print(f"Weighted F1-Score: {f1_ensemble:.4f}")

cm_ensemble = confusion_matrix(y_test_array, y_pred_ensemble)
print(f"\nConfusion Matrix:")
print(cm_ensemble)

# Summary comparison
print("\n[7/7] Model Comparison Summary")
print("=" * 80)
print(f"{'Model':<25} {'Accuracy':<12} {'ROC-AUC':<12} {'F1-Score':<12}")
print("-" * 80)

for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:<12.4f} {res['roc_auc']:<12.4f} {res['f1_score']:<12.4f}")

print(f"{'Ensemble (Voting)':<25} {accuracy_ensemble:<12.4f} {roc_auc_ensemble:<12.4f} {f1_ensemble:<12.4f}")
print("=" * 80)

print(f"\nBest Individual Model: {best_model_name}")
print(f"Best F1-Score: {best_f1:.4f}")

# Save feature importance from best tree-based model
if hasattr(best_model, 'feature_importances_'):
    print(f"\n\nTop 10 Most Important Features ({best_model_name}):")
    print("-" * 80)
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:<50} {row['importance']:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
