import os
import glob
import sys
import types
from datetime import datetime
from typing import List, Optional, Dict
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, CatBoostRegressor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

# ==============================================================================
# Custom Transformer Classes (Copied from data_pipeline.py)
# ==============================================================================

class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        df_transformed = pd.DataFrame(index=X.index)
        transaction_start = pd.to_datetime(X['transaction_start'])
        transaction_end = pd.to_datetime(X['transaction_end'])
        duration = transaction_end - transaction_start
        ft_duration_seconds_val = duration.dt.total_seconds()
        df_transformed['ft_month'] = transaction_start.dt.month
        df_transformed['ft_day'] = transaction_start.dt.day
        df_transformed['ft_hour'] = transaction_start.dt.hour
        df_transformed['ft_minute'] = transaction_start.dt.minute
        df_transformed['ft_second'] = transaction_start.dt.second
        df_transformed['ft_day_of_week'] = transaction_start.dt.dayofweek
        df_transformed['ft_day_of_year'] = transaction_start.dt.dayofyear
        df_transformed['ft_week_of_year'] = transaction_start.dt.isocalendar().week.astype(int)
        df_transformed['ft_quarter'] = transaction_start.dt.quarter
        df_transformed['ft_is_weekend'] = df_transformed['ft_day_of_week'].isin([5, 6]).astype(int)
        df_transformed['ft_is_outside_normal_hours'] = ((df_transformed['ft_hour'] < 8) | (df_transformed['ft_hour'] >= 23)).astype(int)
        df_transformed['ft_month_sin'] = np.sin(2 * np.pi * df_transformed['ft_month']/12)
        df_transformed['ft_month_cos'] = np.cos(2 * np.pi * df_transformed['ft_month']/12)
        df_transformed['ft_hour_sin'] = np.sin(2 * np.pi * df_transformed['ft_hour']/24)
        df_transformed['ft_hour_cos'] = np.cos(2 * np.pi * df_transformed['ft_hour']/24)
        df_transformed['ft_day_of_week_sin'] = np.sin(2 * np.pi * df_transformed['ft_day_of_week']/7)
        df_transformed['ft_day_of_week_cos'] = np.cos(2 * np.pi * df_transformed['ft_day_of_week']/7)
        days_in_year = transaction_start.dt.is_leap_year.map({True: 366, False: 365})
        df_transformed['ft_day_of_year_sin'] = np.sin(2 * np.pi * df_transformed['ft_day_of_year']/days_in_year)
        df_transformed['ft_day_of_year_cos'] = np.cos(2 * np.pi * df_transformed['ft_day_of_year']/days_in_year)
        df_transformed['ft_duration_seconds_log'] = np.log1p(ft_duration_seconds_val)
        df_transformed['ft_inter_weekend_outside_hours'] = df_transformed['ft_is_weekend'] * df_transformed['ft_is_outside_normal_hours']
        df_transformed['ft_inter_hour_cos_x_dow_cos'] = df_transformed['ft_hour_cos'] * df_transformed['ft_day_of_week_cos']
        self.feature_names_out_ = ['ft_month', 'ft_day', 'ft_hour', 'ft_minute', 'ft_second', 'ft_day_of_week', 'ft_day_of_year', 'ft_week_of_year', 'ft_quarter', 'ft_is_weekend', 'ft_is_outside_normal_hours', 'ft_month_sin', 'ft_month_cos', 'ft_hour_sin', 'ft_hour_cos', 'ft_day_of_week_sin', 'ft_day_of_week_cos', 'ft_day_of_year_sin', 'ft_day_of_year_cos', 'ft_duration_seconds_log', 'ft_inter_weekend_outside_hours', 'ft_inter_hour_cos_x_dow_cos']
        return df_transformed[self.feature_names_out_]
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class FeedbackBinnerOHE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.categories_ = ['feedback_null', 'feedback_10', 'feedback_other']
        self.feature_names_out_ = [f"ft_{cat}" for cat in self.categories_]
    def fit(self, X, y=None):
        return self
    def _categorize_feedback(self, feedback_value):
        if pd.isnull(feedback_value): return 'feedback_null'
        if feedback_value == 10.0: return 'feedback_10'
        return 'feedback_other'
    def transform(self, X, y=None):
        feedback_series = X.iloc[:, 0]
        categorized_feedback = feedback_series.apply(self._categorize_feedback)
        categorized_feedback_type = pd.Categorical(categorized_feedback, categories=self.categories_)
        dummies_df = pd.get_dummies(categorized_feedback_type, prefix='ft', dtype=int)
        return dummies_df[self.feature_names_out_]
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class EnhancedTransactionLineFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, price_representation='actual', normalize=False):
        self.price_representation = price_representation
        self.normalize = normalize
        self.all_categories = ['TOBACCO', 'ALCOHOL', 'PERSONAL_CARE', 'LONG_SHELF_LIFE', 'FROZEN_GOODS', 'HOUSEHOLD', 'BEVERAGES', 'BAKERY', 'FRUITS_VEGETABLES_PIECES', 'DAIRY', 'CONVENIENCE', 'FRUITS_VEGETABLES', 'SNACKS']
        self.high_risk_categories = {'SNACKS', 'FRUITS_VEGETABLES_PIECES', 'CONVENIENCE'}
        self.new_product_days = 30
        self._initialize_feature_names()
    def _initialize_feature_names(self):
        self.feature_names_out_ = ['ft_avg_line_price', 'ft_median_line_price', 'ft_avg_discount_ratio', 'ft_avg_price_delta', 'ft_avg_unit_price_deviation', 'ft_frac_high_risk', 'ft_frac_age_restricted', 'ft_max_price_delta', 'ft_mixed_age_flag', 'ft_missing_age_restriction', 'ft_new_product_ratio']
        self.feature_names_out_ += [f'ft_has_category_{cat}' for cat in self.all_categories]
    def _handle_price_representation(self, lines):
        if self.price_representation == 'binary': return (lines['price'] > 0).astype(int)
        if self.price_representation == 'normalized':
            price = lines['price']
            diff = price.max() - price.min()
            return (price - price.min()) / diff if diff != 0 else np.zeros(len(price))
        return lines['price']
    
    def _calculate_features(self, transaction_lines):
        if not isinstance(transaction_lines, (list, np.ndarray)) or len(transaction_lines) == 0: return [0] * len(self.feature_names_out_)
        lines = pd.DataFrame([line for line in transaction_lines if isinstance(line, dict)])
        if lines.empty: return [0] * len(self.feature_names_out_)
        
        prices = self._handle_price_representation(lines)
        avg_price = prices.mean()
        median_price = prices.median()
        
        if 'expected_price' in lines and 'price' in lines and lines['expected_price'].notna().any():
            price_delta = lines['price'] - lines['expected_price']
            avg_price_delta = np.nanmean(price_delta)
            max_price_delta = np.nanmax(price_delta)
        else:
            avg_price_delta = 0
            max_price_delta = 0

        avg_discount_ratio = np.nanmean(lines['discount_amount'] / lines['price']) if 'discount_amount' in lines and 'price' in lines and lines['price'].notna().any() else 0
        
        if 'price_per_unit' in lines and 'category' in lines and lines['price_per_unit'].notna().any():
            cat_means = lines.groupby('category')['price_per_unit'].transform('mean')
            avg_unit_price_deviation = (lines['price_per_unit'] - cat_means).abs().mean()
        else: 
            avg_unit_price_deviation = 0
            
        frac_high_risk = lines['category'].isin(self.high_risk_categories).mean() if 'category' in lines else 0
        frac_age_restricted = lines['age_restricted'].astype(float).mean() if 'age_restricted' in lines else 0
        mixed_age_flag = int(lines['age_restricted'].any() & ~lines['age_restricted'].all()) if 'age_restricted' in lines else 0
        
        if 'category' in lines and 'age_restricted' in lines: 
            risky_lines = lines[lines['category'].isin(self.high_risk_categories)]
            missing_age_restriction = risky_lines['age_restricted'].eq(0).mean() if not risky_lines.empty else 0
        else: 
            missing_age_restriction = 0
            
        if 'product_launch_date' in lines and lines['product_launch_date'].notna().any():
            days_since = (datetime.now() - pd.to_datetime(lines['product_launch_date'])).dt.days
            new_product_ratio = (days_since < self.new_product_days).mean()
        else: 
            new_product_ratio = 0
            
        present = set(lines['category'].unique()) if 'category' in lines else set()
        category_features = [int(cat in present) for cat in self.all_categories]
        
        features = [avg_price, median_price, avg_discount_ratio, avg_price_delta, avg_unit_price_deviation, frac_high_risk, frac_age_restricted, max_price_delta, mixed_age_flag, missing_age_restriction, new_product_ratio] + category_features
        return [0 if pd.isna(x) else x for x in features]

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if isinstance(X, pd.Series): X_df = X.to_frame()
        elif isinstance(X, np.ndarray): X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else: X_df = X.copy()
        stats = X_df.iloc[:, 0].apply(self._calculate_features)
        df_transformed = pd.DataFrame(stats.tolist(), index=X_df.index, columns=self.feature_names_out_).fillna(0)
        return df_transformed
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

# ==============================================================================
# Runtime Patches
# ==============================================================================

pipeline_module = types.ModuleType("pipeline")
pipeline_module.TimeFeatureGenerator = TimeFeatureGenerator
pipeline_module.FeedbackBinnerOHE = FeedbackBinnerOHE
pipeline_module.EnhancedTransactionLineFeatures = EnhancedTransactionLineFeatures
sys.modules["pipeline"] = pipeline_module
sys.modules["pipeline.data_pipeline"] = pipeline_module

_original_ft_transform = FunctionTransformer.transform
def _robust_ft_transform(self, X):
    res = _original_ft_transform(self, X)
    if hasattr(res, 'ndim') and res.ndim == 1:
        res = res.reshape(-1, 1)
    if isinstance(res, float):
        res = np.array([[res]])
    return res
FunctionTransformer.transform = _robust_ft_transform

# --- Configuration ---
MODEL_DIR = './weights'

# --- Load Models, Preprocessor, and SHAP Explainer ---
try:
    classifier_list = glob.glob(os.path.join(MODEL_DIR, '*_model_*.cbm'))
    if not classifier_list: raise FileNotFoundError("No classification model found.")
    latest_classifier_path = max(classifier_list, key=os.path.getctime)
    classifier_model = CatBoostClassifier()
    classifier_model.load_model(latest_classifier_path)

    regressor_list = glob.glob(os.path.join(MODEL_DIR, '*_regressor_*.cbm'))
    if not regressor_list: raise FileNotFoundError("No regression model found.")
    latest_regressor_path = max(regressor_list, key=os.path.getctime)
    regressor_model = CatBoostRegressor()
    regressor_model.load_model(latest_regressor_path)

    preprocessor_list = glob.glob(os.path.join(MODEL_DIR, 'preprocessor*.joblib'))
    if not preprocessor_list: raise FileNotFoundError("No preprocessor found.")
    latest_preprocessor_path = max(preprocessor_list, key=os.path.getctime)
    preprocessor = joblib.load(latest_preprocessor_path)

    # Create the SHAP explainer at startup
    shap_explainer = shap.TreeExplainer(classifier_model)
    print("Models, preprocessor, and SHAP explainer loaded successfully.")

except FileNotFoundError as e:
    raise RuntimeError(f"Could not load model artifacts: {e}")
except Exception as e:
    raise RuntimeError(f"An unexpected error occurred while loading artifacts: {e}")

# --- API Data Models ---

class FeatureImportance(BaseModel):
    feature: str
    value: float
    shap_value: float

class Explanation(BaseModel):
    human_readable_reason: Optional[str] = None
    feature_importance: Optional[List[FeatureImportance]] = None
    offending_products: Optional[List[str]] = None

class FraudPrediction(BaseModel):
    version: str
    is_fraud: bool
    fraud_proba: Optional[float] = Field(None, ge=0, le=1)
    estimated_damage: Optional[float] = None
    explanation: Optional[Explanation] = None

    @field_validator("version")
    @classmethod
    def validate_semantic_version(cls, v):
        parts = v.split(".")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Version must follow semantic versioning (x.y.z)")
        return v

class TransactionLine(BaseModel):
    id: int
    product_id: UUID
    timestamp: datetime
    pieces_or_weight: float
    sales_price: float
    was_voided: bool
    camera_product_similar: bool
    camera_certainty: float
    category: Optional[str] = 'UNKNOWN'
    discount_amount: Optional[float] = 0.0
    expected_price: Optional[float] = None
    price_per_unit: Optional[float] = None
    age_restricted: Optional[bool] = False
    product_launch_date: Optional[datetime] = None
    price: Optional[float] = None

class TransactionHeader(BaseModel):
    store_id: UUID
    cash_desk: int
    transaction_start: datetime
    transaction_end: datetime
    total_amount: float
    payment_medium: str
    customer_feedback: Optional[int] = None
    location: Optional[str] = 'UNKNOWN'
    opening_date: Optional[str] = 'UNKNOWN'
    n_lines: Optional[int] = None

class FraudPredictionRequest(BaseModel):
    transaction_header: TransactionHeader
    transaction_lines: List[TransactionLine]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}

# --- Feature Name Mapping for Human-Readable Explanations ---
feature_name_mapping = {
    'num_log_center__total_amount': 'Total Amount',
    'num_log_center__n_lines': 'Number of Items',
    'onehot_cat__payment_medium_CASH': 'Payment Method (Cash)',
    'time_features__ft_hour_cos': 'Time of Day (Late Night/Early Morning)',
    'transaction_lines_stats__ft_avg_line_price': 'Average Price per Item',
    'transaction_lines_stats__ft_has_category_SNACKS': 'Contains Snacks',
    'transaction_lines_stats__ft_has_category_CONVENIENCE': 'Contains Convenience Items',
    'transaction_lines_stats__ft_has_category_FRUITS_VEGETABLES_PIECES': 'Contains Fruit/Veg (by piece)',
    'time_features__ft_duration_seconds_log': 'Transaction Duration',
    'transaction_lines_stats__ft_frac_high_risk': 'Fraction of High-Risk Items'
}


# --- FastAPI Application ---
app = FastAPI(
    title="SCO Fraud REST API",
    description="A REST API for real-time Self-Checkout (SCO) fraud detection.",
    version="0.1.1"
)

API_VERSION = "0.1.1"
CLASSIFICATION_THRESHOLD = 0.47

@app.get("/", tags=["Health Check"])
async def health_check():
    return {"status": "healthy", "version": API_VERSION}

@app.post("/fraud-prediction", response_model=FraudPrediction, tags=["Fraud Prediction"])
async def predict_fraud(request: FraudPredictionRequest):
    try:
        # --- Data Preparation ---
        header_data = request.transaction_header.dict()
        header_data['n_lines'] = len(request.transaction_lines)
        header_data['transaction_lines_details'] = [line.dict() for line in request.transaction_lines]
        input_df = pd.DataFrame([header_data])

        # --- Preprocessing ---
        preprocessed_features = preprocessor.transform(input_df)
        feature_names = preprocessor.get_feature_names_out()
        preprocessed_features_df = pd.DataFrame(preprocessed_features, columns=feature_names)

        # --- Classification Prediction ---
        fraud_probability = classifier_model.predict_proba(preprocessed_features)[0][1]
        is_fraud = bool(fraud_probability >= CLASSIFICATION_THRESHOLD)

        # --- SHAP Value Explanation (runs for every request) ---
        shap_values = shap_explainer.shap_values(preprocessed_features_df)
        
        if isinstance(shap_values, list):
            shap_values_for_fraud = shap_values[1][0]
        else:
            shap_values_for_fraud = shap_values[0]
        
        feature_impacts = []
        for i, feature_name in enumerate(feature_names):
            feature_value = preprocessed_features_df.iloc[0, i]
            shap_value = shap_values_for_fraud[i]
            
            # Only include features with a noticeable impact
            if abs(shap_value) > 0.01:
                descriptive_name = feature_name_mapping.get(feature_name, feature_name)
                feature_impacts.append({
                    "feature": descriptive_name,
                    "value": feature_value,
                    "shap_value": shap_value
                })

        # Sort by absolute SHAP value for the detailed list
        feature_importance_list = sorted(feature_impacts, key=lambda x: abs(x['shap_value']), reverse=True)

        # --- Initialize response variables ---
        estimated_damage = 0.0
        human_readable_reason = ""
        
        if is_fraud:
            # --- Regression Prediction ---
            predicted_raw_damage = regressor_model.predict(preprocessed_features)
            estimated_damage = round(max(0, predicted_raw_damage[0]), 2)

            # Create a human-readable reason from the top 3 features pushing towards fraud
            top_fraud_features_desc = []
            for item in feature_importance_list[:3]:
                desc = item['feature']
                if 'Price' in desc and item['value'] < 1:
                    desc += " (very low)"
                elif 'Amount' in desc and item['value'] < 1:
                     desc += " (very low)"
                elif 'Cash' in desc and item['value'] == 1:
                    desc = "Payment with Cash"
                elif 'Time of Day' in desc and item['value'] > 0.5:
                     desc = "Late Night Transaction"
                top_fraud_features_desc.append(desc)
            human_readable_reason = "High fraud risk detected. Key factors: " + "; ".join(top_fraud_features_desc) + "."

        else: # NOT FRAUD
            # Sort by SHAP value to find the most protective features (most negative)
            protective_features_sorted = sorted(feature_impacts, key=lambda x: x['shap_value'])
            
            top_protective_features_desc = []
            for item in protective_features_sorted[:3]:
                desc = item['feature']
                # Add context for non-fraudulent reasons
                if 'Cash' in desc and item['value'] == 0:
                    desc = "Payment with Card"
                elif 'Time of Day' in desc and item['value'] < -0.5:
                     desc = "Normal Business Hours"
                elif 'Price' in desc and item['value'] > 10:
                     desc += " (high)"
                top_protective_features_desc.append(desc)
            
            human_readable_reason = "Low fraud risk. Key protective factors: " + "; ".join(top_protective_features_desc) + "."

        # --- Construct Explanation Object ---
        explanation = Explanation(
            human_readable_reason=human_readable_reason,
            feature_importance=[FeatureImportance(**item) for item in feature_importance_list],
            offending_products=[]
        )

        # --- Response Formatting ---
        response = FraudPrediction(
            version=API_VERSION,
            is_fraud=is_fraud,
            fraud_proba=round(fraud_probability, 4),
            estimated_damage=estimated_damage,
            explanation=explanation,
        )

        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")
