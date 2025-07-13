# Standardbibliotheken
from pathlib import Path
from datetime import datetime

# Drittanbieter-Bibliotheken
import numpy as np
import pandas as pd

# Scikit-Learn: Preprocessing, Pipelines, Transformer, etc.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# --- Custom Transformer Classes (Copied from data_pipeline.py) ---

class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df_transformed = pd.DataFrame(index=X.index)
        transaction_start = pd.to_datetime(X['transaction_start'])
        transaction_end = pd.to_datetime(X['transaction_end'])

        df_transformed['ft_month'] = transaction_start.dt.month
        df_transformed['ft_day'] = transaction_start.dt.day
        duration = transaction_end - transaction_start
        ft_duration_seconds_val = duration.dt.total_seconds()

        df_transformed['ft_hour'] = transaction_start.dt.hour
        df_transformed['ft_minute'] = transaction_start.dt.minute
        df_transformed['ft_second'] = transaction_start.dt.second
        df_transformed['ft_day_of_week'] = transaction_start.dt.dayofweek
        df_transformed['ft_day_of_year'] = transaction_start.dt.dayofyear
        df_transformed['ft_week_of_year'] = transaction_start.dt.isocalendar().week.astype(int)
        df_transformed['ft_quarter'] = transaction_start.dt.quarter

        df_transformed['ft_is_weekend'] = df_transformed['ft_day_of_week'].isin([5, 6]).astype(int)
        df_transformed['ft_is_outside_normal_hours'] = \
            ((df_transformed['ft_hour'] < 8) | (df_transformed['ft_hour'] >= 23)).astype(int)

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

        df_transformed['ft_inter_weekend_outside_hours'] = \
            df_transformed['ft_is_weekend'] * df_transformed['ft_is_outside_normal_hours']
        df_transformed['ft_inter_hour_cos_x_dow_cos'] = \
            df_transformed['ft_hour_cos'] * df_transformed['ft_day_of_week_cos']
            
        self.feature_names_out_ = [
            'ft_month', 'ft_day', 'ft_hour', 'ft_minute', 'ft_second', 
            'ft_day_of_week', 'ft_day_of_year', 'ft_week_of_year', 'ft_quarter',
            'ft_is_weekend', 'ft_is_outside_normal_hours', 'ft_month_sin', 'ft_month_cos',
            'ft_hour_sin', 'ft_hour_cos', 'ft_day_of_week_sin', 'ft_day_of_week_cos',
            'ft_day_of_year_sin', 'ft_day_of_year_cos', 'ft_duration_seconds_log',
            'ft_inter_weekend_outside_hours', 'ft_inter_hour_cos_x_dow_cos'
        ]
        
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
        if pd.isnull(feedback_value):
            return 'feedback_null'
        elif feedback_value == 10.0:
            return 'feedback_10'
        else:
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
        self.all_categories = [
            'TOBACCO', 'ALCOHOL', 'PERSONAL_CARE', 'LONG_SHELF_LIFE', 'FROZEN_GOODS', 
            'HOUSEHOLD', 'BEVERAGES', 'BAKERY', 'FRUITS_VEGETABLES_PIECES', 'DAIRY', 
            'CONVENIENCE', 'FRUITS_VEGETABLES', 'SNACKS'
        ]
        self.high_risk_categories = {'SNACKS', 'FRUITS_VEGETABLES_PIECES', 'CONVENIENCE'}
        self.new_product_days = 30
        self._initialize_feature_names()

    def _initialize_feature_names(self):
        self.feature_names_out_ = [
            'ft_avg_line_price', 'ft_median_line_price', 'ft_avg_discount_ratio',
            'ft_avg_price_delta', 'ft_avg_unit_price_deviation', 'ft_frac_high_risk',
            'ft_frac_age_restricted', 'ft_max_price_delta', 'ft_mixed_age_flag',
            'ft_missing_age_restriction', 'ft_new_product_ratio'
        ]
        self.feature_names_out_ += [f'ft_has_category_{cat}' for cat in self.all_categories]

    def _handle_price_representation(self, lines):
        if self.price_representation == 'binary':
            return (lines['price'] > 0).astype(int)
        elif self.price_representation == 'normalized':
            price = lines['price']
            diff = price.max() - price.min()
            return (price - price.min()) / diff if diff != 0 else np.zeros(len(price))
        return lines['price']

    def _calculate_features(self, transaction_lines):
        if not isinstance(transaction_lines, (list, np.ndarray)) or len(transaction_lines) == 0:
            return [0] * len(self.feature_names_out_)
        lines = pd.DataFrame([line for line in transaction_lines if isinstance(line, dict)])
        if lines.empty:
            return [0] * len(self.feature_names_out_)

        prices = self._handle_price_representation(lines)
        avg_price = prices.mean()
        median_price = prices.median()

        if 'discount_amount' in lines and 'price' in lines:
            avg_discount_ratio = np.nanmean(lines['discount_amount'] / lines['price'])
        else:
            avg_discount_ratio = 0

        if 'expected_price' in lines and 'price' in lines:
            avg_price_delta = np.nanmean(lines['price'] - lines['expected_price'])
        else:
            avg_price_delta = 0

        if 'price_per_unit' in lines and 'category' in lines:
            cat_means = lines.groupby('category')['price_per_unit'].transform('mean')
            deviation = (lines['price_per_unit'] - cat_means).abs()
            avg_unit_price_deviation = deviation.mean()
        else:
            avg_unit_price_deviation = 0

        frac_high_risk = lines['category'].isin(self.high_risk_categories).mean() if 'category' in lines else 0
        frac_age_restricted = lines['age_restricted'].astype(float).mean() if 'age_restricted' in lines else 0

        if 'expected_price' in lines and 'price' in lines:
            max_price_delta = np.nanmax(lines['price'] - lines['expected_price'])
        else:
            max_price_delta = 0

        if 'age_restricted' in lines:
            age_flags = lines['age_restricted'].astype(bool)
            mixed_age_flag = int(age_flags.any() & ~age_flags.all())
        else:
            mixed_age_flag = 0

        if 'category' in lines and 'age_restricted' in lines:
            risky = lines['category'].isin(self.high_risk_categories)
            missing_age_restriction = lines[risky]['age_restricted'].eq(0).mean()
        else:
            missing_age_restriction = 0

        if 'product_launch_date' in lines:
            days_since = (datetime.now() - pd.to_datetime(lines['product_launch_date'])).dt.days
            new_product_ratio = (days_since < self.new_product_days).mean()
        else:
            new_product_ratio = 0

        if 'category' in lines:
            present = set(lines['category'].unique())
            category_features = [int(cat in present) for cat in self.all_categories]
        else:
            category_features = [0] * len(self.all_categories)

        features = [
            avg_price, median_price, avg_discount_ratio, avg_price_delta,
            avg_unit_price_deviation, frac_high_risk, frac_age_restricted,
            max_price_delta, mixed_age_flag, missing_age_restriction, new_product_ratio
        ] + category_features

        return [0 if pd.isna(x) else x for x in features]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else:
            X_df = X.copy()

        stats = X_df.iloc[:, 0].apply(self._calculate_features)
        
        df_transformed = pd.DataFrame(stats.tolist(), index=X_df.index, columns=self.feature_names_out_)
        df_transformed = df_transformed.fillna(0)

        if self.normalize:
            numerical = [f for f in self.feature_names_out_ if not f.startswith('ft_has_category_')]
            scaler = MinMaxScaler()
            df_transformed[numerical] = scaler.fit_transform(df_transformed[numerical])

        return df_transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_


def create_preprocessor():
    """Defines and returns the ColumnTransformer for feature processing."""
    
    numerical_features_log_center = ['total_amount', 'n_lines']
    categorical_features_onehot = ['cash_desk', 'payment_medium', 'location', 'opening_date']
    datetime_features_input_for_time_generator = ['transaction_start', 'transaction_end']
    feedback_feature_column = ['customer_feedback']
    transaction_lines_feature_column = ['transaction_lines_details']

    numerical_log_center_transformer = Pipeline(steps=[
        ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')),
        ('mean_centering', StandardScaler(with_std=False))
    ])

    onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int)

    time_feature_processing_pipeline = Pipeline(steps=[
        ('time_generator', TimeFeatureGenerator()),
        ('scaler', StandardScaler())
    ])
    
    feedback_transformer = FeedbackBinnerOHE()
    transaction_line_stats_transformer = EnhancedTransactionLineFeatures()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_log_center', numerical_log_center_transformer, numerical_features_log_center),
            ('onehot_cat', onehot_transformer, categorical_features_onehot),
            ('time_features', time_feature_processing_pipeline, datetime_features_input_for_time_generator),
            ('feedback_processing', feedback_transformer, feedback_feature_column),
            ('transaction_lines_stats', transaction_line_stats_transformer, transaction_lines_feature_column)
        ],
        remainder='drop'
    )
    return preprocessor

def main():
    """
    Main function to run the prediction pipeline.
    1. Fits the preprocessor on training data.
    2. Loads and aggregates test data.
    3. Transforms test data using the fitted preprocessor.
    4. Saves the processed test data.
    """
    # --- 1. Define Paths and Load Training Data to FIT the preprocessor ---
    # This ensures that encoding and scaling are consistent with the model training phase.
    
    # Get the directory of the current file
    module_dir = Path(__file__).resolve().parent
    base_data_path = (module_dir / '../../data/').resolve()

    print("Loading training data to fit the preprocessor...")
    try:
        path_to_train_agg = base_data_path / 'aggregated_train.parquet'
        transactions_train_full = pd.read_parquet(path_to_train_agg)
    except FileNotFoundError:
        print(f"Error: The training file was not found at {path_to_train_agg}")
        print("Please ensure 'aggregated_train.parquet' exists to fit the preprocessor.")
        return

    # Filter out 'UNKNOWN' labels and define feature matrix X_train
    transactions_train = transactions_train_full[transactions_train_full['label'] != 'UNKNOWN'].copy()
    X_train = transactions_train.drop('label', axis=1)

    # --- 2. Create and FIT the Preprocessor ---
    preprocessor = create_preprocessor()
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)
    print("Preprocessor fitting complete.")

    # --- 3. Load Raw TEST Data ---
    print("\nLoading raw test data for prediction...")
    products_source = pd.read_csv(base_data_path / 'products.csv')
    stores_source = pd.read_csv(base_data_path / 'stores.csv')
    transaction_lines_test_source = pd.read_parquet(base_data_path / 'transaction_lines_test_1.parquet')
    transactions_test_source = pd.read_parquet(base_data_path / 'transactions_test_1.parquet')

    # --- 4. Aggregate TEST Data ---
    print("Aggregating test data...")
    # Enrich transactions with store data
    transactions_enriched = pd.merge(
        transactions_test_source,
        stores_source,
        left_on='store_id',
        right_on='id',
        how='left',
        suffixes=('', '_store')
    ).drop(columns=['id_store'], errors='ignore')

    # Enrich transaction lines with product data
    lines_enriched = pd.merge(
        transaction_lines_test_source,
        products_source,
        left_on='product_id',
        right_on='id',
        how='left',
        suffixes=('', '_product')
    ).drop(columns=['id_product'], errors='ignore')

    # Aggregate lines by transaction
    aggregated_lines = lines_enriched.groupby('transaction_id').apply(
        lambda x: x.to_dict('records')
    ).reset_index(name='transaction_lines_details')

    # Merge aggregated lines back to transactions
    X_test = pd.merge(
        transactions_enriched,
        aggregated_lines,
        left_on='id',
        right_on='transaction_id',
        how='left'
    ).drop(columns=['transaction_id'])
    
    print("Test data aggregation complete.")

    # --- 5. TRANSFORM the Test Data ---
    print("Transforming test data using the fitted preprocessor...")
    X_test_processed_array = preprocessor.transform(X_test)

    # Convert to DataFrame with correct feature names
    processed_feature_names = preprocessor.get_feature_names_out()
    X_test_processed = pd.DataFrame(
        X_test_processed_array,
        columns=processed_feature_names,
        index=X_test.index
    )
    
    print("Test data transformation complete.")
    print(f"Shape of processed test data: {X_test_processed.shape}")
    print("\nFirst 5 rows of processed test data:")
    print(X_test_processed.head())

    # --- 6. Save the Processed Data ---
    output_path = base_data_path / 'processed_test_data.parquet'
    X_test_processed.to_parquet(output_path)
    print(f"\nProcessed test data saved to: {output_path}")

def process_unlabeled_data(preprocessor, base_data_path):
    """
    Lädt, aggregiert und transformiert die unbeschrifteten Testdaten.

    Args:
        preprocessor: Ein bereits trainierter scikit-learn Preprocessor.

    Returns:
        tuple: Ein Tupel mit (X_test_processed, X_test_full)
               - X_test_processed: Der transformierte DataFrame für das Modell.
               - X_test_full: Der ursprüngliche, aggregierte DataFrame mit den IDs.
    """
    print("Starte die Verarbeitung der Testdaten mit der Pipeline...")

    # Pfade zu den Rohdaten definieren
    # base_data_path = Path(__file__).resolve().parent.parent / 'data'

    # Rohdaten laden
    print("Lade rohe Testdaten...")
    products_source = pd.read_csv(base_data_path / 'products.csv')
    stores_source = pd.read_csv(base_data_path / 'stores.csv')
    transaction_lines_test = pd.read_parquet(base_data_path / 'transaction_lines_test_1.parquet')
    transactions_test = pd.read_parquet(base_data_path / 'transactions_test_1.parquet')

    # Daten aggregieren
    print("Aggregiere Testdaten...")
    transactions_enriched = pd.merge(
        transactions_test, stores_source, left_on='store_id', right_on='id',
        how='left', suffixes=('', '_store')
    ).drop(columns=['id_store'], errors='ignore')

    lines_enriched = pd.merge(
        transaction_lines_test, products_source, left_on='product_id', right_on='id',
        how='left', suffixes=('', '_product')
    ).drop(columns=['id_product'], errors='ignore')

    aggregated_lines = lines_enriched.groupby('transaction_id').apply(
        lambda x: x.to_dict('records')
    ).reset_index(name='transaction_lines_details')

    X_test_full = pd.merge(
        transactions_enriched, aggregated_lines, left_on='id', right_on='transaction_id',
        how='left'
    ).drop(columns=['transaction_id'])

    # Daten transformieren
    print("Transformiere Testdaten mit dem Preprocessor...")
    X_test_processed_array = preprocessor.transform(X_test_full)
    processed_feature_names = preprocessor.get_feature_names_out()

    X_test_processed = pd.DataFrame(
        X_test_processed_array,
        columns=processed_feature_names,
        index=X_test_full.index
    )

    print("✅ Datenverarbeitung abgeschlossen.")
    return X_test_processed, X_test_full


if __name__ == '__main__':
    main()