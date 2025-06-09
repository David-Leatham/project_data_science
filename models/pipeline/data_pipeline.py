import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path

# Get the directory of the current file (your imported module)
# __file__ is a special variable that holds the path to the current script
module_dir = Path(__file__).resolve().parent

# Construct the path to your data file relative to the module's directory
data_file_path = module_dir / '../../data/aggregated_train.parquet'

# It's a good practice to resolve the path to get the absolute path
# and ensure it's what you expect, especially with '..'
data_file_path = data_file_path.resolve()

transactions_for_pipeline = pd.read_parquet(data_file_path)
transactions_for_pipeline = transactions_for_pipeline[transactions_for_pipeline['label'] != 'UNKNOWN']

class TimeFeatureGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass # Keine Initialisierungsparameter notwendig

    def fit(self, X, y=None):
        # Dieser Transformer lernt keine Parameter aus den Daten.
        return self

    def transform(self, X, y=None):
        # Eingabe X: DataFrame mit 'transaction_start' und 'transaction_end'
        df_transformed = pd.DataFrame(index=X.index) # Erstellt eine Kopie mit dem ursprünglichen Index

        transaction_start = pd.to_datetime(X['transaction_start'])
        transaction_end = pd.to_datetime(X['transaction_end'])

        # --- Basis-Zeitmerkmale ---
        df_transformed['ft_month'] = transaction_start.dt.month
        df_transformed['ft_day'] = transaction_start.dt.day
        
        duration = transaction_end - transaction_start
        ft_duration_seconds_val = duration.dt.total_seconds()

        # --- Zyklische Zeit-basierte Features (aus transaction_start) ---
        df_transformed['ft_hour'] = transaction_start.dt.hour
        df_transformed['ft_minute'] = transaction_start.dt.minute
        df_transformed['ft_second'] = transaction_start.dt.second
        df_transformed['ft_day_of_week'] = transaction_start.dt.dayofweek # Montag=0, Sonntag=6
        df_transformed['ft_day_of_year'] = transaction_start.dt.dayofyear
        df_transformed['ft_week_of_year'] = transaction_start.dt.isocalendar().week.astype(int) # .week kann nullable sein
        df_transformed['ft_quarter'] = transaction_start.dt.quarter

        # --- Binäre/ereignisbasierte Features ---
        df_transformed['ft_is_weekend'] = df_transformed['ft_day_of_week'].isin([5, 6]).astype(int) # Samstag=5, Sonntag=6
        # Annahme: Normale Stunden sind 08:00 - 22:59 Uhr
        df_transformed['ft_is_outside_normal_hours'] = \
            ((df_transformed['ft_hour'] < 8) | (df_transformed['ft_hour'] >= 23)).astype(int)

        # --- Kodierung zyklischer Features (Sinus/Kosinus Transformation) ---
        df_transformed['ft_month_sin'] = np.sin(2 * np.pi * df_transformed['ft_month']/12)
        df_transformed['ft_month_cos'] = np.cos(2 * np.pi * df_transformed['ft_month']/12)

        df_transformed['ft_hour_sin'] = np.sin(2 * np.pi * df_transformed['ft_hour']/24)
        df_transformed['ft_hour_cos'] = np.cos(2 * np.pi * df_transformed['ft_hour']/24)

        df_transformed['ft_day_of_week_sin'] = np.sin(2 * np.pi * df_transformed['ft_day_of_week']/7)
        df_transformed['ft_day_of_week_cos'] = np.cos(2 * np.pi * df_transformed['ft_day_of_week']/7)

        days_in_year = transaction_start.dt.is_leap_year.map({True: 366, False: 365})
        df_transformed['ft_day_of_year_sin'] = np.sin(2 * np.pi * df_transformed['ft_day_of_year']/days_in_year)
        df_transformed['ft_day_of_year_cos'] = np.cos(2 * np.pi * df_transformed['ft_day_of_year']/days_in_year)
        
        # --- Transformation für ft_duration_seconds ---
        # Log-Transformation (np.log1p für Stabilität bei Werten nahe 0)
        df_transformed['ft_duration_seconds_log'] = np.log1p(ft_duration_seconds_val)

        # --- Interaktionsmerkmale ---
        df_transformed['ft_inter_weekend_outside_hours'] = \
            df_transformed['ft_is_weekend'] * df_transformed['ft_is_outside_normal_hours']
        df_transformed['ft_inter_hour_cos_x_dow_cos'] = \
            df_transformed['ft_hour_cos'] * df_transformed['ft_day_of_week_cos']
            
        # Speichert die Namen der generierten Features für get_feature_names_out.
        # Die Reihenfolge muss der Erstellung der Spalten entsprechen.
        self.feature_names_out_ = [
            'ft_month', 'ft_day', 
            'ft_hour', 'ft_minute', 'ft_second', 
            'ft_day_of_week', 'ft_day_of_year', 'ft_week_of_year', 'ft_quarter',
            'ft_is_weekend', 'ft_is_outside_normal_hours',
            'ft_month_sin', 'ft_month_cos',
            'ft_hour_sin', 'ft_hour_cos',
            'ft_day_of_week_sin', 'ft_day_of_week_cos',
            'ft_day_of_year_sin', 'ft_day_of_year_cos',
            'ft_duration_seconds_log', # Log-transformierte Dauer
            'ft_inter_weekend_outside_hours',
            'ft_inter_hour_cos_x_dow_cos'
        ]
        
        return df_transformed[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        # Gibt die Namen der erzeugten Features zurück.
        return self.feature_names_out_
    
class FeedbackBinnerOHE(BaseEstimator, TransformerMixin):
    """
    Benutzerdefinierter Transformer für die 'customer_feedback'-Spalte.
    1. Kategorisiert Feedback-Werte ('feedback_null', 'feedback_10', 'feedback_other').
    2. Führt One-Hot-Encoding für diese Kategorien durch (Präfix 'ft_').
    """
    def __init__(self):
        # Feste Kategorien und daraus resultierende Feature-Namen.
        self.categories_ = ['feedback_null', 'feedback_10', 'feedback_other']
        self.feature_names_out_ = [f"ft_{cat}" for cat in self.categories_]

    def fit(self, X, y=None):
        # Keine Anpassung an die Daten nötig, da Kategorien vordefiniert sind.
        return self

    def _categorize_feedback(self, feedback_value):
        # Interne Hilfsfunktion zur Kategorisierung.
        if pd.isnull(feedback_value):
            return 'feedback_null'
        elif feedback_value == 10.0:
            return 'feedback_10'
        else:  # Alle anderen nicht-null Werte (z.B. 1.0 bis 9.0)
            return 'feedback_other'

    def transform(self, X, y=None):
        # Eingabe X: DataFrame mit der 'customer_feedback'-Spalte.
        feedback_series = X.iloc[:, 0] # Extrahiert die relevante Spalte als Series.
        
        # Anwenden der Kategorisierungsfunktion.
        categorized_feedback = feedback_series.apply(self._categorize_feedback)
        
        # One-Hot-Encoding.
        # pd.Categorical stellt sicher, dass alle definierten Kategorien als Spalten erscheinen.
        categorized_feedback_type = pd.Categorical(categorized_feedback, categories=self.categories_)
        dummies_df = pd.get_dummies(categorized_feedback_type, prefix='ft', dtype=int)
        
        return dummies_df[self.feature_names_out_] # Gibt DataFrame mit Dummy-Variablen zurück.

    def get_feature_names_out(self, input_features=None):
        # Gibt die Namen der erzeugten Features zurück.
        return self.feature_names_out_
    
class TransactionLineFeatures(BaseEstimator, TransformerMixin):
    """
    Custom transformer for 'transaction_lines_details'.
    Calculates average and median 'price' from transaction lines.
    """
    def __init__(self):
        self.feature_names_out_ = ['ft_avg_line_price', 'ft_median_line_price']

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def _calculate_stats(self, transaction_lines):
        # Helper function to calculate stats for a single transaction's lines
        if isinstance(transaction_lines, (list, np.ndarray)) and len(transaction_lines) > 0:
            prices = [line.get('price') for line in transaction_lines if isinstance(line, dict) and line.get('price') is not None]
            if prices:
                return np.mean(prices), np.median(prices)
        return np.nan, np.nan # Return NaN if no valid prices are found or input is not as expected

    def transform(self, X, y=None):
        # X is expected to be a DataFrame with the 'transaction_lines_details' column
        # Ensure X is a DataFrame, as ColumnTransformer might pass a Series
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, np.ndarray): # If it's a numpy array, convert to DataFrame
            X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else:
            X_df = X.copy()


        # Apply the calculation row-wise
        stats = X_df.iloc[:, 0].apply(self._calculate_stats)

        # Create a DataFrame with the new features
        df_transformed = pd.DataFrame(stats.tolist(), index=X_df.index, columns=self.feature_names_out_)

        # Handle potential NaNs from calculations by filling them, e.g., with 0 or mean/median of the new columns
        # For simplicity, let's fill with 0 here, but you might choose a different strategy
        df_transformed.fillna(0, inplace=True)

        return df_transformed

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
    
# Sicherstellen, dass 'label' kategorial ist.
if not pd.api.types.is_categorical_dtype(transactions_for_pipeline['label']):
    transactions_for_pipeline['label'] = transactions_for_pipeline['label'].astype('category')

# Mappt 'label' auf numerische Werte (FRAUD: 1, NORMAL: 0).
y = transactions_for_pipeline['label'].map({'FRAUD': 1, 'NORMAL': 0})

# Prüft auf Mapping-Fehler (NaNs) und löst einen Fehler aus, falls welche vorhanden sind.
if y.isnull().any():
    raise ValueError(
        "Nicht alle Werte in 'label' konnten gemappt werden. "
        "Überprüfe die Kategorien (erwartet: 'FRAUD', 'NORMAL') und das Mapping."
    )

# Extract the raw, untransformed damage values
# Ensure it's aligned with y and X by using the same DataFrame and index
Z_damage = transactions_for_pipeline['damage'].copy()

# --- Handle NaN values in Z_damage by setting them to 0 ---
Z_damage.fillna(0, inplace=True)

# Definiert die Merkmalsmatrix X durch Entfernen der 'label'-Spalte.
X = transactions_for_pipeline.drop('label', axis=1)

# Spaltengruppen für Transformationen
numerical_features_log_center = ['total_amount', 'n_lines']
categorical_features_onehot = ['cash_desk', 'payment_medium', 'location', 'opening_date']
datetime_features_input_for_time_generator = ['transaction_start', 'transaction_end']
feedback_feature_column = ['customer_feedback']
transaction_lines_feature_column = ['transaction_lines_details']

# Numerische Transformation: Logarithmieren (log1p) und Mittelwertzentrierung.
numerical_log_center_transformer = Pipeline(steps=[
    ('log1p', FunctionTransformer(np.log1p, validate=False, feature_names_out='one-to-one')),
    ('mean_centering', StandardScaler(with_std=False)) # Nur Mittelwertzentrierung.
])

# Kategoriale Transformation: One-Hot-Encoding.
onehot_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int) # Ignoriert unbekannte Kategorien, gibt dichte Arrays aus.

# Zeit-Features: Generierung und Skalierung.
time_feature_processing_pipeline = Pipeline(steps=[
    ('time_generator', TimeFeatureGenerator()),   # Erzeugt Zeitmerkmale.
    ('scaler', StandardScaler())                 # Skaliert generierte Zeitmerkmale.
])

# Feedback-Features: Kategorisierung und One-Hot-Encoding.
feedback_transformer = FeedbackBinnerOHE()

transaction_line_stats_transformer = TransactionLineFeatures()

# Zusammenbau des ColumnTransformers
preprocessor = ColumnTransformer(
    transformers=[
        # (Name, Transformer-Objekt, anzuwendende Spalten)
        ('num_log_center', numerical_log_center_transformer, numerical_features_log_center),
        ('onehot_cat', onehot_transformer, categorical_features_onehot),
        ('time_features', time_feature_processing_pipeline, datetime_features_input_for_time_generator),
        ('feedback_processing', feedback_transformer, feedback_feature_column),
        ('transaction_lines_stats', transaction_line_stats_transformer, transaction_lines_feature_column)
    ],
    remainder='drop'  # Nicht genannte Spalten werden verworfen ('passthrough' als Alternative).
)