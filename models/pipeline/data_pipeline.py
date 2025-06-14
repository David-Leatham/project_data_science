'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
'''
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
    
class TransactionLineFeatures_Bsp(BaseEstimator, TransformerMixin):
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
    

class TransactionLineFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_out_ = [
            'ft_avg_line_price',               # Durchschnittlicher Listenpreis pro Transaktion
            'ft_median_line_price',            # Median-Preis pro Transaktion
            'ft_avg_discount_ratio',           # Durchschnittlicher Rabattanteil pro Transaktion
            'ft_avg_price_delta',              # Durchschnittliche Abweichung zwischen Preis und erwartetem Preis
            'ft_avg_unit_price_deviation',     # Durchschnittliche Abweichung des Einzelpreises vom Kategoriendurchschnitt
            'ft_frac_high_risk',               # Anteil der Hochrisiko-Artikel pro Transaktion
            'ft_frac_voided',                  # Anteil der stornierten Artikel pro Transaktion
            'ft_frac_age_restricted',          # Anteil der altersbeschränkten Artikel pro Transaktion
            'ft_frac_zero_weight',             # Anteil der Artikel mit Gewicht = 0 (unplausibel)
            'ft_avg_popularity_deviation'      # Durchschnittliche Abweichung der Popularität vom Transaktionsdurchschnitt
        ]
        # High-Risk-Kategorien (kann angepasst werden)
        self.high_risk_categories = {'SNACKS', 'FRUITS_VEGETABLES_PIECES', 'CONVENIENCE'}
        # Liste der gewünschten Kategorien (One-Hot-Features)
        self.categories_ = [
            'TOBACCO', 'ALCOHOL', 'PERSONAL_CARE', 'LONG_SHELF_LIFE',
            'FROZEN_GOODS', 'HOUSEHOLD', 'BEVERAGES', 'BAKERY',
            'FRUITS_VEGETABLES_PIECES', 'DAIRY', 'CONVENIENCE',
            'FRUITS_VEGETABLES', 'SNACKS'
        ]

    def fit(self, X, y=None):
        # Optional: Kategorien aus den Daten ermitteln, hier aber explizit gesetzt
        return self

    def _calculate_features(self, transaction_lines):
        features = []
        if not isinstance(transaction_lines, (list, np.ndarray)) or len(transaction_lines) == 0:
            features = [np.nan] * len(self.feature_names_out_)
            features += [0] * len(self.categories_)
            return features

        lines = pd.DataFrame([line for line in transaction_lines if isinstance(line, dict)])
        n = len(lines)
        if n == 0:
            features = [np.nan] * len(self.feature_names_out_)
            features += [0] * len(self.categories_)
            return features

        # 1. Ursprüngliche Features berechnen
        prices = lines['price'].dropna()
        avg_price = prices.mean() if not prices.empty else np.nan
        median_price = prices.median() if not prices.empty else np.nan

        if 'discount_amount' in lines and 'price' in lines:
            discount_ratio = (lines['discount_amount'] / lines['price']).replace([np.inf, -np.inf], np.nan)
            avg_discount_ratio = discount_ratio.mean()
        else:
            avg_discount_ratio = np.nan

        if 'expected_price' in lines and 'price' in lines:
            price_delta = (lines['price'] - lines['expected_price']).replace([np.inf, -np.inf], np.nan)
            avg_price_delta = price_delta.mean()
        else:
            avg_price_delta = np.nan

        if 'price_per_unit' in lines and 'category' in lines:
            cat_means = lines.groupby('category')['price_per_unit'].transform('mean')
            unit_price_deviation = (lines['price_per_unit'] - cat_means).abs()
            avg_unit_price_deviation = unit_price_deviation.mean()
        else:
            avg_unit_price_deviation = np.nan

        if 'category' in lines:
            frac_high_risk = lines['category'].isin(self.high_risk_categories).mean()
        else:
            frac_high_risk = np.nan

        if 'was_voided' in lines:
            frac_voided = lines['was_voided'].astype(float).mean()
        else:
            frac_voided = np.nan

        if 'age_restricted' in lines:
            frac_age_restricted = lines['age_restricted'].astype(float).mean()
        else:
            frac_age_restricted = np.nan

        if 'weight' in lines:
            frac_zero_weight = (lines['weight'] == 0).mean()
        else:
            frac_zero_weight = np.nan

        if 'popularity' in lines:
            avg_popularity_deviation = (lines['popularity'] - lines['popularity'].mean()).abs().mean()
        else:
            avg_popularity_deviation = np.nan

        features = [
            avg_price,
            median_price,
            avg_discount_ratio,
            avg_price_delta,
            avg_unit_price_deviation,
            frac_high_risk,
            frac_voided,
            frac_age_restricted,
            frac_zero_weight,
            avg_popularity_deviation
        ]

        # 2. One-Hot-Encoding der Kategorien
        if 'category' in lines:
            present_categories = set(lines['category'].unique())
            for cat in self.categories_:
                features.append(int(cat in present_categories))
        else:
            features += [0] * len(self.categories_)

        return features

    def transform(self, X, y=None):
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else:
            X_df = X.copy()

        stats = X_df.iloc[:, 0].apply(self._calculate_features)
        feature_names = self.feature_names_out_.copy()
        feature_names += [f'ft_has_category_{cat}' for cat in self.categories_]

        df_transformed = pd.DataFrame(stats.tolist(), index=X_df.index, columns=feature_names)
        df_transformed.fillna(0, inplace=True)
        return df_transformed

    def get_feature_names_out(self, input_features=None):
        feature_names = self.feature_names_out_.copy()
        feature_names += [f'ft_has_category_{cat}' for cat in self.categories_]
        return feature_names
class EnhancedTransactionLineFeatures(BaseEstimator, TransformerMixin):
    """
    Diese Klasse kombiniert klassische und erweiterte Features für Transaktionszeilen.
    Ziel: Verdächtige Muster für Fraud-Erkennung an Self-Checkout-Kassen sichtbar machen.
    Features werden aus Produktdetails jeder Transaktion berechnet und aggregiert.
    """

    def __init__(self, price_representation='actual', normalize=False):
        """
        :param price_representation: 'binary' (binär), 'actual' (tatsächlich), 'normalized' (normalisiert)
        :param normalize: True für MinMax-Normalisierung der numerischen Features
        """
        self.price_representation = price_representation
        self.normalize = normalize
        # Alle relevanten Kategorien (kann angepasst werden)
        self.all_categories = [
            'TOBACCO', 'ALCOHOL', 'PERSONAL_CARE', 'LONG_SHELF_LIFE',
            'FROZEN_GOODS', 'HOUSEHOLD', 'BEVERAGES', 'BAKERY',
            'FRUITS_VEGETABLES_PIECES', 'DAIRY', 'CONVENIENCE',
            'FRUITS_VEGETABLES', 'SNACKS'
        ]
        # High-Risk-Kategorien (kann angepasst werden)
        self.high_risk_categories = {'SNACKS', 'FRUITS_VEGETABLES_PIECES', 'CONVENIENCE'}
        # Tage, ab wann ein Produkt nicht mehr als "neu" gilt
        self.new_product_days = 30

        # Feature-Namen dynamisch generieren
        self._initialize_feature_names()

    def _initialize_feature_names(self):
        """Initialisiert die Feature-Namen für die Ausgabe"""
        self.feature_names_out_ = [
            # Basis-Features (Preis, Rabatt, Stornierung, Gewicht, Altersbeschränkung, Popularität)
            'ft_avg_line_price',         # Durchschnittlicher Listenpreis pro Transaktion
            'ft_median_line_price',      # Median-Preis pro Transaktion
            'ft_avg_discount_ratio',     # Durchschnittlicher Rabattanteil pro Transaktion
            'ft_avg_price_delta',        # Durchschnittliche Abweichung zwischen Preis und erwartetem Preis
            'ft_avg_unit_price_deviation', # Durchschnittliche Abweichung des Einzelpreises vom Kategoriendurchschnitt
            'ft_frac_high_risk',         # Anteil der Hochrisikokategorie-Artikel pro Transaktion
            'ft_frac_voided',            # Anteil der stornierten Artikel pro Transaktion
            'ft_frac_age_restricted',    # Anteil der altersbeschränkten Artikel pro Transaktion
            'ft_frac_zero_weight',       # Anteil der Artikel mit Gewicht = 0 (unplausibel)
            'ft_avg_popularity_deviation', # Durchschnittliche Abweichung der Popularität vom Transaktionsdurchschnitt
            # Erweiterte Features
            'ft_max_price_delta',        # Maximale Abweichung zwischen Preis und erwartetem Preis
            'ft_weight_discrepancy',     # Durchschnittliche Abweichung des Gewichts vom erwarteten Gewicht
            'ft_mixed_age_flag',         # Transaktion enthält sowohl altersbeschränkte als auch nicht-altersbeschränkte Artikel
            'ft_missing_age_restriction',# Anteil der Hochrisikoartikel ohne Alterskennzeichnung
            'ft_void_time_variance',     # Varianz der Zeit zwischen Scan und Storno
            'ft_void_high_risk_ratio',   # Anteil der stornierten Hochrisikoartikel
            'ft_popularity_deviation',   # Durchschnittliche Abweichung der Popularität vom Transaktionsmedian
            'ft_new_product_ratio'       # Anteil der neu eingeführten Produkte
        ]
        # One-Hot-Features für jede Kategorie
        self.feature_names_out_ += [f'ft_has_category_{cat}' for cat in self.all_categories]

    def _handle_price_representation(self, lines):
        """Verarbeitet verschiedene Preisrepräsentationen"""
        # Warum: Je nach Anwendungsfall kann es sinnvoll sein, den Preis binär, tatsächlich oder normalisiert zu verwenden.
        # Binär: Nur prüfen, ob ein Preis vorhanden ist (z.B. für bestimmte Betrugsmuster)
        # Normalisiert: Skaliert die Preise auf 0-1, damit sie vergleichbar sind (wichtig für ML-Modelle)
        # Tatsächlich: Rohdaten, z.B. für bestimmte Analysen
        if self.price_representation == 'binary':
            return (lines['price'] > 0).astype(int)
        elif self.price_representation == 'normalized':
            price = lines['price']
            price_diff = price.max() - price.min()
            if price_diff == 0:
                return np.zeros(len(price))
            return (price - price.min()) / price_diff
        return lines['price']

    def _calculate_features(self, transaction_lines):
        """Berechnet alle Features für eine Liste von Produktzeilen (eine Transaktion)"""
        if not isinstance(transaction_lines, (list, np.ndarray)) or len(transaction_lines) == 0:
            # Warum: Leere oder ungültige Transaktionen werden mit NaN/0 belegt, damit das Modell robust bleibt
            features = [np.nan] * (len(self.feature_names_out_) - len(self.all_categories))
            features += [0] * len(self.all_categories)
            return features

        lines = pd.DataFrame([line for line in transaction_lines if isinstance(line, dict)])
        if lines.empty:
            features = [np.nan] * (len(self.feature_names_out_) - len(self.all_categories))
            features += [0] * len(self.all_categories)
            return features

        # 1. Preisbehandlung
        prices = self._handle_price_representation(lines)
        if isinstance(prices, pd.Series):
            avg_price = prices.mean()
            median_price = prices.median()
        else:
            avg_price = np.mean(prices) if len(prices) > 0 else np.nan
            median_price = np.median(prices) if len(prices) > 0 else np.nan

        # 2. Basis-Features
        # Warum: Durchschnittlicher Rabattanteil kann auf ungewöhnliche Rabattaktionen hinweisen
        avg_discount_ratio = np.nanmean(lines['discount_amount'] / lines['price']) if 'discount_amount' in lines and 'price' in lines else np.nan
        # Warum: Preisabweichungen können auf Betrug oder Fehler hinweisen
        avg_price_delta = np.nanmean(lines['price'] - lines['expected_price']) if 'expected_price' in lines and 'price' in lines else np.nan
        # Warum: Abweichungen vom Kategoriendurchschnitt können auf Manipulationen hinweisen
        if 'price_per_unit' in lines and 'category' in lines:
            cat_means = lines.groupby('category')['price_per_unit'].transform('mean')
            unit_price_deviation = (lines['price_per_unit'] - cat_means).abs()
            avg_unit_price_deviation = unit_price_deviation.mean()
        else:
            avg_unit_price_deviation = np.nan

        # 3. Kategorien-Features
        # Warum: Hochrisikokategorien (z.B. SNACKS.etc) sind besonders betrugsanfällig
        if 'category' in lines:
            frac_high_risk = lines['category'].isin(self.high_risk_categories).mean()
        else:
            frac_high_risk = np.nan
        # Warum: Häufige Stornierungen können auf Betrugsmuster hinweisen
        if 'was_voided' in lines:
            frac_voided = lines['was_voided'].astype(float).mean()
        else:
            frac_voided = np.nan
        # Warum: Altersbeschränkte Artikel erfordern besondere Aufmerksamkeit
        if 'age_restricted' in lines:
            frac_age_restricted = lines['age_restricted'].astype(float).mean()
        else:
            frac_age_restricted = np.nan
        # Warum: Artikel mit Gewicht = 0 sind unplausibel und können auf Fehler/Betrug hinweisen
        if 'weight' in lines:
            frac_zero_weight = (lines['weight'] == 0).mean()
        else:
            frac_zero_weight = np.nan
        # Warum: Ungewöhnliche Popularität kann auf Manipulationen hinweisen
        if 'popularity' in lines:
            avg_popularity_deviation = (lines['popularity'] - lines['popularity'].mean()).abs().mean()
        else:
            avg_popularity_deviation = np.nan

        # 4. Erweiterte Features
        # Warum: Maximale Preisabweichung kann auf extreme Manipulationen hinweisen
        max_price_delta = np.nanmax(lines['price'] - lines['expected_price']) if 'expected_price' in lines and 'price' in lines else np.nan
        # Warum: Gewichtsdiskrepanzen können auf falsche Eingaben oder Betrug hinweisen
        if 'weight' in lines and 'pieces_or_weight' in lines and 'base_product_id' in lines:
            expected_weight = lines['pieces_or_weight'] * lines.groupby('base_product_id')['weight'].transform('median')
            weight_discrepancy = np.nanmean(np.abs(lines['weight'] - expected_weight))
        else:
            weight_discrepancy = np.nan
        # Warum: Transaktionen mit gemischten Altersbeschränkungen sind verdächtig
        if 'age_restricted' in lines:
            age_flags = lines['age_restricted'].astype(bool)
            mixed_age_flag = (age_flags.any() & ~age_flags.all()).astype(int)
        else:
            mixed_age_flag = np.nan
        # Warum: Fehlende Alterskennzeichnung bei Hochrisikoartikeln kann auf Betrug hinweisen
        if 'category' in lines and 'age_restricted' in lines:
            mask = lines['category'].isin(self.high_risk_categories)
            missing_age_restriction = lines[mask]['age_restricted'].eq(0).mean()
        else:
            missing_age_restriction = np.nan
        # Warum: Ungewöhnliche Zeit zwischen Scan und Storno kann auf Betrug hinweisen
        if 'void_timestamp' in lines and 'scan_timestamp' in lines:
            time_diff = (pd.to_datetime(lines['void_timestamp']) - pd.to_datetime(lines['scan_timestamp'])).dt.total_seconds()
            void_time_variance = time_diff.var()
        else:
            void_time_variance = np.nan
        # Warum: Häufige Stornierungen bei Hochrisikoartikeln sind besonders verdächtig
        if 'was_voided' in lines and 'category' in lines:
            mask = (lines['was_voided'] == 1)
            void_high_risk_ratio = lines[mask]['category'].isin(self.high_risk_categories).mean()
        else:
            void_high_risk_ratio = np.nan
        # Warum: Abweichung der Popularität vom Median kann auf Manipulationen hinweisen
        popularity_deviation = (lines['popularity'] - lines['popularity'].median()).abs().mean() if 'popularity' in lines else np.nan
        # Warum: Neue Produkte sind weniger bekannt und können leichter manipuliert werden
        if 'product_launch_date' in lines:
            days_since_launch = (datetime.now() - pd.to_datetime(lines['product_launch_date'])).dt.days
            new_product_ratio = (days_since_launch < self.new_product_days).mean()
        else:
            new_product_ratio = np.nan

        # 5. One-Hot-Encoding der Kategorien
        # Warum: Das Vorhandensein bestimmter Kategorien kann auf Betrugsmuster hinweisen
        # (z.B. bestimmte Kategorien werden häufiger manipuliert)
        if 'category' in lines:
            present_categories = set(lines['category'].unique())
            category_features = [int(cat in present_categories) for cat in self.all_categories]
        else:
            category_features = [0] * len(self.all_categories)

        # 6. Alle Features zusammenfassen
        features = [
            avg_price, median_price, avg_discount_ratio,
            avg_price_delta, avg_unit_price_deviation, frac_high_risk,
            frac_voided, frac_age_restricted, frac_zero_weight,
            avg_popularity_deviation,
            max_price_delta, weight_discrepancy, mixed_age_flag,
            missing_age_restriction, void_time_variance, void_high_risk_ratio,
            popularity_deviation, new_product_ratio
        ]
        features += category_features

        # NaN durch 0 ersetzen für Robustheit
        # Warum: Damit das Modell auch mit fehlenden Werten umgehen kann
        return [0 if pd.isna(x) else x for x in features]

    def fit(self, X, y=None):
        # Warum: Keine Anpassung nötig, da keine globalen Parameter gelernt werden
        return self

    def transform(self, X, y=None):
        """Transformiert den DataFrame mit 'transaction_lines_details' in einen DataFrame mit den berechneten Features"""
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else:
            X_df = X.copy()

        stats = X_df.iloc[:, 0].apply(self._calculate_features)
        df_transformed = pd.DataFrame(
            stats.tolist(),
            index=X_df.index,
            columns=self.feature_names_out_
        )

        # Optional: Normalisierung der numerischen Features
        # Warum: Damit alle Features auf einer Skala liegen und das Modell besser lernt
        if self.normalize:
            numerical_features = [
                'ft_avg_line_price', 'ft_median_line_price', 'ft_avg_discount_ratio',
                'ft_avg_price_delta', 'ft_avg_unit_price_deviation', 'ft_frac_high_risk',
                'ft_frac_voided', 'ft_frac_age_restricted', 'ft_frac_zero_weight',
                'ft_avg_popularity_deviation',
                'ft_max_price_delta', 'ft_weight_discrepancy', 'ft_mixed_age_flag',
                'ft_missing_age_restriction', 'ft_void_time_variance', 'ft_void_high_risk_ratio',
                'ft_popularity_deviation', 'ft_new_product_ratio'
            ]
            scaler = MinMaxScaler()
            df_transformed[numerical_features] = scaler.fit_transform(df_transformed[numerical_features])

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