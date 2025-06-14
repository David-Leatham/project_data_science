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
    

class EnhancedTransactionLineFeatures(BaseEstimator, TransformerMixin):
    """
    Transformer zur Berechnung erweiterter Merkmale (Features) aus Transaktionszeilen.
    Geeignet für die Verwendung in sklearn-Pipelines.
    """

    def __init__(self, price_representation='actual', normalize=False):
        """
        Parameter:
        - price_representation: 'actual', 'binary' oder 'normalized'
          → bestimmt, wie Preise behandelt werden
        - normalize: Wenn True, werden numerische Features mit MinMaxScaler normalisiert
        """
        self.price_representation = price_representation
        self.normalize = normalize

        # Alle möglichen Kategorien im System – werden als binäre Features codiert
        self.all_categories = [
            'TOBACCO', 'ALCOHOL', 'PERSONAL_CARE', 'LONG_SHELF_LIFE',
            'FROZEN_GOODS', 'HOUSEHOLD', 'BEVERAGES', 'BAKERY',
            'FRUITS_VEGETABLES_PIECES', 'DAIRY', 'CONVENIENCE',
            'FRUITS_VEGETABLES', 'SNACKS'
        ]

        # Kategorien mit erhöhtem Risiko, z. B. weil sie oft storniert oder missbraucht werden
        self.high_risk_categories = {'SNACKS', 'FRUITS_VEGETABLES_PIECES', 'CONVENIENCE'}

        # Schwelle in Tagen: Produkte gelten als "neu", wenn Launch < X Tage zurückliegt
        self.new_product_days = 30

        # Initialisiert Liste aller Feature-Namen
        self._initialize_feature_names()

    def _initialize_feature_names(self):
        """
        Definiert die Namen aller berechneten Merkmale (Spaltennamen).
        Enthält sowohl Basis- als auch erweiterte Features.
        Zusätzlich werden One-Hot-Features für das Vorhandensein bestimmter Kategorien generiert.
        """

        self.feature_names_out_ = [
            # === Basis-Features ===
            'ft_avg_line_price',            # Durchschnittlicher Listenpreis aller Artikel in der Transaktion
            'ft_median_line_price',         # Median des Artikelpreises in der Transaktion
            'ft_avg_discount_ratio',        # Durchschnittlicher relativer Rabatt pro Artikel (Rabatt / Preis)
            'ft_avg_price_delta',           # Durchschnittliche Abweichung von erwartetem zu tatsächlichem Preis
            'ft_avg_unit_price_deviation',  # Durchschnittliche Abweichung vom Kategoriendurchschnitt im Preis pro Einheit
            'ft_frac_high_risk',            # Anteil an Artikeln aus Hochrisiko-Kategorien (z. B. Snacks)
            'ft_frac_voided',               # Anteil der Artikel, die storniert wurden (z. B. durch Kundeneingriff)
            'ft_frac_age_restricted',       # Anteil der Artikel mit Altersbeschränkung
            'ft_frac_zero_weight',          # Anteil der Artikel mit Gewicht = 0 (häufig unplausibel)
            'ft_avg_popularity_deviation',  # Ø Abweichung der Beliebtheit vom Mittelwert in der Transaktion

            # === Erweiterte Features ===
            'ft_max_price_delta',           # Größte Preisabweichung eines Artikels innerhalb der Transaktion
            'ft_weight_discrepancy',        # Ø Abweichung zwischen erwartetem und tatsächlichem Gewicht pro Artikel
            'ft_mixed_age_flag',            # Boolesches Flag: Mischung aus altersbeschränkten und nicht-beschränkten Artikeln
            'ft_missing_age_restriction',   # Anteil riskanter Produkte ohne Alterskennzeichnung
            'ft_void_time_variance',        # Zeitliche Streuung (Varianz) der Differenz zwischen Scan und Storno
            'ft_void_high_risk_ratio',      # Anteil stornierter Artikel, die Hochrisiko-Kategorien angehören
            'ft_popularity_deviation',      # Ø Abweichung der Beliebtheit vom Median der Transaktion
            'ft_new_product_ratio'          # Anteil der Artikel mit kurzer Markteinführungszeit (z. B. < 30 Tage)
        ]

        # === Kategoriepräsenz-Features ===
        # Für jede bekannte Kategorie wird ein One-Hot-Flag gesetzt: Kommt diese Kategorie in der Transaktion vor?
        self.feature_names_out_ += [f'ft_has_category_{cat}' for cat in self.all_categories]

    def _handle_price_representation(self, lines):
        """Wandelt Preise abhängig von der gewählten Repräsentation um."""
        if self.price_representation == 'binary':
            return (lines['price'] > 0).astype(int)
        elif self.price_representation == 'normalized':
            price = lines['price']
            diff = price.max() - price.min()
            return (price - price.min()) / diff if diff != 0 else np.zeros(len(price))
        return lines['price']  # Standard: tatsächlicher Preis

    def _calculate_features(self, transaction_lines):
        """Berechnet alle numerischen und kategorialen Merkmale für eine Transaktion."""
        # Schutz: leere oder ungültige Eingabe
        if not isinstance(transaction_lines, (list, np.ndarray)) or len(transaction_lines) == 0:
            return [0] * len(self.feature_names_out_)

        # Konvertiere List of Dicts in DataFrame
        lines = pd.DataFrame([line for line in transaction_lines if isinstance(line, dict)])
        if lines.empty:
            return [0] * len(self.feature_names_out_)

        # Preismodellierung
        prices = self._handle_price_representation(lines)
        avg_price = prices.mean()
        median_price = prices.median()

        # Durchschnittlicher Rabatt relativ zum Preis
        if 'discount_amount' in lines and 'price' in lines:
            avg_discount_ratio = np.nanmean(lines['discount_amount'] / lines['price'])
        else:
            avg_discount_ratio = 0

        # Durchschnittliche Abweichung von erwartetem Preis
        if 'expected_price' in lines and 'price' in lines:
            avg_price_delta = np.nanmean(lines['price'] - lines['expected_price'])
        else:
            avg_price_delta = 0

        # Preisabweichung pro Einheit innerhalb der Kategorie
        if 'price_per_unit' in lines and 'category' in lines:
            cat_means = lines.groupby('category')['price_per_unit'].transform('mean')
            deviation = (lines['price_per_unit'] - cat_means).abs()
            avg_unit_price_deviation = deviation.mean()
        else:
            avg_unit_price_deviation = 0

        # Anteil riskanter Kategorien
        frac_high_risk = lines['category'].isin(self.high_risk_categories).mean() if 'category' in lines else 0

        # Anteil stornierter Positionen
        frac_voided = lines['was_voided'].astype(float).mean() if 'was_voided' in lines else 0

        # Anteil altersbeschränkter Produkte
        frac_age_restricted = lines['age_restricted'].astype(float).mean() if 'age_restricted' in lines else 0

        # Anteil Positionen mit Gewicht = 0
        frac_zero_weight = (lines['weight'] == 0).mean() if 'weight' in lines else 0

        # Abweichung zur durchschnittlichen Beliebtheit
        if 'popularity' in lines:
            avg_popularity_deviation = (lines['popularity'] - lines['popularity'].mean()).abs().mean()
            popularity_deviation = (lines['popularity'] - lines['popularity'].median()).abs().mean()
        else:
            avg_popularity_deviation = 0
            popularity_deviation = 0

        # Max. Abweichung vom erwarteten Preis
        if 'expected_price' in lines and 'price' in lines:
            max_price_delta = np.nanmax(lines['price'] - lines['expected_price'])
        else:
            max_price_delta = 0

        # Diskrepanz zwischen erwartetem und tatsächlichem Gewicht
        if all(col in lines for col in ['weight', 'pieces_or_weight', 'base_product_id']):
            expected_weight = lines['pieces_or_weight'] * lines.groupby('base_product_id')['weight'].transform('median')
            weight_discrepancy = np.nanmean(np.abs(lines['weight'] - expected_weight))
        else:
            weight_discrepancy = 0

        # Flag, ob eine Mischung aus altersbeschränkten und nicht-altersbeschränkten Produkten vorliegt
        if 'age_restricted' in lines:
            age_flags = lines['age_restricted'].astype(bool)
            mixed_age_flag = int(age_flags.any() & ~age_flags.all())
        else:
            mixed_age_flag = 0

        # Anteil riskanter Produkte ohne Altersbeschränkung
        if 'category' in lines and 'age_restricted' in lines:
            risky = lines['category'].isin(self.high_risk_categories)
            missing_age_restriction = lines[risky]['age_restricted'].eq(0).mean()
        else:
            missing_age_restriction = 0

        # Varianz der Zeitdifferenz zwischen Scan und Storno
        if 'void_timestamp' in lines and 'scan_timestamp' in lines:
            time_diff = (pd.to_datetime(lines['void_timestamp']) - pd.to_datetime(lines['scan_timestamp'])).dt.total_seconds()
            void_time_variance = time_diff.var()
        else:
            void_time_variance = 0

        # Anteil stornierter Positionen aus risikoreichen Kategorien
        if 'was_voided' in lines and 'category' in lines:
            voided = lines['was_voided'] == 1
            void_high_risk_ratio = lines[voided]['category'].isin(self.high_risk_categories).mean()
        else:
            void_high_risk_ratio = 0

        # Anteil neuer Produkte basierend auf Launch-Datum
        if 'product_launch_date' in lines:
            days_since = (datetime.now() - pd.to_datetime(lines['product_launch_date'])).dt.days
            new_product_ratio = (days_since < self.new_product_days).mean()
        else:
            new_product_ratio = 0

        # Binärcodierung: Ist eine Kategorie enthalten?
        if 'category' in lines:
            present = set(lines['category'].unique())
            category_features = [int(cat in present) for cat in self.all_categories]
        else:
            category_features = [0] * len(self.all_categories)

        # Alle Features zusammenfassen, NaNs durch 0 ersetzen
        features = [
            avg_price, median_price, avg_discount_ratio,
            avg_price_delta, avg_unit_price_deviation, frac_high_risk,
            frac_voided, frac_age_restricted, frac_zero_weight,
            avg_popularity_deviation, max_price_delta, weight_discrepancy,
            mixed_age_flag, missing_age_restriction, void_time_variance,
            void_high_risk_ratio, popularity_deviation, new_product_ratio
        ] + category_features

        return [0 if pd.isna(x) else x for x in features]

    def fit(self, X, y=None):
        """Nichts zu tun – keine lernbaren Parameter."""
        return self

    def transform(self, X, y=None):
        """
        Wandelt das Eingabe-DataFrame in ein Feature-DataFrame um.
        Jede Zeile in X muss eine Liste von Transaktionszeilen (Dicts) enthalten.
        """
        # Konvertierung in DataFrame (z. B. wenn Series oder ndarray)
        if isinstance(X, pd.Series):
            X_df = X.to_frame()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else:
            X_df = X.copy()

        # Features pro Transaktion berechnen
        stats = X_df.iloc[:, 0].apply(self._calculate_features)

        # Warnung bei ungültigen Zeilen
        invalid_rows = stats[stats.apply(lambda row: any(pd.isna(v) for v in row))].index
        if len(invalid_rows) > 0:
            print(f"Warnung: {len(invalid_rows)} Transaktionen enthalten ungültige oder leere Daten.")

        # Zusammenbau des Feature-DataFrames
        df_transformed = pd.DataFrame(
            stats.tolist(),
            index=X_df.index,
            columns=self.feature_names_out_
        )

        # NaNs sicherheitshalber ersetzen
        df_transformed = df_transformed.fillna(0)

        # Optional: Normalisierung mit MinMaxScaler
        if self.normalize:
            numerical = self.feature_names_out_[:18]  # Nur numerische Features skalieren
            scaler = MinMaxScaler()
            df_transformed[numerical] = scaler.fit_transform(df_transformed[numerical])

        return df_transformed

    def get_feature_names_out(self, input_features=None):
        """Gibt die Namen der generierten Features zurück (für sklearn-kompatible Nutzung)."""
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