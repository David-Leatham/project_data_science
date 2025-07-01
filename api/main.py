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
# 1. BENUTZERDEFINIERTE TRANSFORMER-KLASSEN
# Diese Klassen sind für die Merkmalsextraktion (Feature Engineering) zuständig
# und wurden aus der ursprünglichen Datenpipeline (data_pipeline.py) übernommen.
# ==============================================================================

class ZeitmerkmaleGenerator(BaseEstimator, TransformerMixin):
    """
    Erzeugt zeitbasierte Merkmale aus den Start- und Endzeitstempeln einer Transaktion.
    Beispiele: Monat, Stunde, Wochentag, Dauer der Transaktion.
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        # Dieser Transformer lernt keine Parameter aus den Daten.
        return self
    def transform(self, X, y=None):
        # Erstellt einen neuen DataFrame, um die transformierten Daten zu speichern.
        df_transformiert = pd.DataFrame(index=X.index)
        
        # Konvertiert die Zeitstempel-Strings in datetime-Objekte.
        transaktion_start = pd.to_datetime(X['transaction_start'])
        transaktion_ende = pd.to_datetime(X['transaction_end'])
        
        # Berechnet die Dauer der Transaktion in Sekunden.
        dauer = transaktion_ende - transaktion_start
        dauer_sekunden_val = dauer.dt.total_seconds()
        
        # Extrahiert grundlegende Zeitmerkmale.
        df_transformiert['ft_month'] = transaktion_start.dt.month
        df_transformiert['ft_day'] = transaktion_start.dt.day
        df_transformiert['ft_hour'] = transaktion_start.dt.hour
        df_transformiert['ft_minute'] = transaktion_start.dt.minute
        df_transformiert['ft_second'] = transaktion_start.dt.second
        df_transformiert['ft_day_of_week'] = transaktion_start.dt.dayofweek
        df_transformiert['ft_day_of_year'] = transaktion_start.dt.dayofyear
        df_transformiert['ft_week_of_year'] = transaktion_start.dt.isocalendar().week.astype(int)
        df_transformiert['ft_quarter'] = transaktion_start.dt.quarter
        
        # Erzeugt binäre Merkmale (z.B. ob es Wochenende ist).
        df_transformiert['ft_is_weekend'] = df_transformiert['ft_day_of_week'].isin([5, 6]).astype(int)
        df_transformiert['ft_is_outside_normal_hours'] = ((df_transformiert['ft_hour'] < 8) | (df_transformiert['ft_hour'] >= 23)).astype(int)
        
        # Kodiert zyklische Merkmale mit Sinus/Kosinus-Transformation, um die Periodizität zu erfassen.
        df_transformiert['ft_month_sin'] = np.sin(2 * np.pi * df_transformiert['ft_month']/12)
        df_transformiert['ft_month_cos'] = np.cos(2 * np.pi * df_transformiert['ft_month']/12)
        df_transformiert['ft_hour_sin'] = np.sin(2 * np.pi * df_transformiert['ft_hour']/24)
        df_transformiert['ft_hour_cos'] = np.cos(2 * np.pi * df_transformiert['ft_hour']/24)
        df_transformiert['ft_day_of_week_sin'] = np.sin(2 * np.pi * df_transformiert['ft_day_of_week']/7)
        df_transformiert['ft_day_of_week_cos'] = np.cos(2 * np.pi * df_transformiert['ft_day_of_week']/7)
        tage_im_jahr = transaktion_start.dt.is_leap_year.map({True: 366, False: 365})
        df_transformiert['ft_day_of_year_sin'] = np.sin(2 * np.pi * df_transformiert['ft_day_of_year']/tage_im_jahr)
        df_transformiert['ft_day_of_year_cos'] = np.cos(2 * np.pi * df_transformiert['ft_day_of_year']/tage_im_jahr)
        
        # Log-Transformation der Dauer, um extreme Werte abzuschwächen.
        df_transformiert['ft_duration_seconds_log'] = np.log1p(dauer_sekunden_val)
        
        # Interaktionsmerkmale, die Kombinationen von Zeitmerkmalen erfassen.
        df_transformiert['ft_inter_weekend_outside_hours'] = df_transformiert['ft_is_weekend'] * df_transformiert['ft_is_outside_normal_hours']
        df_transformiert['ft_inter_hour_cos_x_dow_cos'] = df_transformiert['ft_hour_cos'] * df_transformiert['ft_day_of_week_cos']
        
        # Definiert die Liste der erzeugten Spaltennamen.
        self.feature_names_out_ = ['ft_month', 'ft_day', 'ft_hour', 'ft_minute', 'ft_second', 'ft_day_of_week', 'ft_day_of_year', 'ft_week_of_year', 'ft_quarter', 'ft_is_weekend', 'ft_is_outside_normal_hours', 'ft_month_sin', 'ft_month_cos', 'ft_hour_sin', 'ft_hour_cos', 'ft_day_of_week_sin', 'ft_day_of_week_cos', 'ft_day_of_year_sin', 'ft_day_of_year_cos', 'ft_duration_seconds_log', 'ft_inter_weekend_outside_hours', 'ft_inter_hour_cos_x_dow_cos']
        
        return df_transformiert[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class FeedbackKategorisierer(BaseEstimator, TransformerMixin):
    """
    Kategorisiert das Kundenfeedback und kodiert es mittels One-Hot-Encoding.
    """
    def __init__(self):
        self.categories_ = ['feedback_null', 'feedback_10', 'feedback_other']
        self.feature_names_out_ = [f"ft_{cat}" for cat in self.categories_]
    def fit(self, X, y=None):
        return self
    def _categorize_feedback(self, feedback_wert):
        if pd.isnull(feedback_wert): return 'feedback_null'
        if feedback_wert == 10.0: return 'feedback_10'
        return 'feedback_other'
    def transform(self, X, y=None):
        feedback_serie = X.iloc[:, 0]
        kategorisiertes_feedback = feedback_serie.apply(self._categorize_feedback)
        kategorisierter_typ = pd.Categorical(kategorisiertes_feedback, categories=self.categories_)
        dummies_df = pd.get_dummies(kategorisierter_typ, prefix='ft', dtype=int)
        return dummies_df[self.feature_names_out_]
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

class ErweiterteTransaktionspositionsMerkmale(BaseEstimator, TransformerMixin):
    """
    Berechnet aggregierte Merkmale aus den einzelnen Transaktionspositionen (Warenkorb).
    Beispiele: Durchschnittlicher Preis pro Artikel, Anteil an Hochrisiko-Kategorien.
    """
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
    
    def _calculate_features(self, transaktionszeilen):
        if not isinstance(transaktionszeilen, (list, np.ndarray)) or len(transaktionszeilen) == 0: return [0] * len(self.feature_names_out_)
        zeilen_df = pd.DataFrame([z for z in transaktionszeilen if isinstance(z, dict)])
        if zeilen_df.empty: return [0] * len(self.feature_names_out_)
        
        preise = self._handle_price_representation(zeilen_df)
        durchschnittlicher_preis = preise.mean()
        median_preis = preise.median()
        
        if 'expected_price' in zeilen_df and 'price' in zeilen_df and zeilen_df['expected_price'].notna().any():
            preis_delta = zeilen_df['price'] - zeilen_df['expected_price']
            durchschnittliche_preis_delta = np.nanmean(preis_delta)
            max_preis_delta = np.nanmax(preis_delta)
        else:
            durchschnittliche_preis_delta = 0
            max_preis_delta = 0

        durchschnittlicher_rabatt = np.nanmean(zeilen_df['discount_amount'] / zeilen_df['price']) if 'discount_amount' in zeilen_df and 'price' in zeilen_df and zeilen_df['price'].notna().any() else 0
        
        if 'price_per_unit' in zeilen_df and 'category' in zeilen_df and zeilen_df['price_per_unit'].notna().any():
            kategorie_mittelwerte = zeilen_df.groupby('category')['price_per_unit'].transform('mean')
            durchschnittliche_einheitspreis_abweichung = (zeilen_df['price_per_unit'] - kategorie_mittelwerte).abs().mean()
        else: 
            durchschnittliche_einheitspreis_abweichung = 0
            
        anteil_hohes_risiko = zeilen_df['category'].isin(self.high_risk_categories).mean() if 'category' in zeilen_df else 0
        anteil_altersbeschraenkt = zeilen_df['age_restricted'].astype(float).mean() if 'age_restricted' in zeilen_df else 0
        gemischtes_alter_flag = int(zeilen_df['age_restricted'].any() & ~zeilen_df['age_restricted'].all()) if 'age_restricted' in zeilen_df else 0
        
        if 'category' in zeilen_df and 'age_restricted' in zeilen_df: 
            riskante_zeilen = zeilen_df[zeilen_df['category'].isin(self.high_risk_categories)]
            fehlende_altersbeschraenkung = riskante_zeilen['age_restricted'].eq(0).mean() if not riskante_zeilen.empty else 0
        else: 
            fehlende_altersbeschraenkung = 0
            
        if 'product_launch_date' in zeilen_df and zeilen_df['product_launch_date'].notna().any():
            tage_seit_einfuehrung = (datetime.now() - pd.to_datetime(zeilen_df['product_launch_date'])).dt.days
            anteil_neue_produkte = (tage_seit_einfuehrung < self.new_product_days).mean()
        else: 
            anteil_neue_produkte = 0
            
        vorhandene_kategorien = set(zeilen_df['category'].unique()) if 'category' in zeilen_df else set()
        kategorie_merkmale = [int(cat in vorhandene_kategorien) for cat in self.all_categories]
        
        merkmale = [durchschnittlicher_preis, median_preis, durchschnittlicher_rabatt, durchschnittliche_preis_delta, durchschnittliche_einheitspreis_abweichung, anteil_hohes_risiko, anteil_altersbeschraenkt, max_preis_delta, gemischtes_alter_flag, fehlende_altersbeschraenkung, anteil_neue_produkte] + kategorie_merkmale
        return [0 if pd.isna(x) else x for x in merkmale]

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if isinstance(X, pd.Series): X_df = X.to_frame()
        elif isinstance(X, np.ndarray): X_df = pd.DataFrame(X, columns=['transaction_lines_details'])
        else: X_df = X.copy()
        statistiken = X_df.iloc[:, 0].apply(self._calculate_features)
        df_transformiert = pd.DataFrame(statistiken.tolist(), index=X_df.index, columns=self.feature_names_out_).fillna(0)
        return df_transformiert
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_

# ==============================================================================
# 2. RUNTIME-PATCHES
# Diese Anpassungen sind notwendig, um Kompatibilitätsprobleme beim Laden
# der mit `joblib` gespeicherten Scikit-Learn-Objekte zu beheben.
# ==============================================================================

# Patch 1: Erstellt ein "virtuelles" Modul, damit joblib die benutzerdefinierten Klassen findet.
pipeline_modul = types.ModuleType("pipeline")
pipeline_modul.TimeFeatureGenerator = ZeitmerkmaleGenerator
pipeline_modul.FeedbackBinnerOHE = FeedbackKategorisierer
pipeline_modul.EnhancedTransactionLineFeatures = ErweiterteTransaktionspositionsMerkmale
sys.modules["pipeline"] = pipeline_modul
sys.modules["pipeline.data_pipeline"] = pipeline_modul

# Patch 2: "Monkey-Patching" des FunctionTransformers, um Fehler bei der Verarbeitung
# einzelner Datenzeilen zu vermeiden (behebt den 'dtype'-Fehler).
original_ft_transform = FunctionTransformer.transform
def robuster_ft_transform(self, X):
    ergebnis = original_ft_transform(self, X)
    if hasattr(ergebnis, 'ndim') and ergebnis.ndim == 1:
        ergebnis = ergebnis.reshape(-1, 1)
    if isinstance(ergebnis, float):
        ergebnis = np.array([[ergebnis]])
    return ergebnis
FunctionTransformer.transform = robuster_ft_transform

# --- Konfiguration ---
MODELL_VERZEICHNIS = './weights'

# --- Laden der Modelle, des Preprocessors und des SHAP-Explainers beim Start ---
try:
    # Laden des Klassifikationsmodells
    klassifikator_liste = glob.glob(os.path.join(MODELL_VERZEICHNIS, '*_model_*.cbm'))
    if not klassifikator_liste: raise FileNotFoundError("Kein Klassifikationsmodell gefunden.")
    aktuellster_klassifikator_pfad = max(klassifikator_liste, key=os.path.getctime)
    klassifikationsmodell = CatBoostClassifier()
    klassifikationsmodell.load_model(aktuellster_klassifikator_pfad)

    # Laden des Regressionsmodells
    regressor_liste = glob.glob(os.path.join(MODELL_VERZEICHNIS, '*_regressor_*.cbm'))
    if not regressor_liste: raise FileNotFoundError("Kein Regressionsmodell gefunden.")
    aktuellster_regressor_pfad = max(regressor_liste, key=os.path.getctime)
    regressionsmodell = CatBoostRegressor()
    regressionsmodell.load_model(aktuellster_regressor_pfad)

    # Laden des Preprocessors
    preprocessor_liste = glob.glob(os.path.join(MODELL_VERZEICHNIS, 'preprocessor*.joblib'))
    if not preprocessor_liste: raise FileNotFoundError("Kein Preprocessor gefunden.")
    aktuellster_preprocessor_pfad = max(preprocessor_liste, key=os.path.getctime)
    preprocessor = joblib.load(aktuellster_preprocessor_pfad)

    # Erstellen des SHAP-Explainers (nur einmal beim Start der Anwendung)
    shap_explainer = shap.TreeExplainer(klassifikationsmodell)
    print("Modelle, Preprocessor und SHAP-Explainer erfolgreich geladen.")

except FileNotFoundError as e:
    raise RuntimeError(f"Modell-Artefakte konnten nicht geladen werden: {e}")
except Exception as e:
    raise RuntimeError(f"Ein unerwarteter Fehler beim Laden der Artefakte ist aufgetreten: {e}")

# --- API-Datenmodelle (gemäß Spezifikation) ---

class Merkmalswichtigkeit(BaseModel):
    feature: str
    value: float
    shap_value: float

class Erklaerung(BaseModel):
    human_readable_reason: Optional[str] = None
    feature_importance: Optional[List[Merkmalswichtigkeit]] = None
    offending_products: Optional[List[str]] = None

class Betrugsvorhersage(BaseModel):
    version: str
    is_fraud: bool
    fraud_proba: Optional[float] = Field(None, ge=0, le=1)
    estimated_damage: Optional[float] = None
    explanation: Optional[Erklaerung] = None

    @field_validator("version")
    @classmethod
    def validate_semantic_version(cls, v):
        teile = v.split(".")
        if len(teile) != 3 or not all(p.isdigit() for p in teile):
            raise ValueError("Version muss dem semantischen Versionierungsschema (x.y.z) folgen")
        return v

class Transaktionszeile(BaseModel):
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

class Transaktionskopf(BaseModel):
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

class BetrugsvorhersageAnfrage(BaseModel):
    transaction_header: Transaktionskopf
    transaction_lines: List[Transaktionszeile]

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat(), UUID: lambda v: str(v)}

# --- Mapping von technischen zu lesbaren Merkmalsnamen ---
merkmalsnamen_mapping = {
    'num_log_center__total_amount': 'Gesamtbetrag',
    'num_log_center__n_lines': 'Anzahl der Artikel',
    'onehot_cat__payment_medium_CASH': 'Zahlungsmethode (Bar)',
    'onehot_cat__payment_medium_CREDIT_CARD': 'Zahlungsmethode (Kreditkarte)',
    'time_features__ft_hour_cos': 'Tageszeit (Spät nachts/Früh morgens)',
    'time_features__ft_hour': 'Stunde der Transaktion',
    'transaction_lines_stats__ft_avg_line_price': 'Durchschnittlicher Preis pro Artikel',
    'transaction_lines_stats__ft_has_category_SNACKS': 'Enthält Snacks',
    'transaction_lines_stats__ft_has_category_CONVENIENCE': 'Enthält Convenience-Produkte',
    'transaction_lines_stats__ft_has_category_FRUITS_VEGETABLES_PIECES': 'Enthält Obst/Gemüse (Stück)',
    'time_features__ft_duration_seconds_log': 'Transaktionsdauer',
    'transaction_lines_stats__ft_frac_high_risk': 'Anteil an Hochrisiko-Artikeln',
    'feedback_processing__ft_feedback_null': 'Kein Kundenfeedback',
    'feedback_processing__ft_feedback_10': 'Kundenfeedback (Perfekt)',
    'time_features__ft_day_of_week': 'Wochentag'
}

# --- Kategorien mit hohem Risiko für auffällige Produkte ---
hochrisiko_kategorien = {'SNACKS', 'FRUITS_VEGETABLES_PIECES', 'CONVENIENCE'}

# --- FastAPI-Anwendung ---
app = FastAPI(
    title="SCO Betrugs-REST-API",
    description="Eine REST-API zur Echtzeit-Betrugserkennung am Self-Checkout (SCO).",
    version="0.1.1"
)

API_VERSION = "0.1.1"
KLASSIFIKATIONS_SCHWELLENWERT = 0.47

@app.get("/", tags=["Health Check"])
async def health_check():
    """Gibt den Status und die Version der API zurück."""
    return {"status": "healthy", "version": API_VERSION}

@app.post("/fraud-prediction", response_model=Betrugsvorhersage, tags=["Betrugsvorhersage"])
async def predict_fraud(anfrage: BetrugsvorhersageAnfrage):
    """Analysiert eine Transaktion auf potenziellen Betrug."""
    try:
        # --- 1. Datenaufbereitung ---
        kopfdaten = anfrage.transaction_header.dict()
        kopfdaten['n_lines'] = len(anfrage.transaction_lines)
        kopfdaten['transaction_lines_details'] = [zeile.dict() for zeile in anfrage.transaction_lines]
        eingabe_df = pd.DataFrame([kopfdaten])

        # --- 2. Vorverarbeitung (Feature Engineering) ---
        vorverarbeitete_merkmale = preprocessor.transform(eingabe_df)
        merkmalsnamen = preprocessor.get_feature_names_out()
        vorverarbeitete_merkmale_df = pd.DataFrame(vorverarbeitete_merkmale, columns=merkmalsnamen)

        # --- 3. Klassifikations-Vorhersage ---
        betrugswahrscheinlichkeit = klassifikationsmodell.predict_proba(vorverarbeitete_merkmale)[0][1]
        ist_betrug = bool(betrugswahrscheinlichkeit >= KLASSIFIKATIONS_SCHWELLENWERT)

        # --- 4. SHAP-Werte zur Erklärung berechnen (für jede Anfrage) ---
        shap_werte = shap_explainer.shap_values(vorverarbeitete_merkmale_df)
        
        if isinstance(shap_werte, list):
            shap_werte_fuer_betrug = shap_werte[1][0]
        else:
            shap_werte_fuer_betrug = shap_werte[0]
        
        merkmalseinfluesse = []
        for i, merkmal_name in enumerate(merkmalsnamen):
            merkmal_wert = vorverarbeitete_merkmale_df.iloc[0, i]
            shap_wert = shap_werte_fuer_betrug[i]
            
            if abs(shap_wert) > 0.01:
                lesbarer_name = merkmalsnamen_mapping.get(merkmal_name, merkmal_name)
                merkmalseinfluesse.append({
                    "feature": lesbarer_name,
                    "value": merkmal_wert,
                    "shap_value": shap_wert
                })

        merkmalswichtigkeit_liste = sorted(merkmalseinfluesse, key=lambda x: abs(x['shap_value']), reverse=True)

        # --- 5. Initialisierung der Antwortvariablen ---
        geschaetzter_schaden = 0.0
        lesbare_begruendung_text = ""
        auffaellige_produkte = []
        
        if ist_betrug:
            # --- Regressions-Vorhersage für den Schaden ---
            vorhergesagter_roh_schaden = regressionsmodell.predict(vorverarbeitete_merkmale)
            geschaetzter_schaden = round(max(0, vorhergesagter_roh_schaden[0]), 2)

            # --- Identifizierung der auffälligen Produkte ---
            auffaellige_produkte_set = set()
            for zeile in anfrage.transaction_lines:
                if zeile.category in hochrisiko_kategorien:
                    auffaellige_produkte_set.add(zeile.category)
            auffaellige_produkte = list(auffaellige_produkte_set)

            # --- Erstellung der lesbaren Begründung ---
            top_betrugsmerkmale_desc = []
            for item in merkmalswichtigkeit_liste[:3]:
                desc = item['feature']
                if 'Preis' in desc and item['value'] < 1: desc += " (sehr niedrig)"
                elif 'Betrag' in desc and item['value'] < 1: desc += " (sehr niedrig)"
                elif 'Bar' in desc and item['value'] == 1: desc = "Barzahlung"
                elif 'Tageszeit' in desc and item['value'] > 0.5: desc = "Späte Transaktion"
                top_betrugsmerkmale_desc.append(desc)
            
            lesbare_begruendung_text = "Hohes Betrugsrisiko erkannt. Hauptfaktoren: " + "; ".join(top_betrugsmerkmale_desc) + "."
            if auffaellige_produkte:
                lesbare_begruendung_text += f" Verdächtige Produktkategorien: {', '.join(auffaellige_produkte)}."

        else: # KEIN BETRUG
            schuetzende_merkmale_sortiert = sorted(merkmalseinfluesse, key=lambda x: x['shap_value'])
            
            top_schuetzende_merkmale_desc = []
            for item in schuetzende_merkmale_sortiert[:3]:
                desc = item['feature']
                if 'Bar' in desc and item['value'] == 0: desc = "Kartenzahlung"
                elif 'Tageszeit' in desc and item['value'] < -0.5: desc = "Normale Geschäftszeiten"
                elif 'Preis' in desc and item['value'] > 10: desc += " (hoch)"
                top_schuetzende_merkmale_desc.append(desc)
            
            lesbare_begruendung_text = "Geringes Betrugsrisiko. Hauptfaktoren: " + "; ".join(top_schuetzende_merkmale_desc) + "."

        # --- Erklärungsobjekt erstellen ---
        erklaerung = Erklaerung(
            human_readable_reason=lesbare_begruendung_text,
            feature_importance=[Merkmalswichtigkeit(**item) for item in merkmalswichtigkeit_liste],
            offending_products=auffaellige_produkte
        )

        # --- Finale Antwort formatieren ---
        antwort = Betrugsvorhersage(
            version=API_VERSION,
            is_fraud=ist_betrug,
            fraud_proba=round(betrugswahrscheinlichkeit, 4),
            estimated_damage=geschaetzter_schaden,
            explanation=erklaerung,
        )

        return antwort

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Ein interner Fehler ist aufgetreten: {e}")
