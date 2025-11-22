import boto3
import pandas as pd
import numpy as np
import re
from io import StringIO
import datetime
from dotenv import load_dotenv
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Nazwa bucketu S3
BUCKET_NAME = "zadmod-9"

# Wczytanie zmiennych środowiskowych
load_dotenv()

# Inicjalizacja klienta S3
s3 = boto3.client("s3")


def load_data_from_s3(file_key):
    """Wczytuje dane z S3 do DataFrame."""
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=file_key)
    csv_data = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_data), sep=";")
    return df

# Wczytywanie danych dla 2023 i 2024
wroclaw_2023_df = load_data_from_s3("Dane_mod9/halfmarathon_wroclaw_2023__final.csv")
wroclaw_2024_df = load_data_from_s3("Dane_mod9/halfmarathon_wroclaw_2024__final.csv")

# I. KONWERSJA CZASU
def convert_time_to_seconds(time):
    if pd.isnull(time) or time in ['DNS', 'DNF']: #DID NOT START / DID NOT FINISH
        return None
    time = time.split(':')
    return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])

# Usuwanie zbędnych kolumn
def get_cleaned_dataframe(df_2023: pd.DataFrame, df_2024: pd.DataFrame) -> pd.DataFrame:
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)  # Łączenie danych z obu lat
    df_model = df_combined[df_combined['Miejsce'].notnull()].copy() # Usuwanie wierszy z brakującymi miejscami 

    # Usuwanie zbędnych kolumn
    columns_to_drop = ['Drużyna', 'Miasto'] # Wybrane kolumny do usunięcia
    df_model.drop(columns=columns_to_drop, inplace=True)

    return df_model

cleaned_df = get_cleaned_dataframe(wroclaw_2023_df, wroclaw_2024_df)
missing_values_count = cleaned_df.isna().sum().reset_index(name='ilość')
cleaned_df.isna().sum().reset_index(name='ilość')

# Uzupełnia brakujące wartości w kolumnie 'Rocznik' medianą dla każdej 'Kategorii wiekowej' i dodaje kolumnę 'Wiek
def impute_rocznik_from_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uzupełnia brakujące wartości w kolumnie 'Rocznik'
    medianą dla każdej 'Kategorii wiekowej' i dodaje kolumnę 'Wiek'.
    """
    if 'Rocznik' not in df.columns or 'Kategoria wiekowa' not in df.columns:
        return df

    df_result = df.copy()

    # Oblicz medianę 'Rocznik' dla każdej kategorii wiekowej
    median_map = df_result.groupby('Kategoria wiekowa')['Rocznik'].transform('median').round(0).astype('Int64')

    # Uzupełnij brakujące 'Rocznik' medianą odpowiedniej kategorii
    df_result['Rocznik'] = df_result['Rocznik'].fillna(median_map)

    # Dodaj kolumnę 'Wiek'
    df_result['Wiek'] = 2024 - df_result['Rocznik']

    # Usuń kolumnę 'Rocznik'
    df_result = df_result.drop(columns=['Rocznik'])

    # Wyświetl liczby brakujących przed i po
    print(f"Brakujące wartości w 'Rocznik' przed imputacją: {(df['Rocznik'].isnull().sum())}")
    print(f"Brakujące wartości w 'Rocznik' po imputacji: {(df_result['Wiek'].isnull().sum())}")

    return df_result

cleaned_df = get_cleaned_dataframe(wroclaw_2023_df, wroclaw_2024_df)
cleaned_df = impute_rocznik_from_category(cleaned_df)

# Zmiana kolumn czasu w cleaned_df - Konwersja czasu na sekundy
time_columns = ['5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', 'Czas']
cleaned_df[time_columns] = cleaned_df[time_columns].applymap(convert_time_to_seconds)

# Teraz wyświetlasz
print(cleaned_df)

# Usuwanie brakujących wartości
cleaned_df.dropna(subset=time_columns, inplace=True)
print(f"Liczba rekordów po usunięciu brakujących: {len(cleaned_df)}")
cleaned_df.isna().sum().reset_index(name='ilość')

# Usuwanie outliers

# Funkcja do policzenia liczby outliers w serii
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers)

# Funkcja do usuwania outliers z DataFrame w podanej kolumnie
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)].copy()

# Lista kolumn, z których chcesz usunąć outliers
columns_to_clean = [
    'Tempo', 
    '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', 'Czas',
    '5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo'
]

# Liczba outliers przed usunięciem
outliers_counts = {}
for col in columns_to_clean:
    if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
        outliers_counts[col] = count_outliers(cleaned_df[col])

print("--- Liczba outliers przed usunięciem ---")
for col, count in outliers_counts.items():
    print(f"{col}: {count}")

# Usuwanie outliers kaskadowo
for col in columns_to_clean:
    if col in cleaned_df.columns and pd.api.types.is_numeric_dtype(cleaned_df[col]):
        cleaned_df = remove_outliers(cleaned_df, col)

print(f"\nLiczba rekordów po usunięciu outliers: {len(cleaned_df)}")

# ========================================
# OD TEGO MIEJSCA NOWA CZĘŚĆ - MODELOWANIE
# ========================================

print("\n" + "="*60)
print("PRZYGOTOWANIE DANYCH DO MODELU")
print("="*60)

# Tylko potrzebne kolumny: płeć, wiek, czas 5km → czas końcowy
model_data = cleaned_df[['5 km Czas', 'Wiek', 'Płeć', 'Czas']].copy()

print(f"📊 Dane treningowe: {len(model_data)} rekordów")
print(f"\nStatystyki:")
print(model_data.describe())

# ========================================
# ANALIZA RZECZYWISTYCH DANYCH
# ========================================

print("\n" + "="*60)
print("ANALIZA KORELACJI W RZECZYWISTYCH DANYCH")
print("="*60)

print("\n📈 KORELACJE:")
print(model_data[['5 km Czas', 'Wiek', 'Czas']].corr())

print("\n👥 Średni czas wg płci:")
print(model_data.groupby('Płeć')['Czas'].agg(['mean', 'std', 'count']))

# Analiza wieku w grupach
model_data_temp = model_data.copy()
model_data_temp['Grupa_wiekowa'] = pd.cut(model_data_temp['Wiek'], 
                                          bins=[0, 30, 40, 50, 60, 100],
                                          labels=['<30', '30-40', '40-50', '50-60', '60+'])
print("\n📊 Średni czas wg wieku:")
age_analysis = model_data_temp.groupby('Grupa_wiekowa')['Czas'].agg(['mean', 'std', 'count'])
print(age_analysis)

# Wizualizacja danych
print("\n📊 Tworzenie wykresów analizy danych...")
try:
    plt.figure(figsize=(15, 5))
    
    sample_data = model_data.sample(min(2000, len(model_data)))
    
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=sample_data, x='Wiek', y='Czas', hue='Płeć', alpha=0.5)
    plt.title('Wiek vs Czas końcowy')
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=sample_data, x='5 km Czas', y='Czas', hue='Płeć', alpha=0.5)
    plt.title('Czas 5km vs Czas końcowy')
    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=sample_data, x='Wiek', y='5 km Czas', hue='Płeć', alpha=0.5)
    plt.title('Wiek vs Czas 5km')
    
    plt.tight_layout()
    plt.savefig('analiza_danych.png', dpi=150, bbox_inches='tight')
    print("✅ Wykresy zapisane jako 'analiza_danych.png'")
except Exception as e:
    print(f"⚠️ Nie można utworzyć wykresów: {e}")

# ========================================
# TRENOWANIE MODELU Z PYCARET
# ========================================

print("\n" + "="*60)
print("TRENOWANIE MODELU")
print("="*60)

from pycaret.regression import setup, compare_models, tune_model, finalize_model, plot_model, save_model, predict_model

# Setup - model sam znajdzie zależności między wiekiem, płcią i czasem 5km
exp = setup(
    data=model_data,
    target='Czas',
    categorical_features=['Płeć'],
    numeric_features=['5 km Czas', 'Wiek'],
    normalize=True,              # Normalizacja pomoże
    polynomial_features=True,    # Model SAM stworzy interakcje!
    polynomial_degree=2,         # Interakcje drugiego stopnia (wiek*czas5km, płeć*wiek itd.)
    remove_outliers=False,       # Już usunęliśmy outliers wcześniej
    session_id=123,
    verbose=False,
)

print("\n🤖 Porównanie modeli...")
best_models = exp.compare_models(
    sort='MAE',
    n_select=3,  # Weź top 3
    include=['lr', 'ridge', 'lasso', 'rf', 'gbr', 'xgboost', 'lightgbm', 'catboost']
)

print("\n⚙️ Tuning najlepszego modelu...")
best_model_tuned = exp.tune_model(
    estimator=best_models[0] if isinstance(best_models, list) else best_models,
    n_iter=100,
    optimize='MAE'
)

# Porównaj przed i po tuningu
print("\n📊 Porównanie przed i po tuningu:")
best_model_original = best_models[0] if isinstance(best_models, list) else best_models
compare_results = exp.compare_models([best_model_original, best_model_tuned])

# Wybierz lepszy model
final_model = exp.finalize_model(best_model_tuned)

# ========================================
# ANALIZA MODELU
# ========================================

print("\n" + "="*60)
print("ANALIZA MODELU")
print("="*60)

print("\n📈 Generowanie wykresów modelu...")
try:
    plot_model(final_model, plot='feature', save=True)
    print("✅ Feature importance zapisany")
except Exception as e:
    print(f"⚠️ Nie można wygenerować feature importance: {e}")

try:
    plot_model(final_model, plot='residuals', save=True)
    print("✅ Residuals plot zapisany")
except Exception as e:
    print(f"⚠️ Nie można wygenerować residuals plot: {e}")

try:
    plot_model(final_model, plot='error', save=True)
    print("✅ Error plot zapisany")
except Exception as e:
    print(f"⚠️ Nie można wygenerować error plot: {e}")

# ========================================
# FUNKCJA PREDYKCJI
# ========================================

def predict_halfmarathon_time(wiek, plec, five_k_time_sec, model):
    """
    Predykcja czasu półmaratonu na podstawie wieku, płci i czasu 5km.
    
    Args:
        wiek (int): wiek biegacza
        plec (str): 'M' lub 'K'
        five_k_time_sec (int): czas 5 km w sekundach
        model: wytrenowany model pycaret
    
    Returns:
        tuple: (czas_str, czas_sekundy)
    """
    test_data = pd.DataFrame([{
        '5 km Czas': five_k_time_sec,
        'Wiek': wiek,
        'Płeć': plec
    }])
    
    prediction = exp.predict_model(model, data=test_data)
    pred_seconds = int(round(prediction['prediction_label'][0]))
    
    hours = pred_seconds // 3600
    minutes = (pred_seconds % 3600) // 60
    seconds = pred_seconds % 60
    
    return f"{hours}:{minutes:02d}:{seconds:02d}", pred_seconds

# ========================================
# TESTOWANIE MODELU
# ========================================

def test_model_predictions(model):
    """Sprawdza czy model przewiduje logicznie"""
    
    print("\n" + "="*60)
    print("🧪 TEST 1: Wpływ WIEKU (stały czas 5km = 1500s = 25:00)")
    print("="*60)
    
    results_wiek = []
    for wiek in [25, 30, 35, 40, 45, 50, 55, 60, 65]:
        pred_str, pred_sec = predict_halfmarathon_time(wiek, 'M', 1500, model)
        results_wiek.append((wiek, pred_sec))
        print(f"Wiek {wiek:2d}: {pred_str} ({pred_sec}s)")
    
    # Sprawdź czy czas rośnie z wiekiem
    times = [t for _, t in results_wiek]
    is_increasing = all(times[i] <= times[i+1] for i in range(len(times)-1))
    
    if is_increasing:
        print(f"\n✅ POPRAWNIE: Czas rośnie z wiekiem!")
    else:
        print(f"\n❌ BŁĄD: Czas NIE rośnie z wiekiem!")
        for i in range(len(results_wiek)-1):
            diff = results_wiek[i+1][1] - results_wiek[i][1]
            print(f"  Wiek {results_wiek[i][0]} → {results_wiek[i+1][0]}: różnica {diff}s")
    
    print("\n" + "="*60)
    print("🧪 TEST 2: Wpływ TEMPA 5KM (stały wiek = 40, mężczyzna)")
    print("="*60)
    
    results_tempo = []
    for czas_5km in [1200, 1350, 1500, 1650, 1800, 1950, 2100]:
        pred_str, pred_sec = predict_halfmarathon_time(40, 'M', czas_5km, model)
        results_tempo.append((czas_5km, pred_sec))
        min_5km = czas_5km // 60
        sec_5km = czas_5km % 60
        print(f"5km: {min_5km:2d}:{sec_5km:02d} → Półmaraton: {pred_str} ({pred_sec}s)")
    
    # Sprawdź czy czas rośnie z wolniejszym tempem 5km
    times = [t for _, t in results_tempo]
    is_increasing = all(times[i] <= times[i+1] for i in range(len(times)-1))
    
    if is_increasing:
        print(f"\n✅ POPRAWNIE: Czas rośnie z wolniejszym tempem 5km!")
    else:
        print(f"\n❌ BŁĄD: Czas NIE rośnie z wolniejszym tempem!")
    
    print("\n" + "="*60)
    print("🧪 TEST 3: Wpływ PŁCI (wiek 40, czas 5km = 1500s)")
    print("="*60)
    
    pred_m_str, pred_m_sec = predict_halfmarathon_time(40, 'M', 1500, model)
    pred_k_str, pred_k_sec = predict_halfmarathon_time(40, 'K', 1500, model)
    
    print(f"Mężczyzna: {pred_m_str} ({pred_m_sec}s)")
    print(f"Kobieta:   {pred_k_str} ({pred_k_sec}s)")
    print(f"Różnica:   {abs(pred_k_sec - pred_m_sec)}s ({abs(pred_k_sec - pred_m_sec)//60} min)")
    
    print("\n" + "="*60)
    print("🧪 TEST 4: Realistyczne różnice")
    print("="*60)
    
    # Młody szybki vs stary wolny
    pred_mlody_str, pred_mlody_sec = predict_halfmarathon_time(25, 'M', 1200, model)
    pred_stary_str, pred_stary_sec = predict_halfmarathon_time(60, 'M', 1800, model)
    
    roznica = pred_stary_sec - pred_mlody_sec
    
    print(f"Młody (25 lat, 5km=20:00): {pred_mlody_str}")
    print(f"Starszy (60 lat, 5km=30:00): {pred_stary_str}")
    print(f"Różnica: {roznica//60} min {roznica%60} sek")
    
    if roznica > 600:
        print(f"\n✅ POPRAWNIE: Różnica realistyczna (>10 min)!")
    else:
        print(f"\n⚠️ UWAGA: Różnica tylko {roznica//60} min")

print("\n" + "="*60)
print("TESTOWANIE MODELU")
print("="*60)

test_model_predictions(final_model)

# ========================================
# ZAPIS MODELU
# ========================================

print("\n" + "="*60)
print("ZAPIS MODELU")
print("="*60)

# Zapisz model lokalnie
save_model(final_model, 'final_regression_pipeline')
print("✅ Model zapisany lokalnie jako 'final_regression_pipeline.pkl'")

# Prześlij do S3
try:
    s3.upload_file(
        Filename='final_regression_pipeline.pkl',
        Bucket=BUCKET_NAME,
        Key='Train_Model/final_regression_pipeline.pkl'
    )
    print("✅ Model przesłany do S3")
except Exception as e:
    print(f"❌ Błąd przesyłania modelu do S3: {e}")

# ========================================
# PRZYKŁAD UŻYCIA
# ========================================

print("\n" + "="*60)
print("PRZYKŁAD UŻYCIA")
print("="*60)

wiek_input = 42
plec_input = 'M'
czas_5km_input = 1500  # 25 minut

predykcja_str, predykcja_sec = predict_halfmarathon_time(wiek_input, plec_input, czas_5km_input, final_model)
print(f"\nDla biegacza:")
print(f"  Wiek: {wiek_input} lat")
print(f"  Płeć: {plec_input}")
print(f"  Czas 5km: {czas_5km_input//60}:{czas_5km_input%60:02d}")
print(f"\nPrzewidywany czas ukończenia półmaratonu: {predykcja_str}")

print("\n" + "="*60)
print("GOTOWE!")
print("="*60)