import pandas as pd
import numpy as np

# 1. Datensatz einlesen
df = pd.read_csv('K4.0026_1.4.4.Ü.01_dirtydata.csv')

print("Originaler Datensatz:")
print(df.head())
print(f"\nForm: {df.shape}")

# 2. Datumsangaben korrigieren
# Leere Datumsangaben löschen
df = df.dropna(subset=['Datum'])

# Datumsformat vereinheitlichen (20201226 → '2020/12/26')
df['Datum'] = df['Datum'].astype(str).str.replace(r'(\d{4})(\d{2})(\d{2})', r'\1/\2/\3', regex=True)
# Anführungszeichen entfernen
df['Datum'] = df['Datum'].str.replace("'", "")

# 3. Duplikate entfernen
df = df.drop_duplicates()

# 4. Numerische Spalten: Leere Zellen durch Mittelwert ersetzen
numerical_columns = ['Dauer', 'Kunden', 'MinKauf', 'MaxKauf']

for col in numerical_columns:
    # Leere Strings und NaN-Werte durch NaN ersetzen
    df[col] = pd.to_numeric(df[col], errors='coerce')
    # Mittelwert berechnen (ignoriert NaN)
    mean_value = df[col].mean()
    # NaN-Werte durch Mittelwert ersetzen
    df[col] = df[col].fillna(mean_value)

print("\nNach Bereinigung fehlender Werte:")
print(f"Form: {df.shape}")

# 5. Ausreißer behandeln - KORRIGIERTE VERSION
def remove_outliers(df, column):
    """Entfernt Ausreißer basierend auf 3 Standardabweichungen"""
    mean = df[column].mean()
    std = df[column].std()
    
    # Filter für Werte innerhalb von 3 Standardabweichungen
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    
    print(f"\n--- {column} Ausreißeranalyse ---")
    print(f"Mittelwert: {mean:.2f}")
    print(f"Standardabweichung: {std:.2f}")
    print(f"Grenzen: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Zeilen identifizieren, die Ausreißer enthalten
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"Gefundene Ausreißer: {len(outliers)}")
    
    if len(outliers) > 0:
        print("Ausreißer-Werte:")
        for idx, value in outliers[column].items():
            print(f"  Zeile {idx}: {value}")
    
    # Nur Zeilen behalten, die innerhalb der Grenzen liegen
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Ausreißer Schritt für Schritt entfernen
df_clean = df.copy()

for col in ['Kunden', 'MinKauf', 'MaxKauf']:
    original_size = len(df_clean)
    df_clean = remove_outliers(df_clean, col)
    removed = original_size - len(df_clean)
    print(f"Entfernte Zeilen für {col}: {removed}")

print("\nBereinigter Datensatz:")
print(df_clean.head())
print(f"\nForm nach Bereinigung: {df_clean.shape}")

# Zusammenfassung der Änderungen
print("\n=== ZUSAMMENFASSUNG ===")
print(f"Ursprüngliche Zeilen: {len(df)}")
print(f"Verbleibende Zeilen nach Bereinigung: {len(df_clean)}")
print(f"Entfernte Zeilen: {len(df) - len(df_clean)}")

print("\nDatentypen nach Bereinigung:")
print(df_clean.dtypes)

print("\nFehlende Werte nach Bereinigung:")
print(df_clean.isnull().sum())

# Bereinigte Daten speichern
df_clean.to_csv('cleaned_data_pandas.csv', index=False)
print("\nBereinigte Daten wurden in 'cleaned_data_pandas.csv' gespeichert!")

# Statistische Übersicht der bereinigten Daten
print("\n=== STATISTISCHE ÜBERSICHT (bereinigt) ===")
print(df_clean[['Kunden', 'MinKauf', 'MaxKauf']].describe())