import pandas as pd
import numpy as np

class DataCleaner:
    """
    Eine Klasse zur Bereinigung von Datensätzen mit folgenden Funktionen:
    - Bereinigung von Datumsangaben
    - Behandlung fehlender Werte
    - Entfernung von Duplikaten
    - Behandlung von Ausreißern
    """
    
    def __init__(self, file_path):
        """
        Initialisiert den DataCleaner mit einem Datensatz
        
        Args:
            file_path (str): Pfad zur CSV-Datei
        """
        self.file_path = file_path
        self.df_original = None
        self.df_cleaned = None
        self.cleaning_report = {}
        
    def load_data(self):
        """
        Lädt die Daten aus der CSV-Datei
        """
        try:
            self.df_original = pd.read_csv(self.file_path)
            self.df_cleaned = self.df_original.copy()
            print(f"Daten erfolgreich geladen. Form: {self.df_original.shape}")
            return True
        except Exception as e:
            print(f"Fehler beim Laden der Daten: {e}")
            return False
    
    def clean_dates(self):
        """
        Bereinigt Datumsangaben:
        - Entfernt Zeilen mit fehlenden Datumsangaben
        - Vereinheitlicht Datumsformate
        - Entfernt Anführungszeichen
        """
        original_rows = len(self.df_cleaned)
        
        # Zeilen mit fehlenden Datumsangaben entfernen
        self.df_cleaned = self.df_cleaned.dropna(subset=['Datum'])
        
        # Datumsformat vereinheitlichen
        self.df_cleaned['Datum'] = (
            self.df_cleaned['Datum']
            .astype(str)
            .str.replace(r'(\d{4})(\d{2})(\d{2})', r'\1/\2/\3', regex=True)
            .str.replace("'", "")
        )
        
        removed_rows = original_rows - len(self.df_cleaned)
        self.cleaning_report['date_cleaning'] = {
            'removed_rows': removed_rows,
            'remaining_rows': len(self.df_cleaned)
        }
        
        print(f"Datumsbereinigung: {removed_rows} Zeilen entfernt")
    
    def remove_duplicates(self):
        """
        Entfernt doppelte Zeilen aus dem Datensatz
        """
        original_rows = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.drop_duplicates()
        removed_duplicates = original_rows - len(self.df_cleaned)
        
        self.cleaning_report['duplicate_removal'] = {
            'removed_duplicates': removed_duplicates,
            'remaining_rows': len(self.df_cleaned)
        }
        
        print(f"Duplikate: {removed_duplicates} doppelte Zeilen entfernt")
    
    def handle_missing_values(self, numerical_columns=None):
        """
        Behandelt fehlende Werte in numerischen Spalten
        
        Args:
            numerical_columns (list): Liste der numerischen Spalten
        """
        if numerical_columns is None:
            numerical_columns = ['Dauer', 'Kunden', 'MinKauf', 'MaxKauf']
        
        missing_before = {}
        missing_after = {}
        
        for col in numerical_columns:
            # Fehlende Werte zählen (vor der Bereinigung)
            missing_before[col] = self.df_cleaned[col].isna().sum()
            
            # Zu numerischen Werten konvertieren
            self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], errors='coerce')
            
            # Mittelwert berechnen und fehlende Werte ersetzen
            mean_value = self.df_cleaned[col].mean()
            self.df_cleaned[col] = self.df_cleaned[col].fillna(mean_value)
            
            # Fehlende Werte zählen (nach der Bereinigung)
            missing_after[col] = self.df_cleaned[col].isna().sum()
        
        self.cleaning_report['missing_values'] = {
            'before': missing_before,
            'after': missing_after,
            'numerical_columns': numerical_columns
        }
        
        print("Fehlende Werte wurden durch Mittelwerte ersetzt")
    
    def detect_outliers(self, column, std_threshold=3):
        """
        Erkennt Ausreißer in einer Spalte
        
        Args:
            column (str): Spaltenname
            std_threshold (float): Schwellenwert für Standardabweichungen
        
        Returns:
            tuple: (lower_bound, upper_bound, outlier_indices)
        """
        data = self.df_cleaned[column]
        mean = data.mean()
        std = data.std()
        
        lower_bound = mean - std_threshold * std
        upper_bound = mean + std_threshold * std
        
        # Indizes der Ausreißer
        outlier_indices = self.df_cleaned[
            (self.df_cleaned[column] < lower_bound) | 
            (self.df_cleaned[column] > upper_bound)
        ].index.tolist()
        
        return lower_bound, upper_bound, outlier_indices
    
    def remove_outliers(self, columns=None, std_threshold=3):
        """
        Entfernt Ausreißer aus den angegebenen Spalten
        
        Args:
            columns (list): Liste der Spalten für Ausreißer-Bereinigung
            std_threshold (float): Schwellenwert für Standardabweichungen
        """
        if columns is None:
            columns = ['Kunden', 'MinKauf', 'MaxKauf']
        
        original_rows = len(self.df_cleaned)
        outlier_report = {}
        
        for col in columns:
            lower_bound, upper_bound, outlier_indices = self.detect_outliers(col, std_threshold)
            
            # Ausreißer entfernen
            self.df_cleaned = self.df_cleaned[~self.df_cleaned.index.isin(outlier_indices)]
            
            outlier_report[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outlier_indices),
                'outlier_indices': outlier_indices
            }
            
            print(f"Ausreißer in {col}: {len(outlier_indices)} entfernt")
        
        removed_rows = original_rows - len(self.df_cleaned)
        self.cleaning_report['outlier_removal'] = {
            'removed_rows': removed_rows,
            'remaining_rows': len(self.df_cleaned),
            'outlier_details': outlier_report
        }
    
    def clean_all(self, numerical_columns=None, outlier_columns=None, std_threshold=3):
        """
        Führt alle Bereinigungsschritte durch
        
        Args:
            numerical_columns (list): Numerische Spalten für fehlende Werte
            outlier_columns (list): Spalten für Ausreißer-Bereinigung
            std_threshold (float): Schwellenwert für Ausreißer-Erkennung
        """
        if numerical_columns is None:
            numerical_columns = ['Dauer', 'Kunden', 'MinKauf', 'MaxKauf']
        if outlier_columns is None:
            outlier_columns = ['Kunden', 'MinKauf', 'MaxKauf']
        
        print("=== BEGINN DER DATENBEREINIGUNG ===")
        
        # Schritt 1: Datumsbereinigung
        self.clean_dates()
        
        # Schritt 2: Duplikate entfernen
        self.remove_duplicates()
        
        # Schritt 3: Fehlende Werte behandeln
        self.handle_missing_values(numerical_columns)
        
        # Schritt 4: Ausreißer entfernen
        self.remove_outliers(outlier_columns, std_threshold)
        
        print("=== DATENBEREINIGUNG ABGESCHLOSSEN ===")
        self.generate_summary()
    
    def generate_summary(self):
        """Generiert eine Zusammenfassung der Bereinigung"""
        print("\n" + "="*50)
        print("ZUSAMMENFASSUNG DER DATENBEREINIGUNG")
        print("="*50)
        
        if self.df_original is not None and self.df_cleaned is not None:
            print(f"Ursprüngliche Zeilen: {len(self.df_original)}")
            print(f"Bereinigte Zeilen: {len(self.df_cleaned)}")
            print(f"Entfernte Zeilen: {len(self.df_original) - len(self.df_cleaned)}")
            print(f"Verbleibende Daten: {len(self.df_cleaned)/len(self.df_original)*100:.1f}%")
        
        print(f"\nBereinigte Daten - Form: {self.df_cleaned.shape}")
        print("\nDatentypen:")
        print(self.df_cleaned.dtypes)
        
        print("\nFehlende Werte nach Bereinigung:")
        print(self.df_cleaned.isnull().sum())
    
    def get_cleaning_report(self):
        """Gibt den detaillierten Bereinigungsbericht zurück"""
        return self.cleaning_report
    
    def save_cleaned_data(self, output_path='cleaned_data.csv'):
        """
        Speichert die bereinigten Daten
        
        Args:
            output_path (str): Pfad für die Ausgabedatei
        """
        try:
            self.df_cleaned.to_csv(output_path, index=False)
            print(f"Bereinigte Daten gespeichert in: {output_path}")
        except Exception as e:
            print(f"Fehler beim Speichern: {e}")
    
    def get_original_data(self):
        """Gibt die originalen Daten zurück"""
        return self.df_original
    
    def get_cleaned_data(self):
        """Gibt die bereinigten Daten zurück"""
        return self.df_cleaned


# Verwendung der Klasse
if __name__ == "__main__":
    # DataCleaner instanziieren
    cleaner = DataCleaner('K4.0026_1.4.4.Ü.01_dirtydata.csv')
    
    # Daten laden
    if cleaner.load_data():
        # Komplette Bereinigung durchführen
        cleaner.clean_all()
        
        # Bereinigte Daten anzeigen
        print("\nErste 5 Zeilen der bereinigten Daten:")
        print(cleaner.get_cleaned_data().head())
        
        # Bereinigte Daten speichern
        cleaner.save_cleaned_data('cleaned_data_class.csv')
        
        # Detaillierten Bericht anzeigen
        report = cleaner.get_cleaning_report()
        print("\nDetaillierter Bereinigungsbericht verfügbar")