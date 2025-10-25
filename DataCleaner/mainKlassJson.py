import pandas as pd
import numpy as np
import json
from datetime import datetime

class DataCleaner:
    """
    Eine Klasse zur Bereinigung von Datensätzen mit folgenden Funktionen:
    - Bereinigung von Datumsangaben
    - Behandlung fehlender Werte
    - Entfernung von Duplikaten
    - Behandlung von Ausreißern
    - JSON-Export des Bereinigungsberichts
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
        self.cleaning_report = {
            'timestamp': datetime.now().isoformat(),
            'original_file': file_path,
            'cleaning_steps': {}
        }
        
    def load_data(self):
        """
        Lädt die Daten aus der CSV-Datei
        """
        try:
            self.df_original = pd.read_csv(self.file_path)
            self.df_cleaned = self.df_original.copy()
            print(f"Daten erfolgreich geladen. Form: {self.df_original.shape}")
            
            # Metadaten zum Report hinzufügen
            self.cleaning_report['original_shape'] = {
                'rows': len(self.df_original),
                'columns': len(self.df_original.columns)
            }
            self.cleaning_report['original_columns'] = self.df_original.columns.tolist()
            
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
        
        # Zum Report hinzufügen
        self.cleaning_report['cleaning_steps']['date_cleaning'] = {
            'removed_rows': int(removed_rows),
            'remaining_rows': int(len(self.df_cleaned)),
            'description': 'Entfernung von Zeilen mit fehlenden Datumsangaben und Formatvereinheitlichung'
        }
        
        print(f"Datumsbereinigung: {removed_rows} Zeilen entfernt")
    
    def remove_duplicates(self):
        """
        Entfernt doppelte Zeilen aus dem Datensatz
        """
        original_rows = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.drop_duplicates()
        removed_duplicates = original_rows - len(self.df_cleaned)
        
        # Zum Report hinzufügen
        self.cleaning_report['cleaning_steps']['duplicate_removal'] = {
            'removed_duplicates': int(removed_duplicates),
            'remaining_rows': int(len(self.df_cleaned)),
            'description': 'Entfernung doppelter Zeilen'
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
        mean_values = {}
        
        for col in numerical_columns:
            # Fehlende Werte zählen (vor der Bereinigung)
            missing_before[col] = int(self.df_cleaned[col].isna().sum())
            
            # Zu numerischen Werten konvertieren
            self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], errors='coerce')
            
            # Mittelwert berechnen und fehlende Werte ersetzen
            mean_value = float(self.df_cleaned[col].mean())
            mean_values[col] = mean_value
            self.df_cleaned[col] = self.df_cleaned[col].fillna(mean_value)
            
            # Fehlende Werte zählen (nach der Bereinigung)
            missing_after[col] = int(self.df_cleaned[col].isna().sum())
        
        # Zum Report hinzufügen
        self.cleaning_report['cleaning_steps']['missing_values'] = {
            'numerical_columns': numerical_columns,
            'missing_before': missing_before,
            'missing_after': missing_after,
            'mean_values_used': mean_values,
            'description': 'Ersetzung fehlender Werte durch Spaltenmittelwerte'
        }
        
        print("Fehlende Werte wurden durch Mittelwerte ersetzt")
    
    def detect_outliers(self, column, std_threshold=3):
        """
        Erkennt Ausreißer in einer Spalte
        
        Args:
            column (str): Spaltenname
            std_threshold (float): Schwellenwert für Standardabweichungen
        
        Returns:
            tuple: (lower_bound, upper_bound, outlier_indices, outlier_values)
        """
        data = self.df_cleaned[column]
        mean = float(data.mean())
        std = float(data.std())
        
        lower_bound = float(mean - std_threshold * std)
        upper_bound = float(mean + std_threshold * std)
        
        # Ausreißer identifizieren
        outlier_mask = (self.df_cleaned[column] < lower_bound) | (self.df_cleaned[column] > upper_bound)
        outlier_data = self.df_cleaned[outlier_mask][column]
        
        outlier_indices = outlier_data.index.tolist()
        outlier_values = outlier_data.tolist()
        
        return lower_bound, upper_bound, outlier_indices, outlier_values
    
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
            lower_bound, upper_bound, outlier_indices, outlier_values = self.detect_outliers(col, std_threshold)
            
            # Ausreißer entfernen
            self.df_cleaned = self.df_cleaned[~self.df_cleaned.index.isin(outlier_indices)]
            
            outlier_report[col] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outlier_indices),
                'outlier_indices': outlier_indices,
                'outlier_values': outlier_values,
                'statistics': {
                    'mean': float(self.df_cleaned[col].mean()),
                    'std': float(self.df_cleaned[col].std())
                }
            }
            
            print(f"Ausreißer in {col}: {len(outlier_indices)} entfernt")
        
        removed_rows = original_rows - len(self.df_cleaned)
        
        # Zum Report hinzufügen
        self.cleaning_report['cleaning_steps']['outlier_removal'] = {
            'removed_rows': int(removed_rows),
            'remaining_rows': int(len(self.df_cleaned)),
            'std_threshold': std_threshold,
            'columns_analyzed': columns,
            'outlier_details': outlier_report,
            'description': f'Entfernung von Ausreißern basierend auf {std_threshold} Standardabweichungen'
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
        
        # Finale Metriken zum Report hinzufügen
        self.cleaning_report['final_shape'] = {
            'rows': int(len(self.df_cleaned)),
            'columns': int(len(self.df_cleaned.columns))
        }
        self.cleaning_report['cleaning_summary'] = {
            'total_rows_removed': int(len(self.df_original) - len(self.df_cleaned)),
            'percentage_remaining': float(len(self.df_cleaned) / len(self.df_original) * 100),
            'cleaning_completed': True
        }
        
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
    
    def get_cleaning_report(self):
        """Gibt den detaillierten Bereinigungsbericht zurück"""
        return self.cleaning_report
    
    def save_cleaning_report(self, output_path='cleaning_report.json'):
        """
        Speichert den Bereinigungsbericht als JSON-Datei
        
        Args:
            output_path (str): Pfad für die JSON-Datei
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.cleaning_report, f, indent=2, ensure_ascii=False)
            print(f"Bereinigungsbericht gespeichert in: {output_path}")
        except Exception as e:
            print(f"Fehler beim Speichern des Berichts: {e}")
    
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
    
    def print_detailed_report(self):
        """Gibt einen detaillierten Bericht in der Konsole aus"""
        report = self.get_cleaning_report()
        print("\n" + "="*60)
        print("DETAILLIERTER BEREINIGUNGSBERICHT")
        print("="*60)
        print(json.dumps(report, indent=2, ensure_ascii=False))


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
        
        # Detaillierten Bericht in Konsole anzeigen
        #cleaner.print_detailed_report()
        
        # Bereinigungsbericht als JSON speichern
        cleaner.save_cleaning_report('cleaning_report.json')
        
        print("\n✅ Alle Bereinigungen abgeschlossen und Berichte gespeichert!")