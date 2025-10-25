"""
Weather Analysis Project
Analyse von Wetterdaten für die Entscheidung "Draussen Essen"
Autor: [Ufuk Baysal]
Datum: [21.10.2025]
"""
import os
import sys
import importlib
import subprocess
import pandas as pd
import numpy as np
from math import log2
from scipy.stats import entropy

# ------------------- Pakete installieren falls nicht vorhanden -------------------
def install_requirements():
    req_file = "requirements.txt"
    if os.path.exists(req_file):
        print(f"Installiere Pakete aus {req_file}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file,"--quiet"])
    else:
        print(f"{req_file} nicht gefunden. Überprüfe manuell oder installiere fehlende Pakete...")
        # Fallback: interne Liste
        required_packages = ["matplotlib", "reportlab"]
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                print(f"{package} wird installiert...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Installation prüfen / Pakete installieren
install_requirements()

class DataAnalyzer:
    """
    Hauptklasse für die Datenanalyse der Wetterdaten
    UML STRUCTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        DataAnalyzer                             │
    ├─────────────────────────────────────────────────────────────────┤
    │ - file_path: str                                                │
    │ - df1: DataFrame                                                │
    │ - df2: DataFrame                                                │
    │ - df2_categorized: DataFrame                                    │
    ├─────────────────────────────────────────────────────────────────┤
    │ + __init__(file_path: str)                                      │
    │ + load_data()                                                   │
    │ + calculate_information_gain() -> float                         │
    │ + categorize_numerical_data()                                   │
    │ + analyze_sheet1() -> dict                                      │
    │ + analyze_sheet2() -> dict                                      │
    │ + perform_statistical_analysis() -> tuple                       │
    │ + additional_analyses()                                         │
    │ + generate_summary()                                            │
    │ + run_complete_analysis()                                       │
    └─────────────────────────────────────────────────────────────────┘
    
    METHOD CATEGORIES:
    • Data Management: load_data(), categorize_numerical_data()
    • Analysis: calculate_information_gain(), analyze_sheet1(), analyze_sheet2()
    • Statistics: perform_statistical_analysis(), additional_analyses()  
    • Control: run_complete_analysis(), generate_summary()
    
    """
    
    def __init__(self, file_path):
        """
        Initialisiert den DataAnalyzer mit dem Pfad zur Excel-Datei
        
        Args:
            file_path (str): Pfad zur Excel-Datei
        """
        self.file_path = file_path
        self.df1 = None  # Tabellenblatt 1 (kategorische Daten)
        self.df2 = None  # Tabellenblatt 2 (numerische Daten)
        self.df2_categorized = None  # Kategorisierte Version von df2
        
    def load_data(self):
        """
        Lädt die Daten aus der Excel-Datei
        """
        try:
            # Tabellenblatt 1 (kategorische Daten)
            self.df1 = pd.read_excel(self.file_path, sheet_name='Tabellenblatt1')
            # Tabellenblatt 2 (numerische Daten)
            self.df2 = pd.read_excel(self.file_path, sheet_name='Tabellenblatt2')
            print("✅ Daten erfolgreich geladen")
        except Exception as e:
            print(f"❌ Fehler beim Laden der Daten: {e}")
    
    def calculate_information_gain(self, data, feature, target):
        """
        Berechnet den Informationsgewinn für ein Feature bezüglich der Zielvariable
        
        Args:
            data (DataFrame): Datensatz
            feature (str): Name des Features
            target (str): Name der Zielvariable
            
        Returns:
            float: Informationsgewinn
        """
        # Entropie der Zielvariable berechnen
        target_counts = data[target].value_counts()
        total_entropy = entropy(target_counts / len(data), base=2)
        
        # Bedingte Entropie berechnen
        feature_values = data[feature].unique()
        weighted_entropy = 0
        
        for value in feature_values:
            # Teilmengen für jeden Feature-Wert erstellen
            subset = data[data[feature] == value]
            if len(subset) > 0:  # Vermeidung von Division durch Null
                subset_target_counts = subset[target].value_counts()
                # Entropie für die Teilmenge berechnen
                subset_entropy = entropy(subset_target_counts / len(subset), base=2)
                # Gewichtete Entropie addieren
                weighted_entropy += (len(subset) / len(data)) * subset_entropy
        
        # Informationsgewinn = Gesamtentropie - bedingte Entropie
        info_gain = total_entropy - weighted_entropy
        return info_gain
    
    def categorize_numerical_data(self):
        """
        Kategorisiert die numerischen Daten aus Tabellenblatt 2
        gemäß den vorgegebenen Kriterien
        """
        if self.df2 is None:
            print("❌ Daten müssen zuerst geladen werden")
            return
        
        df_categorized = self.df2.copy()
        
        # Temperatur kategorisieren
        # < 18°C: kalt, 18-28°C: mild, > 28°C: heiss
        temperature_bins = [-np.inf, 18, 28, np.inf]
        temperature_labels = ['kalt', 'mild', 'heiss']
        df_categorized['Temperatur_kategorie'] = pd.cut(
            df_categorized['Temperatur'], 
            bins=temperature_bins, 
            labels=temperature_labels
        )
        
        # Luftfeuchtigkeit kategorisieren
        # > 5: hoch, <= 5: niedrig
        humidity_bins = [-np.inf, 5, np.inf]
        humidity_labels = ['niedrig', 'hoch']
        df_categorized['Luftfeuchtigkeit_kategorie'] = pd.cut(
            df_categorized['Luftfeuchtigkeit'], 
            bins=humidity_bins, 
            labels=humidity_labels
        )
        
        # Windgeschwindigkeit kategorisieren
        # 1-5 km/h, 6-11 km/h, 12-19 km/h, >19 km/h
        wind_bins = [0, 5, 11, 19, np.inf]
        wind_labels = ['1-5 km/h', '6-11 km/h', '12-19 km/h', '>19 km/h']
        df_categorized['Windgeschwindigkeit_kategorie'] = pd.cut(
            df_categorized['Windgeschwindigkeit'], 
            bins=wind_bins, 
            labels=wind_labels
        )
        
        self.df2_categorized = df_categorized
        print("✅ Numerische Daten erfolgreich kategorisiert")
    
    def analyze_sheet1(self):
        """
        Führt die Analyse für Tabellenblatt 1 durch
        """
        if self.df1 is None:
            print("❌ Daten müssen zuerst geladen werden")
            return
        
        print("\n" + "="*60)
        print("ANALYSE TABELLENBLATT 1 (Kategorische Daten)")
        print("="*60)
        
        print("\nDatenübersicht:")
        print(self.df1.head())
        
        print(f"\nDatensätze: {len(self.df1)}")
        print(f"Features: {list(self.df1.columns)}")
        
        print("\n📊 INFORMATIONSGEWINN für 'Draussen Essen':")
        print("-" * 40)
        
        # Features für Tabellenblatt 1
        features_tb1 = ['Wetteraussicht', 'Temperaturkategorie', 'Luftfeuchtigkeit', 'Windstärke']
        
        information_gains = {}
        for feature in features_tb1:
            ig = self.calculate_information_gain(self.df1, feature, 'Draussen Essen')
            information_gains[feature] = ig
            print(f"  {feature:20}: {ig:.4f}")
        
        # Wichtigstes Feature identifizieren
        best_feature = max(information_gains, key=information_gains.get)
        print(f"\n🎯 Wichtigstes Feature: '{best_feature}' (IG: {information_gains[best_feature]:.4f})")
        
        return information_gains
    
    def analyze_sheet2(self):
        """
        Führt die Analyse für Tabellenblatt 2 durch
        """
        if self.df2 is None or self.df2_categorized is None:
            print("❌ Daten müssen zuerst geladen und kategorisiert werden")
            return
        
        print("\n" + "="*60)
        print("ANALYSE TABELLENBLATT 2 (Numerische Daten)")
        print("="*60)
        
        print("\nDatenübersicht (Original):")
        print(self.df2.head())
        
        print("\nDatenübersicht (Kategorisiert):")
        print(self.df2_categorized[[
            'Temperatur', 'Temperatur_kategorie', 
            'Luftfeuchtigkeit', 'Luftfeuchtigkeit_kategorie',
            'Windgeschwindigkeit', 'Windgeschwindigkeit_kategorie',
            'Aussenverkauf'
        ]].head())
        
        print(f"\nDatensätze: {len(self.df2)}")
        
        print("\n📊 INFORMATIONSGEWINN für 'Aussenverkauf':")
        print("-" * 40)
        
        # Features für Tabellenblatt 2 (kategorisiert)
        features_tb2 = [
            'Temperatur_kategorie', 
            'Luftfeuchtigkeit_kategorie', 
            'Windgeschwindigkeit_kategorie'
        ]
        
        information_gains = {}
        for feature in features_tb2:
            ig = self.calculate_information_gain(self.df2_categorized, feature, 'Aussenverkauf')
            information_gains[feature] = ig
            print(f"  {feature:30}: {ig:.4f}")
        
        # Wichtigstes Feature identifizieren
        if information_gains:
            best_feature = max(information_gains, key=information_gains.get)
            print(f"\n🎯 Wichtigstes Feature: '{best_feature}' (IG: {information_gains[best_feature]:.4f})")
        
        return information_gains
    
    def perform_statistical_analysis(self):
        """
        Führt die statistische Analyse für Tabellenblatt 2 durch
        """
        if self.df2 is None:
            print("❌ Daten müssen zuerst geladen werden")
            return
        
        print("\n" + "="*60)
        print("STATISTISCHE ANALYSE TABELLENBLATT 2")
        print("="*60)
        
        # Numerische Features für die statistische Analyse
        numeric_features = ['Temperatur', 'Luftfeuchtigkeit', 'Windgeschwindigkeit']
        
        print("\n📈 STATISTISCHE KENNZAHLEN:")
        print("-" * 50)
        
        stats_results = {}
        for feature in numeric_features:
            stats = {
                'Mittelwert': self.df2[feature].mean(),
                'Median': self.df2[feature].median(),
                'Standardabweichung': self.df2[feature].std(),
                'Minimum': self.df2[feature].min(),
                'Maximum': self.df2[feature].max()
            }
            stats_results[feature] = stats
            
            print(f"\n{feature}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name:20}: {value:.4f}")
        
        print("\n📊 KORRELATIONEN:")
        print("-" * 30)
        
        # Korrelationen berechnen
        corr_temp_humidity = self.df2['Temperatur'].corr(self.df2['Luftfeuchtigkeit'])
        corr_wind_humidity = self.df2['Windgeschwindigkeit'].corr(self.df2['Luftfeuchtigkeit'])
        corr_wind_temp = self.df2['Windgeschwindigkeit'].corr(self.df2['Temperatur'])
        
        print(f"  Temperatur - Luftfeuchtigkeit    : {corr_temp_humidity:.4f}")
        print(f"  Windgeschwindigkeit - Luftfeuchtigkeit: {corr_wind_humidity:.4f}")
        print(f"  Windgeschwindigkeit - Temperatur : {corr_wind_temp:.4f}")
        
        return stats_results, {
            'Temperatur-Luftfeuchtigkeit': corr_temp_humidity,
            'Wind-Luftfeuchtigkeit': corr_wind_humidity,
            'Wind-Temperatur': corr_wind_temp
        }
    
    def additional_analyses(self):
        """
        Führt zusätzliche Analysen durch
        """
        if self.df2_categorized is None:
            print("❌ Kategorisierte Daten benötigt")
            return
        
        print("\n" + "="*60)
        print("ZUSÄTZLICHE ANALYSEN")
        print("="*60)
        
        print("\n📋 VERTEILUNG DER KATEGORIEN:")
        print("-" * 35)
        
        print("\nTemperaturkategorien:")
        temp_dist = self.df2_categorized['Temperatur_kategorie'].value_counts()
        for category, count in temp_dist.items():
            print(f"  {category}: {count} Datensätze")
        
        print("\nLuftfeuchtigkeitskategorien:")
        humidity_dist = self.df2_categorized['Luftfeuchtigkeit_kategorie'].value_counts()
        for category, count in humidity_dist.items():
            print(f"  {category}: {count} Datensätze")
        
        print("\nWindgeschwindigkeitskategorien:")
        wind_dist = self.df2_categorized['Windgeschwindigkeit_kategorie'].value_counts()
        for category, count in wind_dist.items():
            print(f"  {category}: {count} Datensätze")
        
        print("\n🔍 AUSSENVERKAUF NACH KATEGORIEN:")
        print("-" * 35)
        
        features_tb2 = [
            'Temperatur_kategorie', 
            'Luftfeuchtigkeit_kategorie', 
            'Windgeschwindigkeit_kategorie'
        ]
        
        for feature in features_tb2:
            print(f"\n{feature}:")
            cross_tab = pd.crosstab(
                self.df2_categorized[feature], 
                self.df2_categorized['Aussenverkauf']
            )
            print(cross_tab)
    
    def generate_summary(self, ig_tb1, ig_tb2, correlations):
        """
        Erstellt eine Zusammenfassung der wichtigsten Erkenntnisse
        
        Args:
            ig_tb1 (dict): Informationsgewinne Tabellenblatt 1
            ig_tb2 (dict): Informationsgewinne Tabellenblatt 2
            correlations (dict): Korrelationskoeffizienten
        """
        print("\n" + "="*60)
        print("📋 ZUSAMMENFASSUNG DER ERGEBNISSE")
        print("="*60)
        
        # Wichtigste Features identifizieren
        if ig_tb1:
            best_tb1 = max(ig_tb1, key=ig_tb1.get)
            print(f"\n🎯 TABELLENBLATT 1 - Wichtigstes Feature:")
            print(f"   '{best_tb1}' mit Informationsgewinn: {ig_tb1[best_tb1]:.4f}")
        
        if ig_tb2:
            best_tb2 = max(ig_tb2, key=ig_tb2.get)
            print(f"\n🎯 TABELLENBLATT 2 - Wichtigstes Feature:")
            print(f"   '{best_tb2}' mit Informationsgewinn: {ig_tb2[best_tb2]:.4f}")
        
        if correlations:
            strongest_corr = max(correlations, key=correlations.get)
            print(f"\n📊 STÄRKSTE KORRELATION:")
            print(f"   '{strongest_corr}' mit Wert: {correlations[strongest_corr]:.4f}")
        
        print("\n💡 INTERPRETATION:")
        print("   - Informationsgewinn > 0: Feature ist relevant für die Entscheidung")
        print("   - Höherer Informationsgewinn = größerer Einfluss auf die Entscheidung")
        print("   - Korrelation nahe 0: Kein linearer Zusammenhang")
        print("   - Korrelation nahe ±1: Starker linearer Zusammenhang")
    
    def run_complete_analysis(self):
        """
        Führt die komplette Analyse durch
        """
        print("🚀 STARTE KOMPLETTE ANALYSE")
        print("="*60)
        
        # 1. Daten laden
        self.load_data()
        
        # 2. Numerische Daten kategorisieren
        self.categorize_numerical_data()
        
        # 3. Analysen durchführen
        ig_tb1 = self.analyze_sheet1()
        ig_tb2 = self.analyze_sheet2()
        stats, correlations = self.perform_statistical_analysis()
        self.additional_analyses()
        
        # 4. Zusammenfassung erstellen
        self.generate_summary(ig_tb1, ig_tb2, correlations)
        
        print("\n✅ Analyse erfolgreich abgeschlossen!")


def main():
    """
    Hauptfunktion des Programms
    """
    # Pfad zur Excel-Datei (anpassen falls nötig)
    file_path = 'K4.0026_1.5.C.01_ProjectData.xlsx'
    
    # DataAnalyzer instanziieren
    analyzer = DataAnalyzer(file_path)
    
    # Komplette Analyse durchführen
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()