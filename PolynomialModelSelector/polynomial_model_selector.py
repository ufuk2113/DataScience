import os
import sys
import importlib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Tuple, List, Dict

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

class PolynomialModelSelector:
    """
    Eine Klasse zur Durchführung einer k-fachen Kreuzvalidierung für verschiedene Polynomgrade.
+-----------------------------------------------------------------------+
|                  PolynomialModelSelector                              |
+-----------------------------------------------------------------------+
| - k_folds: int                                                        |
| - degrees: List[int]                                                  |
| - random_state: int                                                   |
| - results: Dict                                                       |
| - best_degree: int                                                    |
| - best_score: float                                                   |
+-----------------------------------------------------------------------+
| + __init__(k_folds: int = 5,                                          |
|            degrees: List[int] = None,                                 |
|            random_state: int = 42)                                    |
| + cross_validate(X: np.ndarray, y: np.ndarray) -> Dict                |
| - _train_model(X_train: np.ndarray, y_train: np.ndarray,              |
|               X_val: np.ndarray, degree: int)                         |
|               -> Tuple[np.ndarray, Pipeline]                          |
| - _evaluate_model(y_true: np.ndarray, y_pred: np.ndarray)             |
|               -> Tuple[float, float]                                  |
| + get_best_degree() -> Tuple[int, float]                              |
| + plot_results(save_path: str = None) -> None                         |
| + get_results_dataframe() -> pd.DataFrame                             |
+-----------------------------------------------------------------------+
    """
    
    def __init__(self, k_folds: int = 5, degrees: List[int] = None, random_state: int = 42):
        """
        Initialisiert die Kreuzvalidierungsklasse.
        
        Args:
            k_folds (int): Anzahl der Folds für die Kreuzvalidierung
            degrees (list): Liste der zu testenden Polynomgrade
            random_state (int): Seed für Reproduzierbarkeit
        """
        self.k_folds = k_folds
        self.degrees = degrees if degrees is not None else [1, 2, 3, 4]
        self.random_state = random_state
        self.results = {}
        self.best_degree = None
        self.best_score = None
        
    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Führt die k-fache Kreuzvalidierung für alle Polynomgrade durch.
        
        Args:
            X (np.ndarray): Feature-Matrix
            y (np.ndarray): Zielvariable
            
        Returns:
            dict: Ergebnisse der Kreuzvalidierung
        """
        # Datenvalidierung
        if len(X) != len(y):
            raise ValueError("X und y müssen die gleiche Länge haben")
        
        if len(X) < self.k_folds:
            raise ValueError(f"Nicht genug Datenpunkte für {self.k_folds} Folds")
        
        # KFold-Splitter initialisieren
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        
        # Ergebnisse-Dictionary initialisieren
        self.results = {
            'degrees': self.degrees,
            'mean_mse': [],
            'std_mse': [],
            'mean_r2': [],
            'std_r2': [],
            'fold_scores': {degree: {'mse': [], 'r2': []} for degree in self.degrees}
        }
        
        # Für jeden Polynomgrad Kreuzvalidierung durchführen
        for degree in self.degrees:
            print(f"Teste Polynomgrad {degree}...")
            
            mse_scores = []
            r2_scores = []
            
            # K-fache Kreuzvalidierung
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Modell trainieren und evaluieren
                y_pred, model = self._train_model(X_train, y_train, X_val, degree)
                
                # Metriken berechnen
                mse, r2 = self._evaluate_model(y_val, y_pred)
                
                mse_scores.append(mse)
                r2_scores.append(r2)
                
                # Fold-Ergebnisse speichern
                self.results['fold_scores'][degree]['mse'].append(mse)
                self.results['fold_scores'][degree]['r2'].append(r2)
                
                print(f"  Fold {fold + 1}: MSE = {mse:.4f}, R² = {r2:.4f}")
            
            # Statistik über alle Folds berechnen
            mean_mse = np.mean(mse_scores)
            std_mse = np.std(mse_scores)
            mean_r2 = np.mean(r2_scores)
            std_r2 = np.std(r2_scores)
            
            self.results['mean_mse'].append(mean_mse)
            self.results['std_mse'].append(std_mse)
            self.results['mean_r2'].append(mean_r2)
            self.results['std_r2'].append(std_r2)
            
            print(f"Grad {degree}: MSE = {mean_mse:.4f} ± {std_mse:.4f}, R² = {mean_r2:.4f} ± {std_r2:.4f}")
            print("-" * 50)
        
        # Besten Grad bestimmen (niedrigster MSE)
        best_idx = np.argmin(self.results['mean_mse'])
        self.best_degree = self.degrees[best_idx]
        self.best_score = self.results['mean_mse'][best_idx]
        
        print(f"\nBester Polynomgrad: {self.best_degree} mit MSE = {self.best_score:.4f}")
        
        return self.results
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, degree: int) -> Tuple[np.ndarray, Pipeline]:
        """
        Trainiert ein Polynommodell für den gegebenen Grad.
        
        Args:
            X_train: Trainingsdaten
            y_train: Trainingslabels
            X_val: Validierungsdaten
            degree: Polynomgrad
            
        Returns:
            tuple: (Vorhersagen, trainiertes Modell)
        """
        # Pipeline mit Polynomial Features und Linear Regression
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        
        # Modell trainieren
        model.fit(X_train, y_train)
        
        # Vorhersagen machen
        y_pred = model.predict(X_val)
        
        return y_pred, model
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """
        Evaluierte die Modellperformance.
        
        Args:
            y_true: Wahre Werte
            y_pred: Vorhergesagte Werte
            
        Returns:
            tuple: (MSE, R²)
        """
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return mse, r2
    
    def get_best_degree(self) -> Tuple[int, float]:
        """
        Gibt den besten Polynomgrad und den dazugehörigen Score zurück.
        
        Returns:
            tuple: (bester_grad, bester_score)
        """
        if self.best_degree is None:
            raise ValueError("Kreuzvalidierung wurde noch nicht durchgeführt")
        
        return self.best_degree, self.best_score
    
    def plot_results(self, save_path: str = None):
        """
        Plottet die Ergebnisse der Kreuzvalidierung.
        
        Args:
            save_path (str): Pfad zum Speichern des Plots (optional)
        """
        if not self.results:
            raise ValueError("Keine Ergebnisse zum Plotten verfügbar")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MSE Plot
        ax1.errorbar(self.degrees, self.results['mean_mse'], 
                    yerr=self.results['std_mse'], fmt='o-', capsize=5, capthick=2)
        ax1.set_xlabel('Polynomgrad')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Kreuzvalidierung: MSE vs. Polynomgrad')
        ax1.grid(True, alpha=0.3)
        
        # R² Plot
        ax2.errorbar(self.degrees, self.results['mean_r2'], 
                    yerr=self.results['std_r2'], fmt='o-', capsize=5, capthick=2)
        ax2.set_xlabel('Polynomgrad')
        ax2.set_ylabel('R² Score')
        ax2.set_title('Kreuzvalidierung: R² vs. Polynomgrad')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Gibt die Ergebnisse als pandas DataFrame zurück.
        
        Returns:
            pd.DataFrame: Ergebnisse der Kreuzvalidierung
        """
        if not self.results:
            raise ValueError("Keine Ergebnisse verfügbar")
        
        df = pd.DataFrame({
            'Polynomgrad': self.degrees,
            'MSE_Mean': self.results['mean_mse'],
            'MSE_Std': self.results['std_mse'],
            'R2_Mean': self.results['mean_r2'],
            'R2_Std': self.results['std_r2']
        })
        
        return df

# Beispiel für die Verwendung
if __name__ == "__main__":
    # Beispiel-Daten generieren
    np.random.seed(42)
    n_samples = 100
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y_true = 2 * X.ravel() + 1.5 * X.ravel()**2 - 0.5 * X.ravel()**3
    y = y_true + np.random.normal(0, 2, n_samples)  # Rauschen hinzufügen
    
    # Kreuzvalidierung durchführen
    cv = PolynomialModelSelector(k_folds=5, degrees=[1, 2, 3, 4], random_state=42)
    results = cv.cross_validate(X, y)
    
    # Ergebnisse anzeigen
    print("\n" + "="*60)
    print("ERGEBNISSE DER KREUZVALIDIERUNG")
    print("="*60)
    
    df_results = cv.get_results_dataframe()
    print(df_results.round(4))
    
    # Besten Grad ausgeben
    best_degree, best_score = cv.get_best_degree()
    print(f"\nEmpfohlener Polynomgrad: {best_degree}")
    
    # Ergebnisse plotten
    cv.plot_results()