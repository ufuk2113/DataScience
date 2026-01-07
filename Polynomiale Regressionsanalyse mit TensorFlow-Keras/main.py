"""
Teilprüfung 1 - TensorFlow Aufgabe
Polynomial Regression mit TensorFlow / Keras

Erstelle ein Skript, das eine polynomiale Regressionsanalyse durchführt. Das Skript soll die folgenden Schritte umfassen:

    Generiere einen nicht-lineare Datensatz.
    Teile die Daten in Trainings- und Testdatensätze auf.
    Standardisiere die Merkmale.
    Definiere und instanziiere ein polynomiales Regressionsmodell mit TensorFlow.
    Kompiliere das Modell.
    Trainiere und evaluiere das Modell.
    Zeige die Trainings- und Testverlustwerte an.
    
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- Einstellbare Parameter ---
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

N_SAMPLES = 300
TEST_RATIO = 0.2
POLY_DEGREE = 5            # Grad des Polynoms (z.B. 3, 5, ...)
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.01

# --- 1) Generiere nichtlinearen Datensatz ---
# Beispielfunktion: y = 0.5*x^3 - x^2 + 2*x + noise
x = np.random.uniform(-3, 3, size=(N_SAMPLES, 1))
# echte (noisefreie) Zielfunktion
def true_function(x):
    return 0.5 * x**3 - 1.0 * x**2 + 2.0 * x

y_true = true_function(x)
noise = np.random.normal(scale=6.0, size=y_true.shape)  # anpassen für mehr/ weniger Rauschen
y = y_true + noise

# --- 2) Trainings- / Testaufteilung ---
idx = np.arange(N_SAMPLES)
np.random.shuffle(idx)
n_test = int(N_SAMPLES * TEST_RATIO)
test_idx = idx[:n_test]
train_idx = idx[n_test:]

x_train = x[train_idx]
y_train = y[train_idx]
x_test = x[test_idx]
y_test = y[test_idx]

# --- 3) Polynom-Feature-Expansion (ohne sklearn) ---
def polynomial_features(x, degree):
    """
    x: numpy array shape (n_samples, 1) or (n_samples, n_features)
    returns: array shape (n_samples, n_features*degree) with powers 1..degree
    Für Einfachheit hier für eine einzige Eingangsvariable (x.shape[1] == 1) implementiert.
    """
    n_samples = x.shape[0]
    # Wenn mehrere Eingangfeatures: erweitern für jedes (nicht nötig für diese Aufgabe)
    X_poly = []
    base = x if x.ndim == 2 else x.reshape(-1, 1)
    for d in range(1, degree+1):
        X_poly.append(base ** d)  # shape (n_samples, 1)
    X_poly = np.hstack(X_poly)  # shape (n_samples, degree)
    return X_poly

X_train_poly = polynomial_features(x_train, POLY_DEGREE)
X_test_poly = polynomial_features(x_test, POLY_DEGREE)

# --- 4) Standardisiere Merkmale (wichtig für Training) ---
# Fit nur auf Trainingsdaten
feature_mean = X_train_poly.mean(axis=0)
feature_std  = X_train_poly.std(axis=0)
# Vermeide Division durch 0
feature_std[feature_std == 0.0] = 1.0

X_train_scaled = (X_train_poly - feature_mean) / feature_std
X_test_scaled  = (X_test_poly  - feature_mean) / feature_std

# --- 5) Definiere und instanziiere das polynomiale Regressionsmodell ---
# Polynomiale Regression = lineares Modell auf polynomierten Features.
input_dim = X_train_scaled.shape[1]

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    # lineare Regression -> Dense(1) ohne Aktivierung
    tf.keras.layers.Dense(1, activation=None)
])

# --- 6) Kompiliere das Modell ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='mse',
    metrics=['mse']
)

model.summary()

# --- 7) Trainiere das Modell ---
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0  # auf 1 setzen für pro-epoch Ausgabe
)

# --- 8) Evaluiere das Modell und zeige Verluste an ---
train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
print(f"Endgültiger Trainingsverlust (MSE): {train_loss:.4f}")
print(f"Endgültiger Testverlust (MSE):      {val_loss:.4f}")

# Besser: Ausgabe der Verlaufswerte (erste/letzte 5 Werte)
print("\nBeispiel: Verlustverlauf (erste 5 Epochen):")
for i in range(min(5, len(history.history['loss']))):
    print(f" Epoche {i+1:3d} - train_loss: {history.history['loss'][i]:.4f} - val_loss: {history.history['val_loss'][i]:.4f}")
print("...")
print(f" Epoche {len(history.history['loss']):3d} - train_loss: {history.history['loss'][-1]:.4f} - val_loss: {history.history['val_loss'][-1]:.4f}")

# --- 9) Visualisiere: Verlustkurven und Modellvorhersage ---
# Verlustkurven
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Train / Test Lossverlauf')
plt.legend()

# Modellvorhersage: zeichne Kurve über dichtes Grid
x_grid = np.linspace(-3.2, 3.2, 400).reshape(-1,1)
X_grid_poly = polynomial_features(x_grid, POLY_DEGREE)
X_grid_scaled = (X_grid_poly - feature_mean) / feature_std
y_pred_grid = model.predict(X_grid_scaled).flatten()

plt.subplot(1,2,2)
# Plot echte Punktdaten
plt.scatter(x_train, y_train, alpha=0.5, label='Train Daten', s=20)
plt.scatter(x_test, y_test, alpha=0.8, label='Test Daten', s=40, marker='x')
# Plot wahre Funktionskurve (ohne Noise)
x_dense = np.linspace(-3, 3, 200).reshape(-1,1)
plt.plot(x_dense, true_function(x_dense), label='Wahre Funktion (noisefree)', linewidth=2)
# Plot Modellvorhersage
plt.plot(x_grid, y_pred_grid, label='Modellvorhersage', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Polynomiale Regression (Grad {POLY_DEGREE})')
plt.legend()
plt.tight_layout()
plt.show()

# --- Optional: Parameter (Koeffizienten) anzeigen ---
weights, biases = model.layers[0].get_weights()
# weights shape: (input_dim, 1)
coefs = weights.flatten()
print("\nGelerntes lineares Modell auf polynomierten Features:")
for d, c in enumerate(coefs, start=1):
    print(f"  x^{d} * {c:.4f}")
print(f"Bias (Intercept): {biases[0]:.4f}")
