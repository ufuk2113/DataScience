# main.py
import numpy as np
from neural_network import SimpleNeuralNetwork

def main():
    print("Neural Network für MNIST Klassifikation")
    print("=" * 50)
    
    # 1. Standard Neural Network
    print("\n1. Standard Neural Network:")
    nn_standard = SimpleNeuralNetwork()
    print(nn_standard.get_network_info())
    
    # 2. Neural Network mit benutzerdefinierten Gewichten
    print("\n2. Neural Network mit benutzerdefinierten Gewichten:")
    
    # Eigene Gewichtsmatrizen erstellen
    custom_weights_input_hidden = np.random.rand(100, 784) * 0.1 - 0.05  # Kleinerer Bereich
    custom_weights_hidden_output = np.random.rand(10, 100) * 0.1 - 0.05
    
    nn_custom_weights = SimpleNeuralNetwork(
        weights_input_hidden=custom_weights_input_hidden,
        weights_hidden_output=custom_weights_hidden_output
    )
    print(nn_custom_weights.get_network_info())
    
    # 3. Test mit simulierten Daten
    print("\n3. Test mit simulierten MNIST-Daten:")
    
    # Simulieren eines MNIST-Datensatzes (784 Pixel, Werte zwischen 0-1)
    simulated_mnist_data = np.random.rand(784)
    
    # Test mit beiden Networks
    output_standard = nn_standard.test(simulated_mnist_data)
    output_custom = nn_custom_weights.test(simulated_mnist_data)
    
    print("Output Standard Network:")
    print(f"  Shape: {output_standard.shape}")
    print(f"  Wertebereich: {output_standard.min():.4f} - {output_standard.max():.4f}")
    print(f"  Vorhersage (höchster Wert): {np.argmax(output_standard)}")
    
    print("\nOutput Custom Weights Network:")
    print(f"  Shape: {output_custom.shape}") 
    print(f"  Wertebereich: {output_custom.min():.4f} - {output_custom.max():.4f}")
    print(f"  Vorhersage (höchster Wert): {np.argmax(output_custom)}")
    
    # 4. Ausgabe der kompletten Output-Vektoren
    print("\n4. Detaillierte Output-Vektoren:")
    print("\nStandard Network Output:")
    for i, value in enumerate(output_standard):
        print(f"  Neuron {i}: {value:.6f}")
    
    print(f"\nVorhergesagte Klasse: {np.argmax(output_standard)}")

if __name__ == "__main__":
    main()