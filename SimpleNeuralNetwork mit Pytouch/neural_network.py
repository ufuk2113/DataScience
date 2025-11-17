# neural_network.py
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size=784, hidden_size=100, output_size=10, learning_rate=0.3,
                 weights_input_hidden=None, weights_hidden_output=None):
        """
        Initialisiert das Neural Network.
        Alle Parameter können beim Erstellen übergeben werden, ansonsten werden Standardwerte verwendet.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Gewichte zufällig initialisieren, falls keine übergeben wurden
        if weights_input_hidden is None:
            self.weights_input_hidden = np.random.rand(self.hidden_size, self.input_size) - 0.5
        else:
            self.weights_input_hidden = weights_input_hidden

        if weights_hidden_output is None:
            self.weights_hidden_output = np.random.rand(self.output_size, self.hidden_size) - 0.5
        else:
            self.weights_hidden_output = weights_hidden_output

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def test(self, input_vector):
        """
        Forward Pass: Berechnet den Output des Netzwerks für einen Input-Vektor.
        input_vector: np.array der Größe (input_size,)
        Rückgabe: np.array der Größe (output_size,)
        """
        # Input -> Hidden
        hidden_input = np.dot(self.weights_input_hidden, input_vector)
        hidden_output = self.sigmoid(hidden_input)

        # Hidden -> Output
        final_input = np.dot(self.weights_hidden_output, hidden_output)
        final_output = self.sigmoid(final_input)

        return final_output

    
    def get_weights(self):
        """
        Gibt die aktuellen Gewichte zurück
        """
        return self.weights_input_hidden, self.weights_hidden_output
    
    def set_weights(self, weights_input_hidden, weights_hidden_output):
        """
        Setzt die Gewichtsmatrizen
        """
        self.weights_input_hidden = weights_input_hidden
        self.weights_hidden_output = weights_hidden_output
    
    def get_network_info(self):
        """
        Gibt Informationen über das Network zurück
        """
        info = f"""
        Neural Network Information:
        - Input Layer: {self.input_size} Neuronen
        - Hidden Layer: {self.hidden_size} Neuronen  
        - Output Layer: {self.output_size} Neuronen
        - Learning Rate: {self.learning_rate}
        - Gewichtsmatrix Input-Hidden: {self.weights_input_hidden.shape}
        - Gewichtsmatrix Hidden-Output: {self.weights_hidden_output.shape}
        """
        return info