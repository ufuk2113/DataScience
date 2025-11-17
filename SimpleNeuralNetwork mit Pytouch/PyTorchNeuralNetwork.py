# pytorch_nn.py
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import SimpleNeuralNetwork

class PyTorchNeuralNetwork(SimpleNeuralNetwork, nn.Module):
    """
    PyTorch-Implementierung, die von deiner numpy-Klasse erbt (SimpleNeuralNetwork).
    Unterstützt verschiedene Aktivierungsfunktionen, GPU, Adam und SGD(momentum).
    """

    def __init__(self,
                 input_size=784,
                 hidden_size=100,
                 output_size=10,
                 activation="sigmoid",
                 activation_param=None,
                 learning_rate=0.01,
                 optimizer_name="adam"):
        # init base classes
        SimpleNeuralNetwork.__init__(self, input_size, hidden_size, output_size, learning_rate)
        nn.Module.__init__(self)

        # PyTorch layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        # Activation selection
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "leaky_relu":
            slope = activation_param if activation_param is not None else 0.01
            self.activation = nn.LeakyReLU(negative_slope=slope)
        elif activation == "prelu":
            # PReLU has a learnable parameter
            self.activation = nn.PReLU()
        elif activation == "elu":
            alpha = activation_param if activation_param is not None else 1.0
            self.activation = nn.ELU(alpha=alpha)
        else:
            raise ValueError(f"Aktivierung '{activation}' nicht unterstützt")

        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name.lower()

        # Loss (classification)
        self.loss_fn = nn.CrossEntropyLoss()

        # device placeholder (set in main)
        self.device = torch.device("cpu")

        # optimizer will be created later when model parameters are on the correct device
        self.optimizer = None

    def to_device(self, device):
        """Move model to device and create optimizer (must be called from main)."""
        self.device = device
        self.to(device)
        # create optimizer after model.to(device)
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            # default: momentum 0.9
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError("optimizer_name must be 'adam' or 'sgd'")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def train_model(self, train_loader, epochs=10, log_every=100):
        self.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader, start=1):
                images = images.view(-1, self.input_size).to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.forward(images)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if log_every and batch_idx % log_every == 0:
                    avg_loss = running_loss / log_every
                    print(f"Epoch [{epoch}/{epochs}]  Batch [{batch_idx}]  Loss: {avg_loss:.4f}")
                    running_loss = 0.0

    def evaluate(self, test_loader):
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, self.input_size).to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(images)
                _, preds = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100.0 * correct / total
