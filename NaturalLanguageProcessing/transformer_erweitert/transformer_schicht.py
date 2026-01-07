#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TransformerSchicht - Eine einzelne Transformer-Schicht

Implementiert eine vollständige Transformer-Schicht mit:
- Mehrkopf-Selbstaufmerksamkeit
- Feed-Forward Netzwerk
- Residual Connections
- Layer Normalization
- Dropout für Regularisierung

Autor: Ufuk Baysal
Datum: 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .mehrkopf_aufmerksamkeit import MehrkopfAufmerksamkeit


class TransformerSchicht(nn.Module):
    """
    Eine einzelne Transformer-Schicht.
    
    Args:
        embedding_dimension (int): Dimension der Eingabe-Embeddings
        anzahl_koepfe (int, optional): Anzahl der Aufmerksamkeitsköpfe. Standard ist 4.
        dropout_rate (float, optional): Dropout-Rate. Standard ist 0.1.
        feedforward_dimension (int, optional): Dimension des Feed-Forward Netzwerks. 
                                              Standard ist 4 * embedding_dimension.
        use_pre_norm (bool, optional): Pre-Normalization statt Post-Normalization. Standard ist True.
        activation (str, optional): Aktivierungsfunktion ('relu', 'gelu'). Standard ist 'gelu'.
    
    Beispiel:
        >>> schicht = TransformerSchicht(embedding_dimension=64, anzahl_koepfe=4)
        >>> x = torch.randn(2, 10, 64)  # (batch_size, seq_len, embedding_dim)
        >>> ausgabe = schicht(x)
        >>> ausgabe.shape
        torch.Size([2, 10, 64])
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 anzahl_koepfe: int = 4,
                 dropout_rate: float = 0.1,
                 feedforward_dimension: Optional[int] = None,
                 use_pre_norm: bool = True,
                 activation: str = 'gelu'):
        """
        Initialisiert die Transformer-Schicht.
        """
        super().__init__()
        
        self.embedding_dimension = embedding_dimension
        self.anzahl_koepfe = anzahl_koepfe
        self.dropout_rate = dropout_rate
        self.use_pre_norm = use_pre_norm
        self.activation = activation
        
        # Feed-Forward Dimension setzen
        if feedforward_dimension is None:
            self.feedforward_dimension = 4 * embedding_dimension
        else:
            self.feedforward_dimension = feedforward_dimension
        
        # Selbstaufmerksamkeit
        self.selbst_aufmerksamkeit = MehrkopfAufmerksamkeit(
            embedding_dimension=embedding_dimension,
            anzahl_koepfe=anzahl_koepfe,
            dropout=dropout_rate
        )
        
        # Layer Normalizations
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        
        # Feed-Forward Netzwerk
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dimension, self.feedforward_dimension),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feedforward_dimension, embedding_dimension)
        )
        
        # Dropout für Residual Connections
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Parameter initialisieren
        self._initialisiere_parameter()
    
    def _get_activation(self, activation_name: str) -> nn.Module:
        """
        Gibt die Aktivierungsfunktion zurück.
        
        Args:
            activation_name: Name der Aktivierungsfunktion
            
        Returns:
            PyTorch Aktivierungsmodul
        """
        if activation_name.lower() == 'relu':
            return nn.ReLU()
        elif activation_name.lower() == 'gelu':
            return nn.GELU()
        elif activation_name.lower() == 'silu':
            return nn.SiLU()
        else:
            print(f"⚠️  Unbekannte Aktivierung: {activation_name}. Verwende GELU.")
            return nn.GELU()
    
    def _initialisiere_parameter(self) -> None:
        """
        Initialisiert die Parameter des Modells.
        """
        # Xavier Initialisierung für lineare Schichten
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                if 'feed_forward' in name:
                    # Feed-Forward Schichten
                    if param.shape[0] == self.feedforward_dimension:
                        # Erste Schicht: Eingabe -> versteckte Dimension
                        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                    else:
                        # Zweite Schicht: versteckte Dimension -> Ausgabe
                        nn.init.xavier_uniform_(param)
                elif 'selbst_aufmerksamkeit' in name:
                    # Aufmerksamkeits-Schichten
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Bias-Terme auf 0 initialisieren
                nn.init.zeros_(param)
    
    def forward(self, 
                x: Tensor,
                maskierung: Optional[Tensor] = None) -> Tensor:
        """
        Führt den Forward Pass durch die Transformer-Schicht.
        
        Args:
            x (Tensor): Eingabetensor der Form (batch_size, seq_len, embedding_dim)
            maskierung (Optional[Tensor]): Padding-Maskierung der Form (batch_size, seq_len)
            
        Returns:
            Tensor: Ausgabetensor der Form (batch_size, seq_len, embedding_dim)
        """
        residual = x
        
        if self.use_pre_norm:
            # Pre-Normalization (neuerer Ansatz)
            # 1. Layer Normalization
            x_norm = self.norm1(x)
            
            # 2. Selbstaufmerksamkeit
            attn_output = self.selbst_aufmerksamkeit(x_norm, x_norm, x_norm, maskierung)
            attn_output = self.dropout1(attn_output)
            
            # 3. Residual Connection
            x = residual + attn_output
            
            # 4. Layer Normalization für Feed-Forward
            residual = x
            x_norm = self.norm2(x)
            
            # 5. Feed-Forward Netzwerk
            ff_output = self.feed_forward(x_norm)
            ff_output = self.dropout2(ff_output)
            
            # 6. Residual Connection
            x = residual + ff_output
            
        else:
            # Post-Normalization (originaler Transformer)
            # 1. Selbstaufmerksamkeit
            attn_output = self.selbst_aufmerksamkeit(x, x, x, maskierung)
            attn_output = self.dropout1(attn_output)
            
            # 2. Residual Connection + Layer Normalization
            x = self.norm1(x + attn_output)
            
            # 3. Feed-Forward Netzwerk
            ff_output = self.feed_forward(x)
            ff_output = self.dropout2(ff_output)
            
            # 4. Residual Connection + Layer Normalization
            x = self.norm2(x + ff_output)
        
        return x
    
    def berechne_aufmerksamkeits_gewichte(self,
                                        x: Tensor,
                                        maskierung: Optional[Tensor] = None) -> Tensor:
        """
        Berechnet die Aufmerksamkeitsgewichte für die gegebene Eingabe.
        
        Args:
            x (Tensor): Eingabetensor
            maskierung (Optional[Tensor]): Padding-Maskierung
            
        Returns:
            Tensor: Aufmerksamkeitsgewichte
        """
        # Zur Evaluations-Modus wechseln
        training_mode = self.training
        self.eval()
        
        with torch.no_grad():
            # Aufmerksamkeitsgewichte berechnen
            _, aufmerksamkeits_gewichte = self.selbst_aufmerksamkeit.berechne_aufmerksamkeits_karten(
                x, x, maskierung
            )
        
        # Zurück zum ursprünglichen Modus
        self.train(training_mode)
        
        return aufmerksamkeits_gewichte