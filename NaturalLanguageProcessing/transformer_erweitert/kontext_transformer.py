#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ContextualTransformer - Erweiterter Transformer mit Padding-Unterstützung

Dieses Modell kombiniert:
- Pre-trained Word2Vec Embeddings
- Gelernte Positionskodierungen
- Mehrkopf-Selbstaufmerksamkeit mit Maskierung
- Residual Connections und Layer Normalization
- Unterstützung für variable Sequenzlängen

Autor: Ufuk Baysal
Datum: 2025

"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch import Tensor
from typing import Optional, Tuple

# Relative Imports
from .transformer_schicht import TransformerSchicht
from .mehrkopf_aufmerksamkeit import MehrkopfAufmerksamkeit


class ContextualTransformer(nn.Module):
    """
    Erweitertes Transformer-Modell für kontextuelle Wortrepräsentationen.
    
    Args:
        wort_embedder (Any): Pre-trained Word2Vec Embedding Modell
        embedding_dimension (int, optional): Dimension der Embeddings. Standard ist 64.
        anzahl_koepfe (int, optional): Anzahl der Aufmerksamkeitsköpfe. Standard ist 4.
        anzahl_schichten (int, optional): Anzahl der Transformer-Schichten. Standard ist 2.
        maximale_sequenz_laenge (int, optional): Maximale Sequenzlänge. Standard ist 100.
        dropout_rate (float, optional): Dropout-Rate für Regularisierung. Standard ist 0.1.
        use_pre_norm (bool, optional): Pre-Normalization statt Post-Normalization. Standard ist True.
    
    Beispiel:
        >>> from wort_embedding import Word2VecEmbedding
        >>> korpus = "Der Hund bellt. Die Katze schnurrt."
        >>> embedder = Word2VecEmbedding(korpus, embedding_dimension=32)
        >>> transformer = ContextualTransformer(embedder, embedding_dimension=32)
        >>> indices = embedder.erhalte_indizes_von_satz("Der Hund bellt")
        >>> ausgabe = transformer(indices.unsqueeze(0))
        >>> ausgabe.shape
        torch.Size([1, 5, 32])  # (batch_size, seq_len, embedding_dim)
    """
    
    def __init__(self,
                 wort_embedder,
                 embedding_dimension: int = 64,
                 anzahl_koepfe: int = 4,
                 anzahl_schichten: int = 2,
                 maximale_sequenz_laenge: int = 100,
                 dropout_rate: float = 0.1,
                 use_pre_norm: bool = True):
        """
        Initialisiert den Contextual Transformer.
        """
        super().__init__()
        
        # Parameter speichern
        self.embedding_dimension = embedding_dimension
        self.anzahl_koepfe = anzahl_koepfe
        self.anzahl_schichten = anzahl_schichten
        self.maximale_sequenz_laenge = maximale_sequenz_laenge
        self.dropout_rate = dropout_rate
        self.use_pre_norm = use_pre_norm
        
        # Embeddings vom Word2Vec Modell
        self.wort_embedder = wort_embedder
        self.embedding_schicht = wort_embedder.embedding_schicht
        
        # Positionskodierungen (gelernt, nicht sinusoid)
        self.positions_kodierung = nn.Parameter(
            torch.zeros(maximale_sequenz_laenge, embedding_dimension)
        )
        
        # Positionskodierungen initialisieren
        self._initialisiere_positions_kodierung()
        
        # Dropout für Embeddings
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # Transformer-Schichten
        self.transformer_schichten = nn.ModuleList([
            TransformerSchicht(
                embedding_dimension=embedding_dimension,
                anzahl_koepfe=anzahl_koepfe,
                dropout_rate=dropout_rate,
                use_pre_norm=use_pre_norm
            )
            for _ in range(anzahl_schichten)
        ])
        
        # Finale Layer Normalization
        self.final_norm = nn.LayerNorm(embedding_dimension)
        
        # Ausgabe-Projektion (optional)
        self.ausgabe_projektion = nn.Linear(embedding_dimension, embedding_dimension)
        
        print(f"✅ ContextualTransformer initialisiert:")
        print(f"   - Embedding Dimension: {embedding_dimension}")
        print(f"   - Aufmerksamkeitsköpfe: {anzahl_koepfe}")
        print(f"   - Transformer-Schichten: {anzahl_schichten}")
        print(f"   - Maximale Sequenzlänge: {maximale_sequenz_laenge}")
        print(f"   - Dropout Rate: {dropout_rate}")
        print(f"   - Pre-Norm: {use_pre_norm}")
    
    def _initialisiere_positions_kodierung(self) -> None:
        """
        Initialisiert die Positionskodierungen.
        """
        # Xavier Initialisierung für gelernte Positionskodierungen
        nn.init.xavier_uniform_(self.positions_kodierung)
        
        # Alternativ: Sinusoidale Initialisierung (wie im originalen Transformer)
        # positions = torch.arange(self.maximale_sequenz_laenge).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.embedding_dimension, 2) * 
        #                     -(math.log(10000.0) / self.embedding_dimension))
        # self.positions_kodierung.data[:, 0::2] = torch.sin(positions * div_term)
        # self.positions_kodierung.data[:, 1::2] = torch.cos(positions * div_term)
    
    def forward(self, 
                indices: Tensor,
                maskierung: Optional[Tensor] = None) -> Tensor:
        """
        Führt den Forward Pass durch den Transformer.
        
        Args:
            indices (Tensor): Tensor von Wortindizes.
                            Form: (batch_size, seq_len) oder (seq_len,)
            maskierung (Optional[Tensor]): Padding-Maskierung.
                                         Form: (batch_size, seq_len)
                                         Werte: 1 für echte Tokens, 0 für Padding
            
        Returns:
            Tensor: Kodierte Sequenz.
                   Form: (batch_size, seq_len, embedding_dimension) oder
                         (seq_len, embedding_dimension)
        
        Beispiel:
            >>> transformer = ContextualTransformer(...)
            >>> indices = torch.tensor([[1, 2, 3, 0, 0]])  # Mit Padding
            >>> maskierung = torch.tensor([[1, 1, 1, 0, 0]])  # Maskierung für Padding
            >>> ausgabe = transformer(indices, maskierung)
            >>> ausgabe.shape
            torch.Size([1, 5, 64])
        """
        # Input-Form normalisieren
        hat_batch_dim = len(indices.shape) == 2
        
        if not hat_batch_dim:
            # Batch-Dimension hinzufügen
            indices = indices.unsqueeze(0)
            if maskierung is not None:
                maskierung = maskierung.unsqueeze(0)
        
        batch_size, seq_len = indices.shape
        
        # Embeddings erhalten
        wort_embeddings = self.embedding_schicht(indices)
        
        # Positionskodierungen hinzufügen
        if seq_len <= self.maximale_sequenz_laenge:
            positions_embeddings = self.positions_kodierung[:seq_len]
        else:
            # Falls Sequenz länger als maximale Länge, abschneiden
            positions_embeddings = self.positions_kodierung
            wort_embeddings = wort_embeddings[:, :self.maximale_sequenz_laenge, :]
            seq_len = self.maximale_sequenz_laenge
        
        # Wort- und Positions-Embeddings kombinieren
        x = wort_embeddings + positions_embeddings.unsqueeze(0)
        
        # Dropout anwenden
        x = self.embedding_dropout(x)
        
        # Transformer-Schichten anwenden
        for schicht in self.transformer_schichten:
            x = schicht(x, maskierung)
        
        # Finale Normalisierung
        x = self.final_norm(x)
        
        # Ausgabe-Projektion (optional)
        x = self.ausgabe_projektion(x)
        
        # Batch-Dimension entfernen falls nötig
        if not hat_batch_dim:
            x = x.squeeze(0)
        
        return x
    
    def _erstelle_padding_maskierung(self, indices: Tensor) -> Tensor:
        """
        Erstellt eine Padding-Maskierung aus den Wortindizes.
        
        Args:
            indices (Tensor): Tensor von Wortindizes.
                            Form: (batch_size, seq_len)
        
        Returns:
            Tensor: Padding-Maskierung.
                   Form: (batch_size, seq_len)
                   Werte: 1 für echte Tokens, 0 für Padding
        """
        # Annahme: 0 ist der Padding-Index
        maskierung = (indices != 0).float()
        return maskierung
    
    def berechne_aufmerksamkeits_gewichte(self,
                                        indices: Tensor,
                                        schicht_index: int = 0) -> Tensor:
        """
        Berechnet die Aufmerksamkeitsgewichte für eine gegebene Eingabe.
        
        Args:
            indices (Tensor): Eingabeindizes
            schicht_index (int): Index der Transformer-Schicht
            
        Returns:
            Tensor: Aufmerksamkeitsgewichte
        """
        if schicht_index >= len(self.transformer_schichten):
            raise ValueError(f"Schichtindex {schicht_index} außerhalb des Bereichs")
        
        # Zur Evaluations-Modus wechseln
        self.eval()
        
        with torch.no_grad():
            # Eingabe vorbereiten
            hat_batch_dim = len(indices.shape) == 2
            
            if not hat_batch_dim:
                indices = indices.unsqueeze(0)
            
            batch_size, seq_len = indices.shape
            
            # Embeddings erhalten
            wort_embeddings = self.embedding_schicht(indices)
            
            # Positionskodierungen hinzufügen
            if seq_len <= self.maximale_sequenz_laenge:
                positions_embeddings = self.positions_kodierung[:seq_len]
            else:
                positions_embeddings = self.positions_kodierung
                wort_embeddings = wort_embeddings[:, :self.maximale_sequenz_laenge, :]
                seq_len = self.maximale_sequenz_laenge
            
            x = wort_embeddings + positions_embeddings.unsqueeze(0)
            
            # Padding-Maskierung erstellen
            maskierung = self._erstelle_padding_maskierung(indices)
            
            # Bis zur gewünschten Schicht propagieren
            for i, schicht in enumerate(self.transformer_schichten):
                if i == schicht_index:
                    # Aufmerksamkeitsgewichte von dieser Schicht erhalten
                    aufmerksamkeits_gewichte = schicht.berechne_aufmerksamkeits_gewichte(
                        x, maskierung
                    )
                    break
                x = schicht(x, maskierung)
            else:
                raise ValueError("Schicht nicht gefunden")
        
        # Zurück zum Trainings-Modus
        self.train()
        
        return aufmerksamkeits_gewichte
    
    def berechne_kontextuelle_aehnlichkeit(self,
                                         indices1: Tensor,
                                         indices2: Tensor) -> float:
        """
        Berechnet die kontextuelle Ähnlichkeit zwischen zwei Sequenzen.
        
        Args:
            indices1 (Tensor): Indizes der ersten Sequenz
            indices2 (Tensor): Indizes der zweiten Sequenz
            
        Returns:
            float: Kosinus-Ähnlichkeit der gemittelten Repräsentationen
        """
        self.eval()
        
        with torch.no_grad():
            # Repräsentationen berechnen
            repr1 = self(indices1.unsqueeze(0) if len(indices1.shape) == 1 else indices1)
            repr2 = self(indices2.unsqueeze(0) if len(indices2.shape) == 1 else indices2)
            
            # Padding-Maskierungen
            mask1 = self._erstelle_padding_maskierung(
                indices1.unsqueeze(0) if len(indices1.shape) == 1 else indices1
            )
            mask2 = self._erstelle_padding_maskierung(
                indices2.unsqueeze(0) if len(indices2.shape) == 1 else indices2
            )
            
            # Gewichtete Durchschnitte berechnen (Padding ignorieren)
            # repr1: (batch_size, seq_len, embedding_dim)
            # mask1: (batch_size, seq_len)
            
            # Maske für die Berechnung erweitern
            mask1_expanded = mask1.unsqueeze(-1)  # (batch_size, seq_len, 1)
            mask2_expanded = mask2.unsqueeze(-1)
            
            # Gewichtete Summe
            sum1 = torch.sum(repr1 * mask1_expanded, dim=1)  # (batch_size, embedding_dim)
            sum2 = torch.sum(repr2 * mask2_expanded, dim=1)
            
            # Summe der Gewichte
            count1 = torch.sum(mask1, dim=1, keepdim=True)  # (batch_size, 1)
            count2 = torch.sum(mask2, dim=1, keepdim=True)
            
            # Durchschnitte
            avg1 = sum1 / count1.clamp(min=1)
            avg2 = sum2 / count2.clamp(min=1)
            
            # Kosinus-Ähnlichkeit berechnen
            cos_sim = F.cosine_similarity(avg1, avg2, dim=1)
            
            # Skalar zurückgeben
            similarity = cos_sim.mean().item()
        
        self.train()
        
        return similarity
    
    def extrahiere_features(self, 
                          indices: Tensor,
                          ebene: str = "last") -> Tensor:
        """
        Extrahiert Features von einer bestimmten Ebene des Transformers.
        
        Args:
            indices (Tensor): Eingabeindizes
            ebene (str): Feature-Ebene ('embedding', 'middle', 'last')
            
        Returns:
            Tensor: Extrahiertes Features
        """
        self.eval()
        
        with torch.no_grad():
            # Eingabe vorbereiten
            hat_batch_dim = len(indices.shape) == 2
            
            if not hat_batch_dim:
                indices = indices.unsqueeze(0)
            
            batch_size, seq_len = indices.shape
            
            # Embeddings erhalten
            wort_embeddings = self.embedding_schicht(indices)
            
            # Positionskodierungen hinzufügen
            if seq_len <= self.maximale_sequenz_laenge:
                positions_embeddings = self.positions_kodierung[:seq_len]
            else:
                positions_embeddings = self.positions_kodierung
                wort_embeddings = wort_embeddings[:, :self.maximale_sequenz_laenge, :]
                seq_len = self.maximale_sequenz_laenge
            
            x = wort_embeddings + positions_embeddings.unsqueeze(0)
            
            # Padding-Maskierung
            maskierung = self._erstelle_padding_maskierung(indices)
            
            if ebene == "embedding":
                features = x
            elif ebene == "middle":
                # Bis zur mittleren Schicht propagieren
                middle_index = len(self.transformer_schichten) // 2
                for i, schicht in enumerate(self.transformer_schichten):
                    if i == middle_index:
                        break
                    x = schicht(x, maskierung)
                features = x
            elif ebene == "last":
                # Durch alle Schichten propagieren
                for schicht in self.transformer_schichten:
                    x = schicht(x, maskierung)
                features = x
            else:
                raise ValueError(f"Unbekannte Ebene: {ebene}")
            
            # Batch-Dimension entfernen falls nötig
            if not hat_batch_dim:
                features = features.squeeze(0)
            
            return features
        
        self.train()