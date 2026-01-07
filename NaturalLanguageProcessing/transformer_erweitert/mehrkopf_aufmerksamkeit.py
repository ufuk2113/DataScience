#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MehrkopfAufmerksamkeit - Implementierung von Multi-Head Attention

Diese Klasse implementiert Mehrkopf-Aufmerksamkeit mit:
- Separaten linearen Projektionen für Query, Key, Value
- Skalierte Punktprodukt-Aufmerksamkeit
- Maskierungsunterstützung für Padding
- Optionale Aufmerksamkeitsgewichts-Rückgabe

Autor: Ufuk Baysal
Datum: 2025

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from typing import Optional, Tuple


class MehrkopfAufmerksamkeit(nn.Module):
    """
    Implementierung von Multi-Head Self-Attention.
    
    Args:
        embedding_dimension (int): Dimension der Eingabe-Embeddings
        anzahl_koepfe (int, optional): Anzahl der Aufmerksamkeitsköpfe. Standard ist 4.
        dropout (float, optional): Dropout-Rate für Aufmerksamkeitsgewichte. Standard ist 0.1.
        bias (bool, optional): Bias in linearen Projektionen verwenden. Standard ist True.
    
    Beispiel:
        >>> attn = MehrkopfAufmerksamkeit(embedding_dimension=64, anzahl_koepfe=4)
        >>> x = torch.randn(2, 10, 64)  # (batch_size, seq_len, embedding_dim)
        >>> ausgabe = attn(x, x, x)
        >>> ausgabe.shape
        torch.Size([2, 10, 64])
    """
    
    def __init__(self,
                 embedding_dimension: int,
                 anzahl_koepfe: int = 4,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        Initialisiert die Mehrkopf-Aufmerksamkeit.
        """
        super().__init__()
        
        # Parameter validieren
        assert embedding_dimension % anzahl_koepfe == 0, \
            f"Embedding Dimension ({embedding_dimension}) muss durch Anzahl Köpfe ({anzahl_koepfe}) teilbar sein"
        
        self.embedding_dimension = embedding_dimension
        self.anzahl_koepfe = anzahl_koepfe
        self.dropout_rate = dropout
        self.bias = bias
        
        # Dimension pro Kopf
        self.dimension_pro_kopf = embedding_dimension // anzahl_koepfe
        
        # Lineare Projektionen für Query, Key, Value
        self.query_projektion = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.key_projektion = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        self.value_projektion = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        
        # Ausgabe-Projektion
        self.ausgabe_projektion = nn.Linear(embedding_dimension, embedding_dimension, bias=bias)
        
        # Dropout für Aufmerksamkeitsgewichte
        self.aufmerksamkeit_dropout = nn.Dropout(dropout)
        
        # Skalierungsfaktor für stabile Softmax
        self.skalierung = math.sqrt(self.dimension_pro_kopf)
        
        # Parameter initialisieren
        self._initialisiere_parameter()
        
        print(f"   MehrkopfAufmerksamkeit: {anzahl_koepfe} Köpfe, "
              f"{self.dimension_pro_kopf} Dimensionen pro Kopf")
    
    def _initialisiere_parameter(self) -> None:
        """
        Initialisiert die Parameter der linearen Schichten.
        """
        # Xavier/Glorot Initialisierung
        nn.init.xavier_uniform_(self.query_projektion.weight)
        nn.init.xavier_uniform_(self.key_projektion.weight)
        nn.init.xavier_uniform_(self.value_projektion.weight)
        nn.init.xavier_uniform_(self.ausgabe_projektion.weight)
        
        if self.bias:
            nn.init.zeros_(self.query_projektion.bias)
            nn.init.zeros_(self.key_projektion.bias)
            nn.init.zeros_(self.value_projektion.bias)
            nn.init.zeros_(self.ausgabe_projektion.bias)
    
    def forward(self,
                query: Tensor,
                schluessel: Tensor,
                wert: Tensor,
                maskierung: Optional[Tensor] = None) -> Tensor:
        """
        Führt den Forward Pass der Mehrkopf-Aufmerksamkeit durch.
        
        Args:
            query (Tensor): Query Tensor der Form (batch_size, query_len, embedding_dim)
            schluessel (Tensor): Key Tensor der Form (batch_size, key_len, embedding_dim)
            wert (Tensor): Value Tensor der Form (batch_size, value_len, embedding_dim)
            maskierung (Optional[Tensor]): Maskierungstensor der Form (batch_size, query_len, key_len)
                                         oder (batch_size, key_len)
            
        Returns:
            Tensor: Aufmerksamkeitsausgabe der Form (batch_size, query_len, embedding_dim)
        """
        batch_size = query.size(0)
        
        # 1. Lineare Projektion und Aufteilung in Köpfe
        query = self.query_projektion(query)
        schluessel = self.key_projektion(schluessel)
        wert = self.value_projektion(wert)
        
        # 2. Reshape für Mehrkopf-Aufmerksamkeit
        # Form: (batch_size, seq_len, anzahl_koepfe, dimension_pro_kopf)
        query = query.view(batch_size, -1, self.anzahl_koepfe, self.dimension_pro_kopf)
        schluessel = schluessel.view(batch_size, -1, self.anzahl_koepfe, self.dimension_pro_kopf)
        wert = wert.view(batch_size, -1, self.anzahl_koepfe, self.dimension_pro_kopf)
        
        # 3. Transponieren für Aufmerksamkeitsberechnung
        # Form: (batch_size, anzahl_koepfe, seq_len, dimension_pro_kopf)
        query = query.transpose(1, 2)
        schluessel = schluessel.transpose(1, 2)
        wert = wert.transpose(1, 2)
        
        # 4. Skalierte Punktprodukt-Aufmerksamkeit
        aufmerksamkeits_ausgabe, aufmerksamkeits_gewichte = self._skalierte_aufmerksamkeit(
            query, schluessel, wert, maskierung
        )
        
        # 5. Köpfe kombinieren
        # Transponieren zurück: (batch_size, seq_len, anzahl_koepfe, dimension_pro_kopf)
        aufmerksamkeits_ausgabe = aufmerksamkeits_ausgabe.transpose(1, 2).contiguous()
        
        # Reshape: (batch_size, seq_len, embedding_dim)
        aufmerksamkeits_ausgabe = aufmerksamkeits_ausgabe.view(
            batch_size, -1, self.embedding_dimension
        )
        
        # 6. Ausgabe-Projektion
        ausgabe = self.ausgabe_projektion(aufmerksamkeits_ausgabe)
        
        return ausgabe
    
    def _skalierte_aufmerksamkeit(self,
                                 query: Tensor,
                                 schluessel: Tensor,
                                 wert: Tensor,
                                 maskierung: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Berechnet skaliertes Punktprodukt-Aufmerksamkeit.
        
        Args:
            query: Query Tensor der Form (batch_size, anzahl_koepfe, query_len, dimension_pro_kopf)
            schluessel: Key Tensor der Form (batch_size, anzahl_koepfe, key_len, dimension_pro_kopf)
            wert: Value Tensor der Form (batch_size, anzahl_koepfe, value_len, dimension_pro_kopf)
            maskierung: Optionaler Maskierungstensor
            
        Returns:
            Tuple[Tensor, Tensor]: (Aufmerksamkeitsausgabe, Aufmerksamkeitsgewichte)
        """
        # Punktprodukt zwischen Query und Key
        # query: (batch_size, anzahl_koepfe, query_len, dimension_pro_kopf)
        # schluessel: (batch_size, anzahl_koepfe, key_len, dimension_pro_kopf)
        # -> scores: (batch_size, anzahl_koepfe, query_len, key_len)
        scores = torch.matmul(query, schluessel.transpose(-2, -1)) / self.skalierung
        
        # Maskierung anwenden falls vorhanden
        if maskierung is not None:
            # Maskierung erweitern für Mehrkopf-Dimension
            if len(maskierung.shape) == 2:
                # (batch_size, key_len) -> (batch_size, 1, 1, key_len)
                maskierung = maskierung.unsqueeze(1).unsqueeze(2)
            elif len(maskierung.shape) == 3:
                # (batch_size, query_len, key_len) -> (batch_size, 1, query_len, key_len)
                maskierung = maskierung.unsqueeze(1)
            
            # Sehr negative Werte für maskierte Positionen
            scores = scores.masked_fill(maskierung == 0, float('-inf'))
        
        # Softmax über die Key-Dimension
        aufmerksamkeits_gewichte = F.softmax(scores, dim=-1)
        
        # Dropout auf Aufmerksamkeitsgewichten
        aufmerksamkeits_gewichte = self.aufmerksamkeit_dropout(aufmerksamkeits_gewichte)
        
        # Gewichtete Summe der Values
        # aufmerksamkeits_gewichte: (batch_size, anzahl_koepfe, query_len, key_len)
        # wert: (batch_size, anzahl_koepfe, key_len, dimension_pro_kopf)
        # -> ausgabe: (batch_size, anzahl_koepfe, query_len, dimension_pro_kopf)
        ausgabe = torch.matmul(aufmerksamkeits_gewichte, wert)
        
        return ausgabe, aufmerksamkeits_gewichte
    
    def berechne_aufmerksamkeits_karten(self,
                                       query: Tensor,
                                       schluessel: Tensor,
                                       maskierung: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Berechnet nur die Aufmerksamkeitsgewichte (ohne Wert-Multiplikation).
        
        Args:
            query: Query Tensor
            schluessel: Key Tensor
            maskierung: Optionaler Maskierungstensor
            
        Returns:
            Tuple[Tensor, Tensor]: (Aufmerksamkeitsscores, Aufmerksamkeitsgewichte)
        """
        batch_size = query.size(0)
        
        # Lineare Projektion
        query = self.query_projektion(query)
        schluessel = self.key_projektion(schluessel)
        
        # Reshape für Mehrkopf
        query = query.view(batch_size, -1, self.anzahl_koepfe, self.dimension_pro_kopf)
        schluessel = schluessel.view(batch_size, -1, self.anzahl_koepfe, self.dimension_pro_kopf)
        
        # Transponieren
        query = query.transpose(1, 2)
        schluessel = schluessel.transpose(1, 2)
        
        # Scores berechnen
        scores = torch.matmul(query, schluessel.transpose(-2, -1)) / self.skalierung
        
        # Maskierung anwenden
        if maskierung is not None:
            if len(maskierung.shape) == 2:
                maskierung = maskierung.unsqueeze(1).unsqueeze(2)
            elif len(maskierung.shape) == 3:
                maskierung = maskierung.unsqueeze(1)
            
            scores = scores.masked_fill(maskierung == 0, float('-inf'))
        
        # Softmax für Gewichte
        gewichte = F.softmax(scores, dim=-1)
        
        # Reshape zurück für Rückgabe
        scores = scores.transpose(1, 2).contiguous()
        gewichte = gewichte.transpose(1, 2).contiguous()
        
        return scores, gewichte