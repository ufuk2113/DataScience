#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wort Embedding Package - Verbesserte Word2Vec Implementierung

Dieses Paket enthält:
- Word2VecEmbedding: Hauptklasse für Skip-Gram Training
- SkipGramModell: Neurales Netzwerk Modell
- SkipGramDatenSatz: Dataset für Training

Autor: Ufuk Baysal
Datum: 2025

"""

from .wort2vec_embedding import Word2VecEmbedding
from .skipgram_modell import SkipGramModell
from .skipgram_datensatz import SkipGramDatenSatz

__all__ = [
    'Word2VecEmbedding',
    'SkipGramModell',
    'SkipGramDatenSatz'
]

print("✅ Wort Embedding Package geladen")