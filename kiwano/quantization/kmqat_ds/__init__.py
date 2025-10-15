"""
KMQAT-DS helpers: I/O, pre-processing (L2/CMVN, PCA), clustering (cosine K-means),
and calibration splits without speaker overlap.

This package centralizes reusable logic for the KMQAT-DS data-subsets pipeline.
"""
from .io import *
from .prep import *
from .cluster import *
from .splits import *