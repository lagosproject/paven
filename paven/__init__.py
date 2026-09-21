"""
PAVEN: A Perceptual Algorithm for Versatile video Encoding using Neural networks
Paper: Elsevier Engineering Applications of Artificial Intelligence (EAAI), 2025.
DOI: 10.1016/j.engappai.2025.111664
UPM Thesis: https://oa.upm.es/88254/
Authors: Antonio Jesús Díaz-Honrubia, Pablo Fernández-Lagos, Roberto Valle, Jesús Bescós
"""

__version__ = "0.1.0"
__author__ = "Pablo Fernández Lagos, Antonio Jesús Díaz-Honrubia"

from .model import PavenModel, VideoSaliencyModel, YUVVideoSaliencyModel
from .qp_grid import QpGrid
from .yuv_loader import YUVReader, YUVFileLoader
from .loss import kldiv, similarity, cc, nss

__all__ = [
    "PavenModel",
    "VideoSaliencyModel",
    "YUVVideoSaliencyModel",
    "QpGrid",
    "YUVReader",
    "YUVFileLoader",
    "kldiv",
    "similarity",
    "cc",
    "nss",
    "__version__",
]
