"""
Moduli per gestione dati e preprocessing.
"""

from .preprocessor import PhonemePreprocessor, preprocess_dataset
from .normalize_ipa import (
    IPANormalizer,
    normalize_ipa,
    normalize_for_training,
    normalize_for_evaluation,
    arpa_to_ipa,
    get_corrected_arpa_to_ipa_mapping,
)

__all__ = [
    "PhonemePreprocessor",
    "preprocess_dataset",
    "IPANormalizer",
    "normalize_ipa",
    "normalize_for_training",
    "normalize_for_evaluation",
    "arpa_to_ipa",
    "get_corrected_arpa_to_ipa_mapping",
]

