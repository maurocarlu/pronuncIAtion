"""
Modulo centralizzato per normalizzazione IPA.

Questo modulo fornisce una normalizzazione consistente delle stringhe IPA
tra training (WordReference) e evaluation (SpeechOcean762 via ARPA).

Design Principles:
1. Una sola fonte di verità per le convenzioni IPA
2. Normalizzazione bidirezionale (può essere applicata a entrambe le direzioni)
3. Supporto per varianti UK/US
4. Configurabile per diversi use-case (training vs evaluation)
"""

import re
from typing import Optional


# =============================================================================
# Mappature di normalizzazione
# =============================================================================

# Varianti fonetiche da unificare
# Formato: {variante: forma_canonica}
VOWEL_NORMALIZATIONS = {
    # Open-mid front unrounded vowel: ɛ è la forma standard IPA
    # 'e' viene usata in alcune trascrizioni "moderne" ma è tecnicamente diversa
    'e': 'ɛ',  # U+0065 → U+025B (solo in contesto fonemico, non dopo consonanti)
    
    # Dittonghi UK vs US
    'əʊ': 'oʊ',  # UK goat → US goat (normalizza a US)
    
    # R-colored vowels - normalizza a forme senza hook
    'ɝ': 'ɜr',   # U+025D → U+025C + r
    'ɚ': 'ər',   # U+025A → U+0259 + r
    
    # Variante di g
    'ɡ': 'g',    # U+0261 (IPA g) → U+0067 (ASCII g)
}

# Caratteri da rimuovere completamente durante la normalizzazione
REMOVE_CHARS = {
    'ˈ',   # U+02C8 - Primary stress
    'ˌ',   # U+02CC - Secondary stress
    'ː',   # U+02D0 - Long vowel marker
    '.',   # Syllable separator
    'ʳ',   # U+02B3 - Rhotic modifier
}

# Caratteri che rappresentano silenzio/pausa
SILENCE_CHARS = {
    ' ',   # Space
    '|',   # Word boundary
    ' ',   # U+2004 - Three-per-em space
}


# =============================================================================
# Classe principale
# =============================================================================

class IPANormalizer:
    """
    Normalizzatore IPA configurabile.
    
    Modes:
        - 'strict': Rimuove stress, normalizza tutte le varianti
        - 'preserve_stress': Mantiene stress markers, normalizza varianti
        - 'minimal': Solo normalizzazione varianti essenziali (e→ɛ, ɡ→g)
    """
    
    def __init__(
        self,
        mode: str = 'strict',
        normalize_e_to_epsilon: bool = True,
        normalize_r_colored: bool = True,
        remove_stress: bool = True,
        remove_length: bool = True,
        unify_diphthongs: bool = True,
    ):
        """
        Args:
            mode: 'strict', 'preserve_stress', o 'minimal'
            normalize_e_to_epsilon: Se True, converte 'e' → 'ɛ'
            normalize_r_colored: Se True, converte ɝ→ɜr, ɚ→ər
            remove_stress: Se True, rimuove ˈ e ˌ
            remove_length: Se True, rimuove ː
            unify_diphthongs: Se True, normalizza əʊ→oʊ
        """
        self.mode = mode
        
        # Configurazione basata sul mode
        if mode == 'strict':
            self.normalize_e = normalize_e_to_epsilon
            self.normalize_r = normalize_r_colored
            self.remove_stress = True
            self.remove_length = True
            self.unify_diphthongs = True
        elif mode == 'preserve_stress':
            self.normalize_e = normalize_e_to_epsilon
            self.normalize_r = normalize_r_colored
            self.remove_stress = False
            self.remove_length = False
            self.unify_diphthongs = unify_diphthongs
        elif mode == 'minimal':
            self.normalize_e = normalize_e_to_epsilon
            self.normalize_r = False
            self.remove_stress = False
            self.remove_length = False
            self.unify_diphthongs = False
        else:
            # Custom mode - use provided parameters
            self.normalize_e = normalize_e_to_epsilon
            self.normalize_r = normalize_r_colored
            self.remove_stress = remove_stress
            self.remove_length = remove_length
            self.unify_diphthongs = unify_diphthongs
    
    def normalize(self, text: str) -> str:
        """
        Normalizza una stringa IPA.
        
        Args:
            text: Stringa IPA da normalizzare
            
        Returns:
            Stringa IPA normalizzata
        """
        if not isinstance(text, str):
            return ""
        
        result = text
        
        # 1. Rimuovi delimitatori di trascrizione (/ /)
        result = result.strip()
        if result.startswith('/'):
            result = result[1:]
        if result.endswith('/'):
            result = result[:-1]
        
        # 2. Rimuovi stress markers se richiesto
        if self.remove_stress:
            result = result.replace('ˈ', '')
            result = result.replace('ˌ', '')
        
        # 3. Rimuovi length markers se richiesto
        if self.remove_length:
            result = result.replace('ː', '')
        
        # 4. Rimuovi altri caratteri non fonemici
        result = result.replace('ʳ', '')
        result = result.replace('.', '')
        
        # 5. Normalizza spazi
        result = result.replace(' ', '')  # U+2004
        result = result.replace(' ', '')  # Regular space
        result = result.replace('|', '')  # Word boundary
        
        # 6. Normalizza varianti di g
        result = result.replace('ɡ', 'g')
        
        # 7. Unifica dittonghi UK/US
        if self.unify_diphthongs:
            result = result.replace('əʊ', 'oʊ')
        
        # 8. Normalizza r-colored vowels
        if self.normalize_r:
            result = result.replace('ɝ', 'ɜr')
            result = result.replace('ɚ', 'ər')
            # Anche la forma con ː
            result = result.replace('ɜːr', 'ɜr')
        
        # 9. Normalizza e → ɛ (deve essere fatto con attenzione)
        # Solo in posizione vocalica, non parte di dittonghi già processati
        if self.normalize_e:
            # Prima proteggi i dittonghi
            result = result.replace('eɪ', '##EY##')
            # Poi converti e isolate
            result = result.replace('e', 'ɛ')
            # Ripristina dittonghi
            result = result.replace('##EY##', 'eɪ')
        
        return result
    
    def normalize_for_comparison(self, text: str) -> str:
        """
        Normalizza per confronto (usato in evaluation).
        Equivalente a normalize() con mode='strict'.
        """
        # Usa sempre strict mode per i confronti
        strict = IPANormalizer(mode='strict')
        return strict.normalize(text)


# =============================================================================
# Funzioni di convenienza
# =============================================================================

def normalize_ipa(text: str, mode: str = 'strict') -> str:
    """
    Funzione di convenienza per normalizzare IPA.
    
    Args:
        text: Stringa IPA
        mode: 'strict', 'preserve_stress', o 'minimal'
        
    Returns:
        Stringa normalizzata
    """
    normalizer = IPANormalizer(mode=mode)
    return normalizer.normalize(text)


def normalize_for_training(text: str) -> str:
    """
    Normalizza IPA per il training.
    Mantiene le informazioni rilevanti ma rimuove inconsistenze.
    """
    normalizer = IPANormalizer(
        mode='custom',
        normalize_e_to_epsilon=False,  # Mantieni e/ɛ come nel source
        normalize_r_colored=False,     # Mantieni r-colored come nel source
        remove_stress=True,            # Rimuovi stress per consistenza
        remove_length=True,
        unify_diphthongs=True,
    )
    return normalizer.normalize(text)


def normalize_for_evaluation(text: str) -> str:
    """
    Normalizza IPA per l'evaluation.
    Applica normalizzazione completa per confronto equo.
    """
    return normalize_ipa(text, mode='strict')


# =============================================================================
# Mappatura ARPA → IPA corretta
# =============================================================================

def get_corrected_arpa_to_ipa_mapping():
    """
    Restituisce la mappatura ARPA→IPA corretta che si allinea
    alle convenzioni standard IPA usate in WordReference.
    
    Correzioni rispetto alla mappatura originale:
    - EH → ɛ (invece di e)
    - Mantiene le altre convenzioni standard
    """
    return {
        # Monophthongs
        'AO': 'ɔ', 'AO0': 'ɔ', 'AO1': 'ɔ', 'AO2': 'ɔ',
        'AA': 'ɑ', 'AA0': 'ɑ', 'AA1': 'ɑ', 'AA2': 'ɑ',
        'IY': 'i', 'IY0': 'i', 'IY1': 'i', 'IY2': 'i',
        'UW': 'u', 'UW0': 'u', 'UW1': 'u', 'UW2': 'u',
        
        # CORREZIONE: EH → ɛ (convenzione IPA standard)
        'EH': 'ɛ', 'EH0': 'ɛ', 'EH1': 'ɛ', 'EH2': 'ɛ',
        
        'IH': 'ɪ', 'IH0': 'ɪ', 'IH1': 'ɪ', 'IH2': 'ɪ',
        'UH': 'ʊ', 'UH0': 'ʊ', 'UH1': 'ʊ', 'UH2': 'ʊ',
        'AH': 'ʌ', 'AH0': 'ə', 'AH1': 'ʌ', 'AH2': 'ʌ',
        'AE': 'æ', 'AE0': 'æ', 'AE1': 'æ', 'AE2': 'æ',
        'AX': 'ə', 'AX0': 'ə', 'AX1': 'ə', 'AX2': 'ə',
        
        # Diphthongs
        'EY': 'eɪ', 'EY0': 'eɪ', 'EY1': 'eɪ', 'EY2': 'eɪ',
        'AY': 'aɪ', 'AY0': 'aɪ', 'AY1': 'aɪ', 'AY2': 'aɪ',
        'OW': 'oʊ', 'OW0': 'oʊ', 'OW1': 'oʊ', 'OW2': 'oʊ',
        'AW': 'aʊ', 'AW0': 'aʊ', 'AW1': 'aʊ', 'AW2': 'aʊ',
        'OY': 'ɔɪ', 'OY0': 'ɔɪ', 'OY1': 'ɔɪ', 'OY2': 'ɔɪ',
        
        # R-colored vowels
        'ER': 'ɜr', 'ER0': 'ɜr', 'ER1': 'ɜr', 'ER2': 'ɜr',
        'AXR': 'ər', 'AXR0': 'ər', 'AXR1': 'ər', 'AXR2': 'ər',
        
        # Stops
        'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'g',
        
        # Affricates
        'CH': 'tʃ', 'JH': 'dʒ',
        
        # Fricatives
        'F': 'f', 'V': 'v', 'TH': 'θ', 'DH': 'ð',
        'S': 's', 'Z': 'z', 'SH': 'ʃ', 'ZH': 'ʒ', 'HH': 'h',
        
        # Nasals
        'M': 'm', 'EM': 'm̩', 'N': 'n', 'EN': 'n̩', 'NG': 'ŋ', 'ENG': 'ŋ̍',
        
        # Liquids
        'L': 'l', 'EL': 'ɫ̩', 'R': 'r', 'DX': 'ɾ', 'NX': 'ɾ̃',
        
        # Semivowels
        'W': 'w', 'Y': 'j', 'Q': 'ʔ',
    }


def arpa_to_ipa(arpa_phone: str, use_corrected: bool = True) -> Optional[str]:
    """
    Converte un fonema ARPABET in IPA.
    
    Args:
        arpa_phone: Fonema ARPABET (es. 'EH1', 'T', 'AY0')
        use_corrected: Se True, usa la mappatura corretta (EH→ɛ)
        
    Returns:
        Stringa IPA o None se non trovato
    """
    mapping = get_corrected_arpa_to_ipa_mapping() if use_corrected else None
    
    if mapping is None:
        # Fallback alla mappatura originale
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "arpa2ipa"))
        from arpa2ipa._arpa_to_ipa import arpa_to_ipa_lookup
        mapping = arpa_to_ipa_lookup
    
    # Lookup diretto
    if arpa_phone in mapping:
        return mapping[arpa_phone]
    
    # Fallback: rimuovi numeri di stress
    phone_clean = ''.join(c for c in arpa_phone if not c.isdigit())
    return mapping.get(phone_clean)
