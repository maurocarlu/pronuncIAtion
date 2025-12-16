"""
Script per verificare il modulo di normalizzazione IPA.

Esegue test sul modulo normalize_ipa per assicurarsi che
le convenzioni siano allineate tra training e evaluation.
"""

import sys
from pathlib import Path

# Force UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.normalize_ipa import (
    IPANormalizer,
    normalize_ipa,
    normalize_for_evaluation,
    arpa_to_ipa,
    get_corrected_arpa_to_ipa_mapping,
)

# Define IPA characters using Unicode escapes
EPSILON = '\u025b'      # open-mid front unrounded vowel
OPEN_MID_CENTRAL = '\u025c'  # open-mid central unrounded vowel
SCHWA = '\u0259'        # schwa
NEAR_CLOSE = '\u026a'   # near-close near-front unrounded vowel
ASH = '\u00e6'          # ash
ENG = '\u014b'          # eng
IPA_G = '\u0261'        # IPA g
RHOTACIZED = '\u025d'   # rhotacized vowel
SCHWA_HOOK = '\u025a'   # schwa with hook
STRESS = '\u02c8'       # primary stress
OPEN_O = '\u0254'       # open-mid back rounded vowel
TURNED_V = '\u028c'     # open-mid back unrounded vowel
ESH = '\u0283'          # esh (sh sound)
HORSESHOE = '\u028a'    # near-close near-back rounded vowel


def test_arpa_to_ipa_corrections():
    """Testa che la mappatura ARPA-IPA sia corretta."""
    print("=== Test Mappatura ARPA to IPA ===")
    print()
    
    # Test case: EH deve mappare a epsilon, non a e
    test_cases = [
        ('EH', EPSILON, "EH deve mappare a epsilon (U+025B)"),
        ('EH1', EPSILON, "EH1 deve mappare a epsilon"),
        ('ER', OPEN_MID_CENTRAL + 'r', "ER deve mappare a U+025C + r"),
        ('IH', NEAR_CLOSE, "IH deve mappare a U+026A"),
        ('AE', ASH, "AE deve mappare a ash (U+00E6)"),
        ('NG', ENG, "NG deve mappare a eng (U+014B)"),
    ]
    
    all_passed = True
    for arpa, expected_ipa, description in test_cases:
        result = arpa_to_ipa(arpa, use_corrected=True)
        status = "PASS" if result == expected_ipa else "FAIL"
        if result != expected_ipa:
            all_passed = False
        print(f"  [{status}] {description}")
        # Print hex codes for clarity
        result_hex = ' '.join(f"U+{ord(c):04X}" for c in result) if result else "None"
        expected_hex = ' '.join(f"U+{ord(c):04X}" for c in expected_ipa)
        print(f"      Got: {result_hex}")
        print(f"      Expected: {expected_hex}")
    
    print()
    return all_passed


def test_normalization_modes():
    """Testa le diverse modalita di normalizzazione."""
    print("=== Test Modalita Normalizzazione ===")
    print()
    
    # Input di test con stress markers e varianti
    test_input = "/" + STRESS + "t" + EPSILON + "st" + NEAR_CLOSE + ENG + "/"
    
    print(f"Input hex: {' '.join(f'U+{ord(c):04X}' for c in test_input)}")
    print()
    
    for mode in ['strict', 'preserve_stress', 'minimal']:
        normalizer = IPANormalizer(mode=mode)
        result = normalizer.normalize(test_input)
        result_hex = ' '.join(f"U+{ord(c):04X}" for c in result) if result else "empty"
        print(f"  Mode '{mode}': {result_hex}")
    
    print()


def test_vowel_normalizations():
    """Testa la normalizzazione delle varianti vocaliche."""
    print("=== Test Normalizzazione Vocali ===")
    print()
    
    normalizer = IPANormalizer(mode='strict')
    
    test_cases = [
        (IPA_G + 'e' + 't', 'g' + EPSILON + 't', "g IPA -> g ASCII, e -> epsilon"),
        (RHOTACIZED, OPEN_MID_CENTRAL + 'r', "rhotacized -> open-mid + r"),
        (SCHWA_HOOK, SCHWA + 'r', "schwa hook -> schwa + r"),
        (SCHWA + HORSESHOE, 'o' + HORSESHOE, "UK diphthong -> US diphthong"),
        (STRESS + "str" + EPSILON + "s", "str" + EPSILON + "s", "stress marker rimosso"),
    ]
    
    all_passed = True
    for input_ipa, expected, description in test_cases:
        result = normalizer.normalize(input_ipa)
        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_passed = False
        print(f"  [{status}] {description}")
        in_hex = ' '.join(f"U+{ord(c):04X}" for c in input_ipa)
        out_hex = ' '.join(f"U+{ord(c):04X}" for c in result)
        exp_hex = ' '.join(f"U+{ord(c):04X}" for c in expected)
        print(f"      Input:    {in_hex}")
        print(f"      Got:      {out_hex}")
        print(f"      Expected: {exp_hex}")
    
    print()
    return all_passed


def test_real_examples():
    """Testa su esempi reali da training e SpeechOcean."""
    print("=== Test Esempi Reali ===")
    print()
    
    normalizer = IPANormalizer(mode='strict')
    
    # Esempi da SpeechOcean (ARPA converted)
    speechocean_arpa = [
        ['M', 'AA1', 'R', 'K'],      # MARK
        ['IH0', 'Z'],                # IS
        ['G', 'OW1', 'IH0', 'NG'],   # GOING
    ]
    
    print("SpeechOcean examples (ARPA -> IPA):")
    for phones in speechocean_arpa:
        ipa_list = [arpa_to_ipa(p, use_corrected=True) for p in phones]
        ipa_joined = ''.join(ipa_list)
        normalized = normalizer.normalize(ipa_joined)
        word = ''.join(phones)
        joined_hex = ' '.join(f"U+{ord(c):04X}" for c in ipa_joined)
        norm_hex = ' '.join(f"U+{ord(c):04X}" for c in normalized)
        print(f"  {phones}")
        print(f"    IPA: {joined_hex}")
        print(f"    Norm: {norm_hex}")
    
    print()


def main():
    print("=" * 60)
    print("[TEST] MODULO NORMALIZZAZIONE IPA")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("ARPA to IPA Mapping", test_arpa_to_ipa_corrections()))
    test_normalization_modes()
    results.append(("Vowel Normalizations", test_vowel_normalizations()))
    test_real_examples()
    
    print("=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]: {name}")
    
    all_passed = all(r[1] for r in results)
    print()
    print("=" * 60)
    if all_passed:
        print("[OK] Tutti i test passati!")
    else:
        print("[ERROR] Alcuni test falliti - verificare output sopra")
    print("=" * 60)


if __name__ == "__main__":
    main()
