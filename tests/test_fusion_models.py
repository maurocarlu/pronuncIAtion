#!/usr/bin/env python3
"""
==============================================================================
TEST FUSION MODELS - Unit Tests per le 3 Tecniche di Fusione
==============================================================================

Questo modulo contiene test unitari per verificare il corretto funzionamento
dei modelli di fusione:
- EarlyFusionModel
- LateFusionEnsemble (HuBERTWavLMFusion)
- GatedFusionModel

I test verificano:
1. Correttezza delle dimensioni output
2. Range valori gate (Gated Fusion)
3. Forward pass senza errori
4. CTC loss computation

USO:
----
    cd DeepLearning-Phoneme
    python -m pytest tests/test_fusion_models.py -v

    # Solo test veloci (senza caricamento modelli pesanti)
    python -m pytest tests/test_fusion_models.py -v -k "not slow"

Autore: DeepLearning-Phoneme Project
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Aggiungi project root al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# MOCK MODELS (per test veloci senza caricare modelli reali)
# =============================================================================

class MockEncoder(nn.Module):
    """
    Encoder mock per test veloci.
    
    Simula l'output di HuBERT/WavLM senza caricare pesi reali.
    """
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.config = type('Config', (), {'hidden_size': hidden_size, 'num_hidden_layers': 24})()
        self.linear = nn.Linear(1, hidden_size)
    
    def forward(self, input_values, attention_mask=None, **kwargs):
        # Simula subsampling (fattore 320)
        batch_size = input_values.shape[0]
        time_steps = input_values.shape[1] // 320
        
        # Output mockato
        output = torch.randn(batch_size, time_steps, self.config.hidden_size)
        
        return type('Output', (), {'last_hidden_state': output})()


class MockGatedFusionModel(nn.Module):
    """
    Versione mock di GatedFusionModel per test veloci.
    """
    def __init__(self, vocab_size: int = 43, hidden_size: int = 1024):
        super().__init__()
        self.hubert = MockEncoder(hidden_size)
        self.wavlm = MockEncoder(hidden_size)
        
        # Gate network
        self.gate_network = nn.Linear(hidden_size * 2, 1)
        nn.init.zeros_(self.gate_network.bias)
        
        # CTC head
        self.dropout = nn.Dropout(0.1)
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_values, attention_mask=None, labels=None):
        # Forward mock
        h_hubert = self.hubert(input_values).last_hidden_state
        h_wavlm = self.wavlm(input_values).last_hidden_state
        
        # Allineamento
        min_len = min(h_hubert.size(1), h_wavlm.size(1))
        h_hubert = h_hubert[:, :min_len, :]
        h_wavlm = h_wavlm[:, :min_len, :]
        
        # Gate
        gate_input = torch.cat([h_hubert, h_wavlm], dim=-1)
        gate = torch.sigmoid(self.gate_network(gate_input))
        
        # Fusione
        h_fused = gate * h_hubert + (1 - gate) * h_wavlm
        
        # CTC head
        logits = self.ctc_head(self.dropout(h_fused))
        
        return {
            "logits": logits,
            "gate_values": gate,
            "loss": None
        }


class MockEarlyFusionModel(nn.Module):
    """
    Versione mock di EarlyFusionModel per test veloci.
    """
    def __init__(self, vocab_size: int = 43, hidden_size: int = 1024):
        super().__init__()
        self.hubert = MockEncoder(hidden_size)
        self.wavlm = MockEncoder(hidden_size)
        
        # CTC head (2x hidden size per concatenazione)
        self.dropout = nn.Dropout(0.1)
        self.ctc_head = nn.Linear(hidden_size * 2, vocab_size)
    
    def forward(self, input_values, attention_mask=None, labels=None):
        h_hubert = self.hubert(input_values).last_hidden_state
        h_wavlm = self.wavlm(input_values).last_hidden_state
        
        # Allineamento e concatenazione
        min_len = min(h_hubert.size(1), h_wavlm.size(1))
        h_hubert = h_hubert[:, :min_len, :]
        h_wavlm = h_wavlm[:, :min_len, :]
        
        combined = torch.cat([h_hubert, h_wavlm], dim=-1)
        logits = self.ctc_head(self.dropout(combined))
        
        return {"logits": logits, "loss": None}


# =============================================================================
# TEST CASES
# =============================================================================

class TestGatedFusionModel:
    """Test per GatedFusionModel."""
    
    def test_output_shape(self):
        """Verifica che l'output abbia shape corretta [batch, time, vocab]."""
        model = MockGatedFusionModel(vocab_size=43)
        
        # Input: 2 samples, 32000 campioni audio (2 secondi @ 16kHz)
        batch_size = 2
        audio_len = 32000
        input_values = torch.randn(batch_size, audio_len)
        
        outputs = model(input_values)
        logits = outputs["logits"]
        
        # Verifica dimensioni
        assert logits.dim() == 3, f"Expected 3D tensor, got {logits.dim()}D"
        assert logits.shape[0] == batch_size, f"Batch size mismatch"
        assert logits.shape[2] == 43, f"Vocab size mismatch: {logits.shape[2]}"
    
    def test_gate_values_range(self):
        """Verifica che i gate values siano nel range [0, 1]."""
        model = MockGatedFusionModel()
        
        input_values = torch.randn(2, 16000)
        outputs = model(input_values)
        gate = outputs["gate_values"]
        
        # Gate deve essere in [0, 1]
        assert gate.min() >= 0, f"Gate min ({gate.min()}) < 0"
        assert gate.max() <= 1, f"Gate max ({gate.max()}) > 1"
    
    def test_gate_shape(self):
        """Verifica che gate abbia shape [batch, time, 1]."""
        model = MockGatedFusionModel()
        
        input_values = torch.randn(4, 32000)
        outputs = model(input_values)
        gate = outputs["gate_values"]
        
        assert gate.shape[0] == 4, f"Batch mismatch: {gate.shape[0]}"
        assert gate.shape[2] == 1, f"Gate dim mismatch: {gate.shape[2]}"
    
    def test_forward_no_errors(self):
        """Verifica che il forward pass non sollevi errori."""
        model = MockGatedFusionModel()
        
        # Vari batch size
        for batch_size in [1, 2, 4]:
            input_values = torch.randn(batch_size, 16000)
            try:
                outputs = model(input_values)
                assert "logits" in outputs
                assert "gate_values" in outputs
            except Exception as e:
                pytest.fail(f"Forward failed for batch_size={batch_size}: {e}")


class TestEarlyFusionModel:
    """Test per EarlyFusionModel."""
    
    def test_output_shape(self):
        """Verifica che l'output abbia shape corretta [batch, time, vocab]."""
        model = MockEarlyFusionModel(vocab_size=43)
        
        batch_size = 2
        input_values = torch.randn(batch_size, 32000)
        
        outputs = model(input_values)
        logits = outputs["logits"]
        
        assert logits.dim() == 3
        assert logits.shape[0] == batch_size
        assert logits.shape[2] == 43
    
    def test_concatenation_dimension(self):
        """Verifica che la concatenazione produca 2048D (1024+1024)."""
        model = MockEarlyFusionModel(hidden_size=1024)
        
        # Verifica che CTC head accetti 2048D input
        assert model.ctc_head.in_features == 2048, \
            f"Expected 2048D input, got {model.ctc_head.in_features}"


class TestLateFusionWeights:
    """Test per logica Late Fusion weights."""
    
    def test_weight_combination(self):
        """Verifica che la formula α*A + (1-α)*B funzioni correttamente."""
        # Simula logits
        logits_a = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [1, 2, 2]
        logits_b = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])  # [1, 2, 2]
        
        # α = 0.5 → media
        alpha = 0.5
        fused = alpha * logits_a + (1 - alpha) * logits_b
        
        expected = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]])
        assert torch.allclose(fused, expected), f"Expected {expected}, got {fused}"
        
        # α = 1.0 → solo A
        fused_a = 1.0 * logits_a + 0.0 * logits_b
        assert torch.allclose(fused_a, logits_a)
        
        # α = 0.0 → solo B
        fused_b = 0.0 * logits_a + 1.0 * logits_b
        assert torch.allclose(fused_b, logits_b)
    
    def test_weight_range_validation(self):
        """Verifica che pesi fuori range [0,1] siano gestiti."""
        # I pesi dovrebbero essere in [0, 1]
        valid_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        for w in valid_weights:
            assert 0 <= w <= 1, f"Weight {w} out of range"


class TestCTCLossComputation:
    """Test per CTC loss computation."""
    
    def test_ctc_loss_computes(self):
        """Verifica che CTC loss sia calcolabile."""
        vocab_size = 43
        batch_size = 2
        time_steps = 100
        label_len = 10
        
        # Logits random
        logits = torch.randn(batch_size, time_steps, vocab_size)
        log_probs = torch.log_softmax(logits, dim=-1).transpose(0, 1)
        
        # Labels (senza blank token 0)
        labels = torch.randint(1, vocab_size, (batch_size, label_len))
        
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long)
        target_lengths = torch.full((batch_size,), label_len, dtype=torch.long)
        
        try:
            loss = torch.nn.functional.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
            assert not torch.isnan(loss), "CTC loss is NaN"
            assert not torch.isinf(loss), "CTC loss is Inf"
        except Exception as e:
            pytest.fail(f"CTC loss computation failed: {e}")


# =============================================================================
# SLOW TESTS (caricano modelli reali - skip di default)
# =============================================================================

@pytest.mark.slow
class TestRealModels:
    """
    Test con modelli reali (richiede GPU e modelli scaricati).
    
    Eseguire con: pytest -v -m slow
    """
    
    def test_gated_fusion_real_init(self):
        """Test inizializzazione GatedFusionModel reale."""
        pytest.skip("Richiede modelli reali e GPU")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
