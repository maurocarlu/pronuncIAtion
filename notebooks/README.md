# 📓 Notebooks

Questa cartella contiene i Jupyter notebook per training e valutazione dei modelli.

## Notebooks Disponibili

| Notebook | Descrizione | Piattaforma |
|----------|-------------|-------------|
| `unified_trainer.ipynb` | **Trainer unificato** - Multi-piattaforma con auto-detection ambiente | Colab/Kaggle/Local |
| `colab_train_augmented.ipynb` | Training WavLM con dataset augmented | Colab |
| `colab_train_wavlm_weighted.ipynb` | Training WavLM con Weighted Layer Sum | Colab |
| `colab_train_xlsr.ipynb` | Training XLS-R multilingual | Colab |
| `colab_ensemble.ipynb` | Late Fusion ensemble di modelli | Colab |

## Come Usare

### Su Google Colab
1. Carica il notebook su Colab
2. Collega Google Drive quando richiesto
3. Assicurati che `phonemeRef.zip` sia in `/MyDrive/`
4. Esegui le celle in ordine

### In Locale
1. Apri con Jupyter Lab/Notebook
2. Esegui dal root del progetto
3. I path si auto-configurano

## Dipendenze
I notebook installano automaticamente le dipendenze necessarie. Vedi `requirements.txt` nella root per la lista completa.

## Note
- I checkpoint vengono salvati automaticamente su Drive (Colab) o in `outputs/` (locale)
- Il training può essere ripreso automaticamente se esistono checkpoint
