"""
Script per scaricare batch di parole da WordReference
Utile per costruire dataset di parole con fonemi e audio
"""

import os
import sys
import time
from wordreference_scraper import WordReferenceScraper


def download_word_list(words, from_lang='en', to_lang='es', 
                       output_dir='dataset', download_audio=True,
                       filter_usa=True, delay=2):
    """
    Scarica dati per una lista di parole.
    
    Args:
        words: Lista di parole da scaricare
        from_lang: Lingua di partenza (default: 'en')
        to_lang: Lingua di destinazione (default: 'es')
        output_dir: Directory dove salvare i dati (default: 'dataset')
        download_audio: Se True, scarica anche gli audio (default: True)
        filter_usa: Se True, scarica solo audio USA (default: True)
        delay: Secondi di pausa tra le richieste (default: 2)
    """
    scraper = WordReferenceScraper()
    
    # Crea directory di output
    json_dir = os.path.join(output_dir, 'json')
    audio_dir = os.path.join(output_dir, 'audio')
    os.makedirs(json_dir, exist_ok=True)
    if download_audio:
        os.makedirs(audio_dir, exist_ok=True)
    
    results = {
        'success': [],
        'failed': [],
        'total': len(words)
    }
    
    print(f"Scaricamento di {len(words)} parole...")
    print(f"Lingue: {from_lang} → {to_lang}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    for i, word in enumerate(words, 1):
        try:
            print(f"\n[{i}/{len(words)}] {word}")
            
            # Scarica dati
            data = scraper.get_word_data(word, from_lang=from_lang, to_lang=to_lang)
            
            print(f"  ✓ Parola: {data['word']}")
            print(f"  ✓ Fonemi: {data['phonetics']}")
            print(f"  ✓ Audio: {len(data['audio'])} file")
            print(f"  ✓ Traduzioni: {len(data['translations'])} tabelle")
            
            # Salva JSON
            json_path = os.path.join(json_dir, f"{word}.json")
            scraper.save_to_json(data, json_path)
            
            # Scarica audio
            audio_count = 0
            if download_audio and data['audio']:
                word_audio_dir = os.path.join(audio_dir, word)
                downloaded = scraper.download_audio(
                    data['audio'], 
                    output_dir=word_audio_dir,
                    filter_usa=filter_usa
                )
                audio_count = len(downloaded)
                print(f"  ✓ Scaricati {audio_count} audio")
            
            results['success'].append({
                'word': word,
                'json': json_path,
                'audio_count': audio_count
            })
            
            # Pausa per non sovraccaricare il server
            if i < len(words):
                time.sleep(delay)
            
        except Exception as e:
            print(f"  ✗ Errore: {e}")
            results['failed'].append({
                'word': word,
                'error': str(e)
            })
    
    # Riepilogo
    print("\n" + "=" * 60)
    print("RIEPILOGO")
    print("=" * 60)
    print(f"Totale parole: {results['total']}")
    print(f"✓ Successo: {len(results['success'])}")
    print(f"✗ Fallite: {len(results['failed'])}")
    
    if results['failed']:
        print("\nParole fallite:")
        for item in results['failed']:
            print(f"  - {item['word']}: {item['error']}")
    
    # Salva report
    report_path = os.path.join(output_dir, 'download_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Download Report\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Lingue: {from_lang} → {to_lang}\n")
        f.write(f"Totale parole: {results['total']}\n")
        f.write(f"Successo: {len(results['success'])}\n")
        f.write(f"Fallite: {len(results['failed'])}\n\n")
        
        f.write("Parole scaricate con successo:\n")
        for item in results['success']:
            f.write(f"  - {item['word']} ({item['audio_count']} audio)\n")
        
        if results['failed']:
            f.write("\nParole fallite:\n")
            for item in results['failed']:
                f.write(f"  - {item['word']}: {item['error']}\n")
    
    print(f"\nReport salvato in: {report_path}")
    
    return results


def load_words_from_file(filepath):
    """Carica lista di parole da un file di testo (una parola per riga)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


def main():
    """Esempio di utilizzo."""
    
    # Opzione 1: Lista di parole inline
    words = [
        'hello',
        'world',
        'computer',
        'internet',
        'science',
        'technology',
        'python',
        'programming',
        'database',
        'algorithm'
    ]
    
    # Opzione 2: Da file (decommenta per usare)
    # words = load_words_from_file('wordlist.txt')
    
    # Scarica le parole
    results = download_word_list(
        words=words,
        from_lang='en',
        to_lang='it',
        output_dir='my_dataset',
        download_audio=True,
        filter_usa=True,
        delay=2  # Pausa di 2 secondi tra le richieste
    )
    
    print(f"\n✅ Download completato!")
    print(f"Scaricate {len(results['success'])}/{results['total']} parole")


if __name__ == "__main__":
    # Se viene passato un file come argomento
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Caricamento parole da: {filepath}")
        words = load_words_from_file(filepath)
        
        # Parametri opzionali
        from_lang = sys.argv[2] if len(sys.argv) > 2 else 'en'
        to_lang = sys.argv[3] if len(sys.argv) > 3 else 'es'
        
        results = download_word_list(
            words=words,
            from_lang=from_lang,
            to_lang=to_lang,
            output_dir='dataset',
            download_audio=True,
            filter_usa=True,
            delay=2
        )
    else:
        # Usa l'esempio predefinito
        main()
