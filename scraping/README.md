# WordReference Scraping Tools

Strumenti per lo scraping di dati fonetici da WordReference.

## File

- `wordreference_scraper.py` - Classe principale `WordReferenceScraper` per estrarre fonemi US e audio
- `batch_download.py` - Script per download batch di parole
- `phonetics_progress.csv` - Progressi dello scraping (parole con fonemi US trovati)
- `phonetics_failed.csv` - Parole fallite durante lo scraping
- `wordreference-api-js/` - Versione JavaScript originale (riferimento)

## Utilizzo

### Scraping singola parola

```python
from scraping.wordreference_scraper import WordReferenceScraper

scraper = WordReferenceScraper()
data = scraper.get_word_data('hello', from_lang='en', to_lang='it')

print(data['word'])       # hello
print(data['phonetics'])  # /hɛˈloʊ/
print(data['audio'])      # Lista di URL audio US
```

### Batch download

```bash
python scraping/batch_download.py wordlist.txt en it
```

## Note

- Lo scraper estrae specificamente i **fonemi US** (IPA americano)
- Gli audio vengono filtrati per includere solo quelli **US**
- Delay random tra 1-2.5 secondi per evitare blocchi
