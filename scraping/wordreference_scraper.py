"""
WordReference Scraper - Versione Python
Basato su wordreference-api (JavaScript)

Estrae parole, fonemi (US) e audio (US) da WordReference.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from typing import Optional, List, Dict, Any
import re


class WordReferenceScraper:
    """
    Scraper per WordReference che estrae:
    - Parola
    - Fonemi (pronuncia US)
    - Audio (pronuncia US)
    - Traduzioni
    """
    
    VALID_LANGUAGES = ['es', 'en', 'it', 'fr', 'de', 'pt', 'pl', 'ro', 'cz', 'gr', 'tr', 'zh', 'ja', 'ko', 'ar']
    BASE_URL = "https://www.wordreference.com"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def validate_language(self, lang: str) -> str:
        """Valida che la lingua sia supportata."""
        if lang.lower() not in self.VALID_LANGUAGES:
            raise ValueError(f"Lingua non supportata: {lang}. Lingue valide: {self.VALID_LANGUAGES}")
        return lang.lower()
    
    def get_word_data(self, word: str, from_lang: str = 'en', to_lang: str = 'es') -> Dict[str, Any]:
        """
        Ottiene i dati di una parola da WordReference.
        
        Args:
            word: La parola da cercare
            from_lang: Lingua di partenza (default: 'en')
            to_lang: Lingua di destinazione (default: 'es')
            
        Returns:
            Dict con word, phonetics, audio, translations
        """
        from_lang = self.validate_language(from_lang)
        to_lang = self.validate_language(to_lang)
        
        url = f"{self.BASE_URL}/{from_lang}{to_lang}/{word}"
        
        response = self.session.get(url)
        response.raise_for_status()
        
        return self._process_html(response.text, word)
    
    def _process_html(self, html: str, original_word: str) -> Dict[str, Any]:
        """
        Processa l'HTML e estrae i dati.
        
        Args:
            html: HTML della pagina
            original_word: Parola originale cercata
            
        Returns:
            Dict con i dati estratti
        """
        soup = BeautifulSoup(html, 'html.parser')
        result = {}
        
        # Estrai la parola
        header_word = soup.find('h3', class_='headerWord')
        result['word'] = header_word.text.strip() if header_word else original_word
        
        # Estrai i fonemi - cerca specificamente la pronuncia US
        result['phonetics'] = self._extract_us_phonetics(soup)
        
        # Estrai tutti gli audio
        all_audio = self._extract_audio(soup)
        
        # Filtra solo audio US
        result['audio'] = [a for a in all_audio if self._is_us_audio(a)]
        result['audio_all'] = all_audio  # Mantieni anche tutti gli audio per riferimento
        
        # Estrai le traduzioni
        result['translations'] = self._extract_translations(soup)
        
        return result
    
    def _extract_us_phonetics(self, soup: BeautifulSoup) -> str:
        """
        Estrae la pronuncia fonetica US (solo IPA, prima variante).
        
        WordReference mostra i fonemi in span tooltip:
        - pronWR: pronuncia UK (es. /ækˈnɒlɪdʒ/)
        - "USA pronunciation: IPA": pronuncia US IPA (es. /ækˈnɑlɪdʒ/)
        - pronRH: pronuncia US respelling (es. (ak nol′ij)) - NON IPA!
        
        Estraiamo solo la PRIMA variante IPA US (prima della virgola se presente).
        Esempio: /hɛˈloʊ, hə-, ˈhɛloʊ/ -> /hɛˈloʊ/
        """
        # METODO 1 (PRIORITARIO): Cerca span con "USA pronunciation: IPA"
        # Questo contiene la vera pronuncia US in formato IPA
        for span in soup.find_all('span', class_='tooltip'):
            text = span.get_text()
            if 'USA pronunciation: IPA' in text or 'USA pronunciation:IPA' in text:
                # Estrai l'IPA da questo span
                ipa_match = re.search(r'/([^/]+)/', text)
                if ipa_match:
                    full_ipa = ipa_match.group(1)
                    # Prendi solo la prima variante (prima della virgola)
                    first_variant = full_ipa.split(',')[0].strip()
                    return '/' + first_variant + '/'
        
        # METODO 2: Cerca nel widget di pronuncia tutti gli IPA e prendi quello US
        # WordReference mostra: UK:/ækˈnɒlɪdʒ/ US:/ækˈnɑlɪdʒ/
        pron_widget = soup.find('div', id='pronunciation_widget')
        if pron_widget:
            text = pron_widget.get_text()
            # Cerca pattern "US:" o "US :" seguito da IPA
            us_match = re.search(r'US\s*:\s*/([^/]+)/', text)
            if us_match:
                full_ipa = us_match.group(1)
                first_variant = full_ipa.split(',')[0].strip()
                return '/' + first_variant + '/'
        
        # METODO 3: Cerca nell'header tutti gli IPA, il secondo è solitamente US
        header_area = soup.find('div', id='articleHead')
        if header_area:
            text = header_area.get_text()
            # Trova tutti gli IPA
            all_ipa = re.findall(r'/([^/]+)/', text)
            # Filtra quelli che sembrano IPA validi (non troppo lunghi, non frasi)
            valid_ipa = [ipa for ipa in all_ipa if len(ipa) < 30 and not any(c in ipa for c in ['\\n', 'acknowledge', 'recognize'])]
            
            if len(valid_ipa) >= 2:
                # Se ci sono 2+ IPA, il secondo è tipicamente US
                # Verifica: se il primo ha ɒ (UK) e il secondo ha ɑ (US), prendi il secondo
                if 'ɒ' in valid_ipa[0] and 'ɑ' in valid_ipa[1]:
                    first_variant = valid_ipa[1].split(',')[0].strip()
                    return '/' + first_variant + '/'
                # Altrimenti prendi il secondo (US)
                first_variant = valid_ipa[1].split(',')[0].strip()
                return '/' + first_variant + '/'
            elif valid_ipa:
                # Se c'è solo un IPA, usalo
                first_variant = valid_ipa[0].split(',')[0].strip()
                return '/' + first_variant + '/'
        
        # METODO 4: Fallback - cerca pronWR (UK) se non c'è alternativa US
        pron_wr = soup.find('span', class_='pronWR')
        if pron_wr:
            text = pron_wr.get_text()
            ipa_match = re.search(r'/([^/]+)/', text)
            if ipa_match:
                full_ipa = ipa_match.group(1)
                first_variant = full_ipa.split(',')[0].strip()
                return '/' + first_variant + '/'
        
        return ""
        
        return ""
    
    def _extract_audio(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """
        Estrae tutti gli URL degli audio.
        
        Returns:
            Lista di dict con 'url' e 'type' (us/uk/other)
        """
        audio_list = []
        
        # METODO 1: Cerca negli script JavaScript (var audioFiles = [...])
        for script in soup.find_all('script'):
            script_text = script.string or ''
            if 'audioFiles' in script_text:
                # Estrai l'array audioFiles
                match = re.search(r'var\s+audioFiles\s*=\s*\[(.*?)\]', script_text, re.DOTALL)
                if match:
                    audio_array = match.group(1)
                    # Estrai tutti i path
                    paths = re.findall(r"'([^']+\.mp3)'", audio_array)
                    for path in paths:
                        full_url = self._make_full_url(path)
                        audio_type = self._determine_audio_type_from_path(path)
                        audio_list.append({
                            'url': full_url,
                            'type': audio_type
                        })
        
        # METODO 2: Cerca nel widget audio (tag <audio>)
        listen_widget = soup.find('div', id='listen_widget')
        if listen_widget:
            for audio in listen_widget.find_all('audio'):
                for source in audio.find_all('source'):
                    src = source.get('src', '')
                    if src:
                        full_url = self._make_full_url(src)
                        if not any(a['url'] == full_url for a in audio_list):
                            audio_type = self._determine_audio_type(src, audio)
                            audio_list.append({
                                'url': full_url,
                                'type': audio_type
                            })
        
        # METODO 3: Cerca audio inline
        for audio in soup.find_all('audio'):
            for source in audio.find_all('source'):
                src = source.get('src', '')
                if src:
                    full_url = self._make_full_url(src)
                    if not any(a['url'] == full_url for a in audio_list):
                        audio_type = self._determine_audio_type(src, audio)
                        audio_list.append({
                            'url': full_url,
                            'type': audio_type
                        })
        
        return audio_list
    
    def _determine_audio_type_from_path(self, path: str) -> str:
        """
        Determina il tipo di audio dal path.
        
        Esempi di path:
        - /audio/en/us/us/en042667.mp3 -> us
        - /audio/en/uk/general/en042667.mp3 -> uk
        - /audio/en/us/south/en042667.mp3 -> us
        """
        path_lower = path.lower()
        
        # Controlla per US (vari accenti americani)
        if '/us/' in path_lower:
            return 'us'
        
        # Controlla per UK (vari accenti britannici)
        if '/uk/' in path_lower or '/rp/' in path_lower:
            return 'uk'
        
        # Altri accenti
        if '/scot/' in path_lower or '/scottish/' in path_lower:
            return 'scottish'
        if '/irish/' in path_lower:
            return 'irish'
        if '/jamaica/' in path_lower:
            return 'jamaican'
        if '/australia/' in path_lower or '/aus/' in path_lower:
            return 'australian'
        
        return 'other'
    
    def _determine_audio_type(self, src: str, audio_element) -> str:
        """
        Determina se l'audio è US, UK o altro.
        """
        src_lower = src.lower()
        
        # Controlla l'URL
        if 'en/us/' in src_lower or '_us' in src_lower or '/us/' in src_lower:
            return 'us'
        if 'en/uk/' in src_lower or '_uk' in src_lower or '/uk/' in src_lower:
            return 'uk'
        
        # Controlla attributi dell'elemento audio
        audio_id = audio_element.get('id', '').lower() if audio_element else ''
        if 'us' in audio_id:
            return 'us'
        if 'uk' in audio_id:
            return 'uk'
        
        # Controlla il parent per context
        parent = audio_element.parent if audio_element else None
        if parent:
            parent_text = parent.get_text().lower()
            if 'us' in parent_text and 'uk' not in parent_text:
                return 'us'
            if 'uk' in parent_text and 'us' not in parent_text:
                return 'uk'
        
        # Default: sconosciuto (potrebbe essere US)
        # WordReference spesso mette US come primo/default
        return 'unknown'
    
    def _is_us_audio(self, audio_info: Dict[str, str]) -> bool:
        """Verifica se l'audio è US."""
        return audio_info['type'] == 'us'
    
    def _make_full_url(self, src: str) -> str:
        """Converte URL relativo in assoluto."""
        if src.startswith('//'):
            return 'https:' + src
        if src.startswith('/'):
            return self.BASE_URL + src
        if not src.startswith('http'):
            return self.BASE_URL + '/' + src
        return src
    
    def _extract_translations(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Estrae le tabelle di traduzione.
        """
        translations = []
        
        tables = soup.find_all('table', class_='WRD')
        for table in tables:
            translation_table = self._parse_translation_table(table)
            if translation_table['translations']:  # Solo se ha contenuto
                translations.append(translation_table)
        
        return translations
    
    def _parse_translation_table(self, table) -> Dict[str, Any]:
        """
        Parsa una singola tabella di traduzione.
        """
        result = {
            'title': '',
            'translations': []
        }
        
        rows = table.find_all('tr')
        
        for row in rows:
            row_class = row.get('class', [])
            row_id = row.get('id', '')
            
            # Header
            if 'wrtopsection' in row_class:
                result['title'] = row.get_text(strip=True)
            
            # Traduzione principale (ha id e class even/odd)
            elif row_id and ('even' in row_class or 'odd' in row_class):
                translation = self._create_translation_item(row)
                result['translations'].append(translation)
            
            # Esempio (no id, class even/odd)
            elif not row_id and ('even' in row_class or 'odd' in row_class):
                self._add_example_to_last(result, row)
        
        return result
    
    def _create_translation_item(self, row) -> Dict[str, Any]:
        """
        Crea un item di traduzione da una riga.
        """
        # Parola originale
        from_elem = row.find('strong')
        from_word = from_elem.get_text(strip=True) if from_elem else ''
        
        # Tipo della parola originale
        fr_wrd = row.find(class_='FrWrd')
        from_type = ''
        if fr_wrd:
            em = fr_wrd.find('em')
            if em:
                # Rimuovi span interni
                for span in em.find_all('span'):
                    span.decompose()
                from_type = em.get_text(strip=True)
        
        # Traduzione e tipo
        to_wrd = row.find(class_='ToWrd')
        to_word = ''
        to_type = ''
        if to_wrd:
            em = to_wrd.find('em')
            if em:
                for span in em.find_all('span'):
                    span.decompose()
                to_type = em.get_text(strip=True)
                em.decompose()
            to_word = to_wrd.get_text(strip=True)
        
        return {
            'from': from_word,
            'fromType': from_type,
            'to': to_word,
            'toType': to_type,
            'example': {
                'from': [],
                'to': []
            }
        }
    
    def _add_example_to_last(self, result: Dict, row) -> None:
        """
        Aggiunge un esempio all'ultima traduzione.
        """
        if not result['translations']:
            return
        
        last_translation = result['translations'][-1]
        
        # Esempio nella lingua di partenza
        fr_ex = row.find(class_='FrEx')
        if fr_ex and fr_ex.get_text(strip=True):
            last_translation['example']['from'].append(fr_ex.get_text(strip=True))
        
        # Esempio nella lingua di destinazione
        to_ex = row.find(class_='ToEx')
        if to_ex and to_ex.get_text(strip=True):
            last_translation['example']['to'].append(to_ex.get_text(strip=True))
    
    def save_to_json(self, data: Dict[str, Any], filepath: str) -> None:
        """
        Salva i dati in un file JSON.
        
        Args:
            data: Dati da salvare
            filepath: Percorso del file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def download_audio(self, audio_list: List[Dict[str, str]], 
                       output_dir: str, 
                       filter_usa: bool = True) -> List[str]:
        """
        Scarica i file audio.
        
        Args:
            audio_list: Lista di dict con 'url' e 'type'
            output_dir: Directory di output
            filter_usa: Se True, scarica solo audio US
            
        Returns:
            Lista dei file scaricati
        """
        os.makedirs(output_dir, exist_ok=True)
        downloaded = []
        
        for i, audio_info in enumerate(audio_list):
            # Filtra per US se richiesto
            if filter_usa and audio_info['type'] not in ['us', 'unknown']:
                continue
            
            url = audio_info['url']
            
            try:
                response = self.session.get(url)
                response.raise_for_status()
                
                # Determina l'estensione dal content-type o dall'URL
                ext = '.mp3'
                if 'audio/ogg' in response.headers.get('content-type', ''):
                    ext = '.ogg'
                elif 'audio/wav' in response.headers.get('content-type', ''):
                    ext = '.wav'
                elif url.endswith('.ogg'):
                    ext = '.ogg'
                elif url.endswith('.wav'):
                    ext = '.wav'
                
                # Nome file
                filename = f"audio_{audio_info['type']}_{i}{ext}"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                downloaded.append(filepath)
                
            except Exception as e:
                print(f"Errore download audio {url}: {e}")
        
        return downloaded
    
    def get_us_phonetics_only(self, word: str, from_lang: str = 'en', to_lang: str = 'es') -> str:
        """
        Metodo di convenienza per ottenere solo i fonemi US.
        
        Args:
            word: Parola da cercare
            from_lang: Lingua di partenza
            to_lang: Lingua di destinazione
            
        Returns:
            Stringa con i fonemi US
        """
        data = self.get_word_data(word, from_lang, to_lang)
        return data.get('phonetics', '')
    
    def get_us_audio_urls(self, word: str, from_lang: str = 'en', to_lang: str = 'es') -> List[str]:
        """
        Metodo di convenienza per ottenere solo gli URL audio US.
        
        Args:
            word: Parola da cercare
            from_lang: Lingua di partenza
            to_lang: Lingua di destinazione
            
        Returns:
            Lista di URL audio US
        """
        data = self.get_word_data(word, from_lang, to_lang)
        return [a['url'] for a in data.get('audio', [])]


# Test se eseguito direttamente
if __name__ == "__main__":
    scraper = WordReferenceScraper()
    
    # Test con una parola
    test_word = "hello"
    print(f"Test scraping per: {test_word}")
    print("=" * 50)
    
    try:
        data = scraper.get_word_data(test_word, from_lang='en', to_lang='it')
        
        print(f"Parola: {data['word']}")
        print(f"Fonemi US: {data['phonetics']}")
        print(f"Audio US: {len(data['audio'])} file")
        for audio in data['audio']:
            print(f"  - {audio['type']}: {audio['url']}")
        print(f"Traduzioni: {len(data['translations'])} tabelle")
        
        # Salva esempio
        scraper.save_to_json(data, 'test_output/hello.json')
        print(f"\nDati salvati in test_output/hello.json")
        
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
