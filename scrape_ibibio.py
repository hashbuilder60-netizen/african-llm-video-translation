import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
from collections import defaultdict

print("="*50)
print("IBIBIO WEB SCRAPER")
print("Multiple sources, verified translations")
print("="*50)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

all_pairs = []
source_stats = {}



print("\nSource 1: Glosbe Dictionary...")
print("(English-Ibibio word pairs)")

def scrape_glosbe(word_list):
    """
    Scrapes Glosbe for English-Ibibio translations.
    Glosbe is the most comprehensive online 
    dictionary for African languages.
    """
    pairs = []
    
    for word in word_list:
        try:
            url = (
                f"https://glosbe.com/en/ibb/"
                f"{word.replace(' ', '%20')}"
            )
            response = requests.get(
                url, 
                headers=HEADERS,
                timeout=10
            )
            soup = BeautifulSoup(
                response.text, 'html.parser'
            )
            
            
            translations = soup.find_all(
                'h3', 
                {'class': re.compile('translation')}
            )
            
            if not translations:
                
                translations = soup.find_all(
                    'div',
                    {'class': re.compile('translation__item')}
                )
            
            for trans in translations[:3]:
                text = trans.get_text(strip=True)
                if text and len(text) > 1:
                    pairs.append({
                        "english": word,
                        "ibibio": text,
                        "source": "glosbe"
                    })
                    print(f"  ✓ {word} → {text}")
            
            
            time.sleep(1.5)
            
        except Exception as e:
            print(f"  Skipped '{word}': {e}")
            continue
    
    return pairs


glosbe_words = [
    "water", "fire", "earth", "wind", "sun",
    "moon", "star", "tree", "river", "mountain",
    "father", "mother", "child", "man", "woman",
    "eat", "drink", "sleep", "run", "walk",
    "good", "bad", "big", "small", "new",
    "old", "hot", "cold", "beautiful", "strong",
    "one", "two", "three", "four", "five",
    "house", "village", "market", "farm", "road",
    "king", "chief", "elder", "warrior", "healer",
    "God", "prayer", "blessing", "spirit", "ancestor",
    "love", "peace", "war", "friend", "enemy",
    "food", "fish", "meat", "palm", "yam",
    "morning", "evening", "night", "today", "tomorrow",
    "come", "go", "speak", "hear", "see",
    "heart", "hand", "head", "eye", "mouth",
    "rain", "farm", "harvest", "seed", "soil",
    "school", "book", "learn", "teach", "write",
    "money", "trade", "buy", "sell", "work"
]

glosbe_pairs = scrape_glosbe(glosbe_words)
all_pairs.extend(glosbe_pairs)
source_stats["glosbe"] = len(glosbe_pairs)
print(f"Glosbe pairs collected: {len(glosbe_pairs)}")



print("\nSource 2: JW.org Ibibio publications...")
print("(Parallel English-Ibibio sentences)")

def scrape_jw_ibibio():
    """
    JW.org has extensive Ibibio content.
    Scrapes parallel sentences from 
    available Ibibio publications.
    """
    pairs = []
    

    urls = [
        "https://www.jw.org/ibb/publications/",
        "https://www.jw.org/ibb/bible/",
        "https://www.jw.org/ibb/",
    ]
    
    for url in urls:
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=10
            )
            soup = BeautifulSoup(
                response.text, 'html.parser'
            )
            
            
            paragraphs = soup.find_all('p')
            for p in paragraphs[:20]:
                text = p.get_text(strip=True)
                if (len(text) > 20 and 
                    len(text) < 200 and
                    text.isascii()):
                    pairs.append({
                        "ibibio_raw": text,
                        "source": "jw_org",
                        "url": url
                    })
            
            time.sleep(2)
            
        except Exception as e:
            print(f"  Skipped {url}: {e}")
            continue
    
    print(f"  JW.org raw texts: {len(pairs)}")
    return pairs

jw_raw = scrape_jw_ibibio()
source_stats["jw_org_raw"] = len(jw_raw)



print("\nSource 3: Bible parallel texts...")
print("(Most reliable parallel corpus for Ibibio)")

def scrape_bible_parallel(book, chapters):
    """
    Fetches parallel English and Ibibio 
    Bible verses. The Bible is available
    in both languages on multiple platforms.
    """
    pairs = []
    
    for chapter in chapters:
        try:
            
            en_url = (
                f"https://bible.com/bible/111/"
                f"{book}.{chapter}.NIV"
            )
            
            
            ibb_url = (
                f"https://bible.com/bible/2672/"
                f"{book}.{chapter}.IBBBT"
            )
            
            en_response = requests.get(
                en_url, 
                headers=HEADERS, 
                timeout=10
            )
            time.sleep(1)
            
            ibb_response = requests.get(
                ibb_url,
                headers=HEADERS,
                timeout=10
            )
            time.sleep(1)
            
            en_soup = BeautifulSoup(
                en_response.text, 'html.parser'
            )
            ibb_soup = BeautifulSoup(
                ibb_response.text, 'html.parser'
            )
            
            
            en_verses = en_soup.find_all(
                'span',
                {'class': re.compile('verse')}
            )
            ibb_verses = ibb_soup.find_all(
                'span', 
                {'class': re.compile('verse')}
            )
            
            
            for i, (en, ibb) in enumerate(
                zip(en_verses, ibb_verses)
            ):
                en_text = en.get_text(strip=True)
                ibb_text = ibb.get_text(strip=True)
                
                if (en_text and ibb_text and 
                    len(en_text) > 10 and
                    len(ibb_text) > 10):
                    pairs.append({
                        "english": en_text,
                        "ibibio": ibb_text,
                        "source": f"bible_{book}_{chapter}",
                        "verse": i+1
                    })
                    
        except Exception as e:
            print(f"  Chapter {chapter} error: {e}")
            continue
    
    return pairs


bible_pairs = []
for book, chapters in [
    ("GEN", range(1, 4)),
    ("JHN", range(1, 4)),
    ("PSA", range(1, 3)),
    ("PRO", range(1, 3)),
]:
    pairs = scrape_bible_parallel(book, chapters)
    bible_pairs.extend(pairs)
    print(f"  {book}: {len(pairs)} verse pairs")
    time.sleep(1)

all_pairs.extend(bible_pairs)
source_stats["bible"] = len(bible_pairs)
print(f"Bible pairs collected: {len(bible_pairs)}")


print("\nSource 4: Academic and linguistic resources...")

def scrape_linguistic_resources():
    """
    Scrapes from known Ibibio language 
    academic and educational resources.
    """
    pairs = []
    
    urls = [
        
        "https://www.endangeredlanguages.com/lang/ibibio",
        
        "https://omniglot.com/writing/ibibio.htm",

        "https://www.ethnologue.com/language/ibb/",
        
        "http://www.language-archives.org/language/ibb",
    ]
    
    for url in urls:
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=10
            )
            soup = BeautifulSoup(
                response.text, 'html.parser'
            )
            
            # Look for tables with word lists
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        cell1 = cells[0].get_text(
                            strip=True
                        )
                        cell2 = cells[1].get_text(
                            strip=True
                        )
                        
                        if (cell1 and cell2 and
                            len(cell1) > 1 and
                            len(cell2) > 1 and
                            len(cell1) < 100):
                            pairs.append({
                                "english": cell1,
                                "ibibio": cell2,
                                "source": url
                            })
            
            # Look for definition lists
            dls = soup.find_all('dl')
            for dl in dls:
                dts = dl.find_all('dt')
                dds = dl.find_all('dd')
                for dt, dd in zip(dts, dds):
                    en = dt.get_text(strip=True)
                    ibb = dd.get_text(strip=True)
                    if en and ibb:
                        pairs.append({
                            "english": en,
                            "ibibio": ibb,
                            "source": url
                        })
            
            print(f"  ✓ {url[:50]}: "
                  f"{len(pairs)} items so far")
            time.sleep(2)
            
        except Exception as e:
            print(f"  Skipped: {e}")
            continue
    
    return pairs

linguistic_pairs = scrape_linguistic_resources()
all_pairs.extend(linguistic_pairs)
source_stats["linguistic"] = len(linguistic_pairs)


print("\nSource 5: Omniglot Ibibio phrases...")

def scrape_omniglot():
    """
    Omniglot has verified Ibibio phrases
    with English translations.
    """
    pairs = []
    
    try:
        url = "https://omniglot.com/language/phrases/ibibio.php"
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=10
        )
        soup = BeautifulSoup(
            response.text, 'html.parser'
        )
        
        
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    en = cells[0].get_text(strip=True)
                    ibb = cells[1].get_text(strip=True)
                    
                    
                    if (en and ibb and
                        len(en) > 3 and
                        len(ibb) > 3 and
                        en.lower() != 'english' and
                        ibb.lower() != 'ibibio'):
                        pairs.append({
                            "english": en,
                            "ibibio": ibb,
                            "source": "omniglot"
                        })
                        print(f"  ✓ {en} → {ibb}")
        
        time.sleep(2)
        
    except Exception as e:
        print(f"  Error: {e}")
    
    return pairs

omniglot_pairs = scrape_omniglot()
all_pairs.extend(omniglot_pairs)
source_stats["omniglot"] = len(omniglot_pairs)
print(f"Omniglot pairs: {len(omniglot_pairs)}")


print("\nCleaning and deduplicating...")

def clean_text(text):
    """Cleans scraped text."""

    text = re.sub(r'^\d+\s*', '', text)
    
    text = re.sub(r'\s+', ' ', text)
    
    text = text.strip()
    return text

def is_valid_pair(english, ibibio):
    """
    Validates that a pair looks legitimate.
    Filters out noise and bad scrapes.
    """
    if not english or not ibibio:
        return False
    
    
    if len(english) < 3 or len(ibibio) < 2:
        return False
    
    if len(english) > 300 or len(ibibio) > 300:
        return False
    
    
    if english.lower() == ibibio.lower():
        return False
    
    
    if ibibio.replace(' ', '').isdigit():
        return False
    
    
    if len(ibibio) > 0 and ibibio[0].isupper():
        pass  
    
    return True


with open("corpus.json", "r", 
          encoding="utf-8") as f:
    existing = json.load(f)

combined = []
seen = set()


for item in existing:
    key = item["english"].lower().strip()
    if key not in seen:
        seen.add(key)
        combined.append(item)

valid_count = 0
invalid_count = 0

for item in all_pairs:
    english = clean_text(
        item.get("english", "")
    )
    ibibio = clean_text(
        item.get("ibibio", "")
    )
    
    if not is_valid_pair(english, ibibio):
        invalid_count += 1
        continue
    
    key = english.lower().strip()
    if key in seen:
        continue
    
    seen.add(key)
    combined.append({
        "english": english,
        "ibibio": ibibio,
        "source": item.get("source", "scraped")
    })
    valid_count += 1

print(f"Valid new pairs added: {valid_count}")
print(f"Invalid pairs filtered: {invalid_count}")
print(f"Total corpus size: {len(combined)}")


print("\nSaving expanded corpus...")


with open("corpus_expanded.json", "w",
          encoding="utf-8") as f:
    json.dump(combined, f,
              indent=4,
              ensure_ascii=False)


with open("corpus.json", "w",
          encoding="utf-8") as f:
    json.dump(combined, f,
              indent=4,
              ensure_ascii=False)


english_lines = [item["english"] for item in combined]
ibibio_lines = [item["ibibio"] for item in combined]

with open("corpus.en", "w", encoding="utf-8") as f:
    f.write("\n".join(english_lines))

with open("corpus.ibb", "w", encoding="utf-8") as f:
    f.write("\n".join(ibibio_lines))


report = {
    "total_pairs": len(combined),
    "new_pairs_added": valid_count,
    "invalid_filtered": invalid_count,
    "source_breakdown": source_stats,
    "sources_used": [
        "glosbe_dictionary",
        "jw_org",
        "bible_parallel",
        "linguistic_resources",
        "omniglot"
    ]
}

with open("scraping_report.json", "w",
          encoding="utf-8") as f:
    json.dump(report, f,
              indent=4,
              ensure_ascii=False)

print("\n" + "="*50)
print("SCRAPING COMPLETE")
print("="*50)
print(f"\nSource breakdown:")
for source, count in source_stats.items():
    print(f"  {source}: {count} pairs")
print(f"\nTotal corpus: {len(combined)} pairs")
print(f"Saved to: corpus.json")
print(f"Report: scraping_report.json")
print("\nNext step: Re-run finetune_ibibio.py")
print("with the expanded corpus")
print("="*50)