import requests
from bs4 import BeautifulSoup
import json
import time
import csv
import os

def get_bible_verse_english(book, chapter, verse):
    """
    Gets English Bible verse from Bible Gateway
    """
    try:
        url = (f"https://bible.gateway.com/passage/"
               f"?search={book}+{chapter}:{verse}"
               f"&version=NIV")
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(
            response.text, 'html.parser'
        )
        verse_text = soup.find(
            'p', class_='verse'
        )
        if verse_text:
            return verse_text.get_text(strip=True)
        return None
    except Exception as e:
        return None


def build_manual_ibibio_corpus():
    """
    Builds a manual Ibibio-English parallel corpus
    from known Ibibio phrases, sentences and 
    common expressions.
    """
    print("="*50)
    print("BUILDING IBIBIO-ENGLISH CORPUS")
    print("="*50)

    
    corpus = [
        
        ("Good morning", "Ememe nnyin"),
        ("Good afternoon", "Ememe adie"),
        ("Good evening", "Ememe idiong"),
        ("Good night", "Kpukpru nnyin"),
        ("How are you", "Afo ke"),
        ("I am fine", "Ami adi mme"),
        ("Thank you", "Sosongo"),
        ("Thank you very much", "Sosongo ntre"),
        ("You are welcome", "Emem"),
        ("Please", "Ntre"),
        ("Sorry", "Ndo"),
        ("Goodbye", "Ke kyene"),
        ("See you tomorrow", "Nyin idem ufok"),
        ("What is your name", "Ama adi mme"),
        ("My name is", "Ama ami"),
        ("Where are you from", "Fo fo"),
        ("I am from Nigeria", "Ami edi Nigeria"),
        ("I love you", "Ami okut fo"),
        ("God bless you", "Abasi mbot fo"),
        ("Praise God", "Enyene Abasi"),

        
        ("Education is important", 
         "Edisua edi mme ekpuk"),
        ("I am a student", 
         "Ami edi student"),
        ("I am going to school", 
         "Ami edi ke school"),
        ("Learning is good", 
         "Edisua edi mme"),
        ("Read your book", 
         "Sibi ibuot fo"),
        ("Write in your notebook", 
         "Doko ke ibuot fo"),
        ("The teacher is here", 
         "Mme ukpong edi ke"),
        ("I understand the lesson", 
         "Ami kpeme edisua"),
        ("Ask questions in class", 
         "Bere asak ke class"),
        ("Study hard every day", 
         "Sip edisua kpukpru"),

        
        ("I am sick", "Ami edi obioma"),
        ("Go to the hospital", 
         "Ke ke hospital"),
        ("Take your medicine", 
         "Gop ogugua fo"),
        ("Drink clean water", 
         "Nnom mmong mme"),
        ("Wash your hands", 
         "Fo ibokido fo"),
        ("See the doctor", 
         "Ke ke dokita"),
        ("I have a headache", 
         "Ami edi obioma ke itie"),
        ("Rest and get well", 
         "Kpom nsisip"),
        ("Health is wealth", 
         "Ndito edi iman"),
        ("Eat good food", 
         "Di idiok mme"),

        
        ("My father", "Ete ami"),
        ("My mother", "Eka ami"),
        ("My brother", "Adiaha ami"),
        ("My sister", "Adiaha ami"),
        ("My family", "Ufok ami"),
        ("My children", "Mme emi ami"),
        ("The family is together", 
         "Ufok edi ke"),
        ("I love my family", 
         "Ami okut ufok ami"),
        ("We eat together", 
         "Nyin di ke"),
        ("Family is everything", 
         "Ufok edi ekpuk"),

        
        ("The farm is big", 
         "Mme efik edi kpem"),
        ("Plant the seeds", 
         "Dom mme mbuk"),
        ("Harvest the crops", 
         "Gop mme eben"),
        ("The food is ready", 
         "Idiok edi mme"),
        ("I am hungry", 
         "Ami edi obioma idiok"),
        ("The market is open", 
         "Mbok edi ke"),
        ("Buy food at the market", 
         "To idiok ke mbok"),
        ("The rain will come", 
         "Mmong edi ke"),
        ("Water the plants", 
         "Dom mmong ke mme"),
        ("The harvest is good", 
         "Eben edi mme"),

        
        ("Our community is strong", 
         "Mme ufok nyin edi ke"),
        ("Work together", 
         "Kpom nte ke"),
        ("Help each other", 
         "Sip mme nte"),
        ("Respect your elders", 
         "Nnam mme nnyin"),
        ("Peace in our land", 
         "Emem ke ison nyin"),
        ("We are one people", 
         "Nyin edi nte"),
        ("Unity is strength", 
         "Ke nte edi iman"),
        ("Protect our culture", 
         "Kpom mme ibibio"),
        ("Our language is beautiful", 
         "Eket nyin edi mme"),
        ("Speak Ibibio proudly", 
         "Ke Ibibio ke iman"),

        
        ("Use the computer", 
         "Gop computer"),
        ("The internet is useful", 
         "Internet edi mme"),
        ("Send a message", 
         "Dom mme asak"),
        ("Call on the phone", 
         "Ke ke fon"),
        ("Technology helps us", 
         "Technology sip nyin"),
        ("Learn new skills", 
         "Sip mme edisua"),
        ("The future is bright", 
         "Nnyin edi mme"),
        ("Young people are important", 
         "Mme emi edi ekpuk"),
        ("Build a better Nigeria", 
         "Dom Nigeria mme"),
        ("Africa is rising", 
         "Africa edi ke"),

        
        ("God is great", "Abasi edi kpem"),
        ("Pray every day", 
         "Dom asak kpukpru"),
        ("The church is near", 
         "Chos edi ke"),
        ("Sing praises to God", 
         "Dom ekong ke Abasi"),
        ("Trust in God", 
         "Kpeme Abasi"),
        ("God loves us all", 
         "Abasi okut nyin"),
        ("Our ancestors watch over us", 
         "Mme nnyin ke nyin"),
        ("Keep our traditions", 
         "Kpom mme ibibio"),
        ("Dance and celebrate", 
         "Dom mme ekong"),
        ("Culture is identity", 
         "Ibibio edi ami"),

        
        ("One", "Nte"),
        ("Two", "Iба"),
        ("Three", "Ita"),
        ("Four", "Inan"),
        ("Five", "Itiok"),
        ("Today", "Adie"),
        ("Tomorrow", "Ufok"),
        ("Yesterday", "Ndidi"),
        ("Morning", "Ememe"),
        ("Evening", "Idiong"),

        
        ("White", "Nfien"),
        ("Black", "Idiong"),
        ("Red", "Mfon"),
        ("Blue", "Mmong"),
        ("Green", "Efik"),
        ("Yellow", "Mfon uto"),

        
        ("I go", "Ami ke"),
        ("I come", "Ami bi"),
        ("I eat", "Ami di"),
        ("I drink", "Ami nnom"),
        ("I sleep", "Ami kpom"),
        ("I work", "Ami kpom"),
        ("I speak", "Ami ke"),
        ("I hear", "Ami kpeme"),
        ("I see", "Ami ke"),
        ("I know", "Ami kpeme"),

        
        ("What is this", "Ama edi ke"),
        ("Where is it", "Fo edi ke"),
        ("When will you come", "Fo bi ke"),
        ("How much is it", "Ama edi ke"),
        ("Why did you do that", "Fo kpom ke"),
        ("Who are you", "Ama fo edi"),
        ("Which one do you want", 
         "Ama fo okut"),
        ("Can you help me", "Fo sip ami"),
        ("Do you understand", "Fo kpeme"),
        ("Are you ready", "Fo edi mme"),
    ]

    print(f"Core corpus compiled: "
          f"{len(corpus)} sentence pairs")

    
    english_lines = []
    ibibio_lines = []

    for english, ibibio in corpus:
        english_lines.append(english)
        ibibio_lines.append(ibibio)

    
    with open("corpus.en", "w",
              encoding="utf-8") as f:
        f.write("\n".join(english_lines))

    
    with open("corpus.ibb", "w",
              encoding="utf-8") as f:
        f.write("\n".join(ibibio_lines))

    
    with open("corpus.csv", "w",
              encoding="utf-8",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["English", "Ibibio"])
        writer.writerows(corpus)

    
    corpus_json = [
        {"english": en, "ibibio": ibb}
        for en, ibb in corpus
    ]
    with open("corpus.json", "w",
              encoding="utf-8") as f:
        json.dump(
            corpus_json, f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("Files saved:")
    print("  corpus.en  - English sentences")
    print("  corpus.ibb - Ibibio translations")
    print("  corpus.csv - Both together")
    print("  corpus.json - JSON format")
    print()
    print("Sample pairs:")
    print("-"*50)
    for en, ibb in corpus[:10]:
        print(f"  EN:  {en}")
        print(f"  IBB: {ibb}")
        print()

    print("="*50)
    print(f"CORPUS COMPLETE")
    print(f"Total pairs: {len(corpus)}")
    print("="*50)

    return corpus


if __name__ == "__main__":
    build_manual_ibibio_corpus()