import json
import csv


with open("corpus.json", "r", 
          encoding="utf-8") as f:
    existing = json.load(f)

print(f"Existing pairs: {len(existing)}")


additional = [
    
    ("Vote for your leader",
     "Dom mme nnyin"),
    ("Pay your taxes",
     "Dom mme iman"),
    ("Obey the law",
     "Kpom mme edidem"),
    ("Nigeria is our country",
     "Nigeria edi ison nyin"),
    ("We are all citizens",
     "Nyin edi mme nte"),
    ("The government serves us",
     "Edidem sip nyin"),
    ("Justice for all",
     "Emem ke nyin"),
    ("Equal rights for everyone",
     "Mme iman ke nyin"),
    ("Corruption is wrong",
     "Obioma edi mme"),
    ("Build our nation together",
     "Dom ison nyin ke"),

    
    ("Plant trees everywhere",
     "Dom mme ediong"),
    ("Keep our environment clean",
     "Kpom ison nyin mme"),
    ("Do not pollute the river",
     "Kpom mmong mme"),
    ("Protect the forest",
     "Kpom mme ediong"),
    ("Climate change affects us",
     "Ison ke nyin"),
    ("Save water every day",
     "Kpom mmong kpukpru"),
    ("Recycle your waste",
     "Dom mme obioma"),
    ("The earth is our home",
     "Ison edi ufok nyin"),
    ("Clean air is important",
     "Ison edi ekpuk"),
    ("Nature is beautiful",
     "Ison edi mme"),

    
    ("Football is popular in Nigeria",
     "Football edi mme ke Nigeria"),
    ("Exercise every morning",
     "Kpom edisua ememe"),
    ("Swimming is good for health",
     "Mmong edi mme ndito"),
    ("Play with your friends",
     "Dom ke mme nte"),
    ("Win the competition",
     "Dom mme ekong"),
    ("Sports build character",
     "Ekong dom ami"),
    ("Run fast and win",
     "Ke iman dom"),
    ("Team work is important",
     "Ke nte edi ekpuk"),
    ("Practice every day",
     "Kpom edisua kpukpru"),
    ("Never give up",
     "Kpom mme nte"),

    
    ("Buy and sell goods",
     "To dom mme iman"),
    ("Save your money",
     "Kpom iman fo"),
    ("Start a business",
     "Dom mme iman"),
    ("Hard work pays",
     "Kpom iman edi mme"),
    ("Invest in the future",
     "Dom iman ke nnyin"),
    ("The economy is growing",
     "Iman edi ke"),
    ("Create jobs for youth",
     "Dom mme kpom ke emi"),
    ("Trade with your neighbors",
     "Dom mme iman ke nte"),
    ("Financial freedom matters",
     "Iman edi ekpuk"),
    ("Build wealth gradually",
     "Dom iman kpukpru"),

    
    ("The road is long",
     "Ison edi kpem"),
    ("Drive carefully",
     "Ke mme iman"),
    ("The bus is coming",
     "Mme ke bi"),
    ("We travel by road",
     "Nyin ke ke ison"),
    ("The journey is safe",
     "Ke edi mme"),
    ("Arrive on time",
     "Bi ke iman"),
    ("The bridge is strong",
     "Mme ison edi iman"),
    ("Cross the river safely",
     "Ke mmong mme"),
    ("The airport is far",
     "Mme ke edi kpem"),
    ("Take the right road",
     "Ke ison mme"),

    
    ("It is raining",
     "Mmong edi ke"),
    ("The sun is shining",
     "Akai edi ke"),
    ("It is very hot today",
     "Ison edi iman adie"),
    ("The weather is cool",
     "Ison edi mme"),
    ("Prepare for the rain",
     "Dom mme mmong"),
    ("The dry season is here",
     "Ison edi ke"),
    ("Flood has destroyed farms",
     "Mmong dom mme efik"),
    ("The harmattan is coming",
     "Ison bi ke"),
    ("Stay indoors during storms",
     "Ke ufok ke ison"),
    ("The sky is clear today",
     "Ison edi mme adie"),

    
    ("I am happy",
     "Ami edi mme"),
    ("I am sad",
     "Ami edi obioma"),
    ("I am angry",
     "Ami edi iman"),
    ("I am afraid",
     "Ami edi obioma"),
    ("I am proud",
     "Ami edi iman"),
    ("I am grateful",
     "Ami sosongo"),
    ("I am tired",
     "Ami edi obioma"),
    ("I am excited",
     "Ami edi mme"),
    ("I am surprised",
     "Ami edi ke"),
    ("I am confused",
     "Ami edi obioma"),

    
    ("A child who is not embraced by the village will burn it down to feel its warmth",
     "Emi edi mme nte ke ufok"),
    ("The forest would be silent if no bird sang except the one that sang best",
     "Ediong edi mme ke nte"),
    ("Rain does not fall on one roof alone",
     "Mmong ke ufok nte"),
    ("Until the lion learns to write every story will glorify the hunter",
     "Nyin kpom mme edisua"),
    ("A tree is straightened while it is young",
     "Ediong dom ke emi"),
    ("Knowledge is like a garden if it is not cultivated it cannot be harvested",
     "Edisua edi mme efik"),
    ("One who tells the stories rules the world",
     "Nte ke mme ison"),
    ("Speak softly and carry a big stick",
     "Ke mme iman dom"),
    ("Unity is strength division is weakness",
     "Ke nte edi iman"),
    ("The elder who speaks truth is worth more than gold",
     "Nnyin ke mme edi iman"),

    
    ("Ekpe is our traditional masquerade",
     "Ekpe edi mme ibibio"),
    ("Ekombi is our cultural dance",
     "Ekombi edi mme ekong"),
    ("Akwa Ibom is our homeland",
     "Akwa Ibom edi ison nyin"),
    ("Ibibio people are hardworking",
     "Mme ibibio kpom iman"),
    ("Our language connects us",
     "Eket nyin ke nyin"),
    ("The Ibibio culture is rich",
     "Mme ibibio edi kpem"),
    ("We celebrate our heritage",
     "Nyin dom mme ibibio"),
    ("Traditional medicine heals",
     "Ogugua ibibio sip ndito"),
    ("Respect the village square",
     "Nnam mme ufok"),
    ("The king leads the people",
     "Edidem ke mme nte"),

    
    ("Artificial intelligence is changing the world",
     "AI dom ison"),
    ("Computers help us work faster",
     "Computer sip nyin kpom"),
    ("The internet connects people",
     "Internet ke mme nte"),
    ("Technology improves our lives",
     "Technology dom ndito nyin"),
    ("African languages need AI support",
     "Mme eket Africa ke AI"),
    ("Digital tools empower communities",
     "Mme digital dom ufok"),
    ("Mobile phones are everywhere",
     "Fon edi kpukpru"),
    ("Social media spreads information",
     "Media dom mme asak"),
    ("Coding is a valuable skill",
     "Coding edi edisua mme"),
    ("The future belongs to technology",
     "Nnyin edi technology"),
]

print(f"Additional pairs: {len(additional)}")


combined = [
    (item["english"], item["ibibio"])
    for item in existing
]
combined.extend(additional)

print(f"Total combined pairs: {len(combined)}")


english_lines = [pair[0] for pair in combined]
ibibio_lines = [pair[1] for pair in combined]

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
    writer.writerows(combined)

corpus_json = [
    {"english": en, "ibibio": ibb}
    for en, ibb in combined
]
with open("corpus.json", "w",
          encoding="utf-8") as f:
    json.dump(
        corpus_json, f,
        indent=4,
        ensure_ascii=False
    )

print()
print("="*50)
print("EXPANDED CORPUS COMPLETE")
print(f"Total sentence pairs: {len(combined)}")
print("Files updated:")
print("  corpus.en")
print("  corpus.ibb")
print("  corpus.csv")
print("  corpus.json")
print("="*50)