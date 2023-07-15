def get_class_labels(dataset=None):
    if not dataset:
        return [
            "nausea",
            "diarrhea",
            "fatigue",
            "vomiting",
            "loss of appetite",
            "headache",
            "fever",
            "interstitial lung disease",
            "liver damage",
            "dizziness",
            "pain",
            "alopecia",
            "analgesic asthma syndrome",
            "renal impairment",
            "hypersensitivity",
            "insomnia",
            "constipation",
            "bone marrow dysfunction",
            "abdominal pain",
            "hemorrhagic cystitis",
            "rash",
            "stomatitis",
        ]
    else:
        labels = [
            label
            for label in dataset["train"].features.keys()
            if label not in ["train_id", "text", "test_id"]
        ]
        return labels


def symptoms_for_generation():
    return [
        "Übelkeit",
        "Durchfall",
        "Müdigkeit",
        "Erbrechen",
        "Appetitlosigkeit",
        "Kopfschmerzen",
        "Fieber",
        "interstitieller Lungenentzündung",
        "Leberschädigung",
        "Schwindel",
        "Schmerz",
        "Alopezie",
        "Aspirin-Asthma",
        "Nierenschäden",
        "allergischer Reaktion",
        "Schlaflosigkeit",
        "Verstopfung",
        "Knochenmarksuppression",
        "Bauchschmerzen",
        "hämorrhagischer Zystitis",
        "Ausschlag",
        "Mundgeschwüren",
    ]


def drug_examples():
    return [
        "Aspirin",
        "Paracetamol",
        "Ibuprofen",
        "Acetaminophen",
        "Penicillin",
        "Amoxicillin",
        "Ciprofloxacin",
        "Prednisolon",
        "Omeprazol",
        "Simvastatin",
        "Metformin",
        "Amlodipin",
        "Warfarin",
        "Metoprolol",
        "Losartan",
        "Cisplatin",
        "Azanin",
        "Infliximab",
        "Inflix",
        "Methotrexat",
        "Mesalazin",
        "Metformin",
    ]
