# A small, manually curated reference of well-established drug-condition
# treatment relationships, from general public medical knowledge.
# This exists specifically to compensate for a known limitation: the
# trained NER/RE models only learned CAUSES/ASSOCIATED relations from
# adverse-event corpora (BC5CDR/BioRED/ADE), which contain no genuine
# "TREATS" relation type. Without this reference, common drugs are
# sometimes misclassified as harmful for the condition they actually treat.

KNOWN_INDICATIONS = {
    "metformin": ["type 2 diabetes", "diabetes", "type 2 diabetes mellitus"],
    "atorvastatin": ["high cholesterol", "hypercholesterolemia", "heart disease"],
    "lisinopril": ["high blood pressure", "hypertension", "heart disease"],
    "amlodipine": ["high blood pressure", "hypertension", "angina", "chest pain"],
    "albuterol": ["asthma", "copd", "chronic obstructive pulmonary disease"],
    "salbutamol": ["asthma", "copd", "chronic obstructive pulmonary disease"],
    "omeprazole": ["gerd", "gastroesophageal reflux disease", "stomach ulcers", "peptic ulcer"],
    "levothyroxine": ["hypothyroidism", "low thyroid function"],
    "sertraline": ["depression", "anxiety", "anxiety disorders"],
    "amoxicillin": ["pneumonia", "ear infections", "bacterial infection"],
    "insulin": ["type 1 diabetes", "type 2 diabetes", "diabetes"],
    "clonidine": ["hypertension", "high blood pressure"],
    "aspirin": ["pain", "headache", "fever", "inflammation"],
    "ibuprofen": ["pain", "inflammation", "fever"],
    "paracetamol": ["pain", "fever"],
    "acetaminophen": ["pain", "fever"],
    "warfarin": ["blood clots", "atrial fibrillation", "deep vein thrombosis"],
    "furosemide": ["edema", "heart failure", "fluid retention"],
    "metoprolol": ["hypertension", "high blood pressure", "heart failure"],
    "simvastatin": ["high cholesterol", "hypercholesterolemia"],
    "prednisone": ["inflammation", "asthma", "arthritis", "allergic reactions"],
}


def is_known_indication(drug_name, disease_name):
    drug_name = drug_name.strip().lower()
    disease_name = disease_name.strip().lower()
    indications = KNOWN_INDICATIONS.get(drug_name, [])
    return any(disease_name in ind or ind in disease_name for ind in indications)


def get_known_treating_drugs(disease_name):
    disease_name = disease_name.strip().lower()
    matches = []
    for drug, conditions in KNOWN_INDICATIONS.items():
        if any(disease_name in c or c in disease_name for c in conditions):
            matches.append(drug)
    return matches
