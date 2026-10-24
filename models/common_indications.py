"""Small, manually curated reference of well-established drug-indication pairs.

The literature models learn TREATS only from BioRED negative-correlation annotations,
which cover few drugs. This reference guarantees that very common first-line treatments
are never reported as harmful for the condition they treat.

Matching is EXACT after normalisation (lower case, collapsed whitespace) against the
listed indication names and their synonyms - e.g. "pain" does not match "chest pain".
"""
from text_utils import normalize_name

KNOWN_INDICATIONS = {
    "metformin": ["type 2 diabetes", "type 2 diabetes mellitus", "diabetes", "diabetes mellitus"],
    "insulin": ["type 1 diabetes", "type 2 diabetes", "diabetes", "diabetes mellitus"],
    "atorvastatin": ["high cholesterol", "hypercholesterolemia", "hyperlipidemia", "coronary heart disease"],
    "simvastatin": ["high cholesterol", "hypercholesterolemia", "hyperlipidemia"],
    "lisinopril": ["high blood pressure", "hypertension", "heart failure"],
    "amlodipine": ["high blood pressure", "hypertension", "angina", "angina pectoris"],
    "metoprolol": ["high blood pressure", "hypertension", "heart failure", "angina"],
    "clonidine": ["high blood pressure", "hypertension"],
    "furosemide": ["edema", "oedema", "heart failure", "fluid retention"],
    "warfarin": ["blood clots", "atrial fibrillation", "deep vein thrombosis", "pulmonary embolism"],
    "albuterol": ["asthma", "copd", "chronic obstructive pulmonary disease", "bronchospasm"],
    "salbutamol": ["asthma", "copd", "chronic obstructive pulmonary disease", "bronchospasm"],
    "prednisone": ["inflammation", "asthma", "rheumatoid arthritis", "allergic reactions"],
    "omeprazole": ["gerd", "gastroesophageal reflux disease", "peptic ulcer", "stomach ulcer"],
    "levothyroxine": ["hypothyroidism"],
    "sertraline": ["depression", "major depressive disorder", "anxiety", "anxiety disorder"],
    "amoxicillin": ["pneumonia", "otitis media", "ear infection", "bacterial infection"],
    "aspirin": ["pain", "headache", "fever", "inflammation"],
    "ibuprofen": ["pain", "headache", "fever", "inflammation"],
    "paracetamol": ["pain", "headache", "fever"],
    "acetaminophen": ["pain", "headache", "fever"],
}

_INDEX = {drug: {normalize_name(c) for c in conds} for drug, conds in KNOWN_INDICATIONS.items()}


def is_known_indication(drug_name, disease_name):
    return normalize_name(disease_name) in _INDEX.get(normalize_name(drug_name), set())


def get_known_treating_drugs(disease_name):
    disease = normalize_name(disease_name)
    return sorted(drug for drug, conds in _INDEX.items() if disease in conds)
