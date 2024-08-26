from sklearn.metrics import precision_score, recall_score, f1_score
import spacy
# Entity Recognition
nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Example ground truth and predictions
true_entities = [("Norway", "GPE"), ("electric vehicles", "PRODUCT")]
predicted_entities = extract_entities("Norway offers incentives for electric vehicles.")

def evaluate_ner(true_entities, predicted_entities):
    # Extract just the entity texts for comparison
    true_labels = [entity for entity, _ in true_entities]
    predicted_labels = [entity for entity, _ in predicted_entities]
    
    # Convert to sets to calculate true positives, false positives, and false negatives
    true_set = set(true_labels)
    print(true_set)
    predicted_set = set(predicted_labels)
    print(predicted_set)
    
    # Calculate Precision, Recall, F1
    true_positives = len(true_set & predicted_set)
    false_positives = len(predicted_set - true_set)
    false_negatives = len(true_set - predicted_set)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Example ground truth and predictions
true_entities = [("Norway", "GPE"), ("electric vehicles", "PRODUCT")]
predicted_entities = extract_entities("Norway offers incentives for electric vehicles.")

# Evaluate based on entity texts only
precision, recall, f1 = evaluate_ner(true_entities, predicted_entities)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

