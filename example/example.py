import pandas as pd
import numpy as np
from data_processing.data_processing import preprocess_room_names, enhanced_room_matching, main
from models.bert_xgb import BertXGBoostRoomMatcher
from models.evaluate_model import evaluate_model
from sentence_transformers import SentenceTransformer
import time

# ===== STEP 1: Load and Preprocess Data =====
print("Loading data...")
reference_rooms = pd.read_csv('./data_source/raw/reference_rooms.csv')
supplier_rooms = pd.read_csv('./data_source/raw/updated_core_rooms.csv')

# For demonstration, use a smaller subset
reference_subset = reference_rooms.head(1000)
supplier_subset = supplier_rooms.head(1000)

# Preprocess room names
print("Preprocessing room names...")
reference_subset, supplier_subset = main(reference_subset, supplier_subset)

# ===== STEP 2: Train the Model =====
print("\n===== TRAINING THE MODEL =====")
# Initialize the model 
matcher = BertXGBoostRoomMatcher(
    bert_model_name='sentence-transformers/all-MiniLM-L6-v2', #  perform better than 'prajjwal1/bert-tiny'
    model_path='./room_matcher_model.joblib',
    max_length=64,
    batch_size=16
)

# Create some labeled data for training
# In a real scenario, you would have actual labeled data
print("Creating labeled training data...")
# For demonstration, we'll create synthetic labeled data
# In practice, you would use real labeled matches

# training
matcher.train(reference_subset, supplier_subset)
# Fine-tune with fewer iterations for demonstration
# put labeled data here
print("Fine-tuning hyperparameters...")
start_time = time.time()
find_label_data1= pd.read_csv('./data_source/labelled/labeled_data.csv')
find_label_data1['label'] = find_label_data1['text_similarity'].apply(lambda x: 1 if x >=0.9 else 0)
matcher.tune_hyperparameters(find_label_data1, new_model_path='./room_matcher_tuned.joblib', n_iter=5)
print(f"Fine-tuning completed in {time.time() - start_time:.2f} seconds")

# ===== STEP 4: Evaluate the Model =====
print("\n===== EVALUATING THE MODEL =====")
# Evaluate on validation data
print("Evaluating model performance...")
test_data = pd.read_csv('./data_source/labelled/matches.csv')
test_data['label'] = test_data['text_similarity'].apply(lambda x: 1 if x >0.9 else 0)
metrics, eval_results = evaluate_model(matcher, test_data, threshold=0.5)

# Print summary of evaluation results
print("\nEvaluation Summary:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
if metrics['roc_auc']:
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
if metrics['pr_auc']:
    print(f"PR AUC: {metrics['pr_auc']:.4f}")

# ===== STEP 5: Make Predictions on New Data =====
print("\n===== MAKING PREDICTIONS =====")
# Use the model to predict matches on new data
print("Predicting matches on test data...")
# For demonstration, we'll use a small subset of the data
test_reference = reference_rooms.iloc[1000:1100]
test_supplier = supplier_rooms.iloc[1000:1100]

# Preprocess test data
test_reference, test_supplier = main(test_reference, test_supplier)

# Make predictions
predicted_matches = matcher.predict(test_reference, test_supplier, threshold=0.7)

# Display sample predictions
print(f"\nFound {len(predicted_matches)} potential matches")
if len(predicted_matches) > 0:
    print("\nSample predictions:")
    for i, match in predicted_matches.head(5).iterrows():
        print(f"Reference: {match['reference_room_name']}")
        print(f"Supplier: {match['supplier_room_name']}")
        print(f"Match probability: {match['match_probability']:.4f}")
        print("---")

print("\nRoom matching workflow complete!")