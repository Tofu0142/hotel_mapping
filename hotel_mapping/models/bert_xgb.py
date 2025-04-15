import pandas as pd
import numpy as np
import time
import re
import os
import torch
from transformers import AutoTokenizer, AutoModel
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sentence_transformers import SentenceTransformer

class BertXGBoostRoomMatcher:
    """
    BERT + XGBoost Room Matcher - CPU Optimized Version
    """
    
    def __init__(self, 
                bert_model_name='sentence-transformers/all-MiniLM-L6-v2',  # 4-layer BERT, 16x smaller than base
                model_path='./bert_xgb_room_matcher.joblib',
                max_length=64,
                batch_size=32):
        
        self.model_path = model_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.bert_model_name = bert_model_name
        self.xgb_model = None
        
        # Load lightweight BERT model
        print(f"Loading BERT model: {bert_model_name}")
        try:
            # try use  SentenceTransformer
            self.sentence_transformer = SentenceTransformer(bert_model_name)
            print("Successfully loaded SentenceTransformer model")
            
            # for compatibility, we still keep tokenizer and bert_model
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
            
            # set evaluation mode
            self.bert_model.eval()
            
            # check device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.bert_model.to(self.device)
            print(f"Using device: {self.device}")
        except Exception as e:
            print(f"Error loading SentenceTransformer: {e}")
            # 回退到原始加载方式
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_model = AutoModel.from_pretrained(bert_model_name)
            self.bert_model.eval()
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.bert_model.to(self.device)
            print(f"Using device: {self.device}")
    
    def extract_features(self, room_name):
       
        features = {
            'room_type': None,
            'bed_type': None,
            'view_type': None,  
            'has_view': False,
            'no_view': False,   
            'capacity': None,
            'people_capacity': None,  
            'bed_capacity': None     
        }
        
        text_lower = str(room_name).lower()
        
        # Room type
        room_types = [ 'suite', 'loft',
                     'deluxe', 'luxury', 'standard', 'superior', 'family', 'executive','home','house','studio']
        for rt in room_types:
            if rt in text_lower:
                features['room_type'] = rt
                break
                
       # Bed type
        bed_types = {'king': 'king bed', 'queen': 'queen bed', 'double': 'double bed',
                    'twin': 'twin bed', 'single': 'single bed'}
        for bt, value in bed_types.items():
            if bt in text_lower:
                features['bed_type'] = value
                break
                
      # Enhanced view detection
        view_types = {
            'garden': 'garden view',
            'sea': 'sea view',
            'ocean': 'ocean view',
            'mountain': 'mountain view',
            'pool': 'pool view',
            'lagoon': 'lagoon view',
            'city': 'city view'
        }
        
         # Detect if explicitly labeled as no view
        if 'no view' in text_lower or '(no view)' in text_lower:
            features['no_view'] = True
        else:
            # Detect specific view type
            for key, value in view_types.items():
                if key in text_lower:
                    features['view_type'] = value
                    features['has_view'] = True
                    break
        
       # Capacity
        capacity_people = r'(\d+)\s*(person|people|adult|pax)'
        capacity_peopele_match = re.search(capacity_people, text_lower)
        if capacity_peopele_match:
            features['people_capacity'] = int(capacity_peopele_match.group(1))
        
        capacity_bed = r'(\d+)\s*(bed|beds)'
        capacity_bed_match = re.search(capacity_bed, text_lower)
        if capacity_bed_match:
            features['bed_capacity'] = int(capacity_bed_match.group(1))
            
        return features
    
    def get_bert_embeddings(self, texts, use_cache=True, cache_dir='./embedding_cache'):
        """Get BERT embeddings, with caching support"""
        # Create cache directory
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            
           # Create cache key
            import hashlib
            texts_hash = hashlib.md5(str(texts).encode()).hexdigest()
            cache_key = f"{self.bert_model_name.replace('/', '_')}_{texts_hash}"
            cache_path = os.path.join(cache_dir, f"{cache_key}.npy")
            
            # Check cache
            if os.path.exists(cache_path):
                return np.load(cache_path)
        
        if hasattr(self, 'sentence_transformer'):
            print("Using SentenceTransformer for encoding")
            embeddings = self.sentence_transformer.encode(texts, batch_size=self.batch_size)
            
            if use_cache:
                np.save(cache_path, embeddings)
                
            return embeddings
        # Process texts in batches
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT encoding"):
            batch_texts = texts[i:i+self.batch_size]
            
            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
           
            with torch.no_grad():
                model_output = self.bert_model(**encoded_input)
                
            # Use [CLS] token output as sentence embedding
            sentence_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(sentence_embeddings)
        
         # Combine embeddings from all batches
        embeddings = np.vstack(all_embeddings)
        
        if use_cache:
            np.save(cache_path, embeddings)
            
        return embeddings
    
    def create_pair_features(self, ref_embedding, sup_embedding, ref_features, sup_features, ref_name, sup_name):
        """Create feature vector for a pair of rooms"""
        
        text_sim = np.dot(ref_embedding, sup_embedding) / (
            np.linalg.norm(ref_embedding) * np.linalg.norm(sup_embedding)
        )
        
        # Add more text features
        # 1. Jaccard similarity
        ref_tokens = set(str(ref_name).lower().split())
        sup_tokens = set(str(sup_name).lower().split())
        
        if len(ref_tokens) == 0 or len(sup_tokens) == 0:
            jaccard_sim = 0
        else:
            intersection = len(ref_tokens.intersection(sup_tokens))
            union = len(ref_tokens.union(sup_tokens))
            jaccard_sim = intersection / union if union > 0 else 0
        
        # 2. Token overlap
        token_overlap = len(ref_tokens.intersection(sup_tokens))
        
        # 3. Complete containment check
        ref_contains_sup = 1 if all(token in ref_tokens for token in sup_tokens) else 0
        sup_contains_ref = 1 if all(token in sup_tokens for token in ref_tokens) else 0
        
         # 4. Number matching (beds, area, etc.)
        import re
        ref_numbers = set(re.findall(r'\d+', str(ref_name)))
        sup_numbers = set(re.findall(r'\d+', str(sup_name)))
        
        number_match = 0
        if ref_numbers and sup_numbers:
            number_match = len(ref_numbers.intersection(sup_numbers)) / max(len(ref_numbers), len(sup_numbers))
        
        # 5. Levenshtein edit distance
        import Levenshtein
        lev_distance = Levenshtein.distance(str(ref_name).lower(), str(sup_name).lower())
         # Normalize edit distance (0-1 range, 1 means identical)
        max_len = max(len(str(ref_name)), len(str(sup_name)))
        normalized_lev_sim = 1 - (lev_distance / max_len) if max_len > 0 else 0
       
        room_type_match = int(ref_features['room_type'] == sup_features['room_type'] 
                             and ref_features['room_type'] is not None)
        bed_type_match = int(ref_features['bed_type'] == sup_features['bed_type']
                            and ref_features['bed_type'] is not None)
        
        # Feature matching
        view_match = 0
        view_conflict = 0  
        
        # Enhanced view matching logic
        if (ref_features['has_view'] and sup_features['no_view']) or \
           (sup_features['has_view'] and ref_features['no_view']):
            view_conflict = 1
        # If both have view types and they match
        elif ref_features['view_type'] and sup_features['view_type'] and \
             ref_features['view_type'] == sup_features['view_type']:
            view_match = 1
        # If neither has a view or neither specifies a view
        elif (not ref_features['has_view'] and not sup_features['has_view']) and \
             (not ref_features['no_view'] and not sup_features['no_view']):
            view_match = 1
        
        
        capacity_people_match = 0
        capacity_bed_match = 0
        if ref_features['people_capacity'] is not None and sup_features['people_capacity'] is not None:
            if ref_features['people_capacity'] == sup_features['people_capacity']:
                capacity_people_match = 1
        if ref_features['bed_capacity'] is not None and sup_features['bed_capacity'] is not None:
            if ref_features['bed_capacity'] == sup_features['bed_capacity']:
                capacity_bed_match = 1
        
        combined_features = np.concatenate([
            [text_sim, jaccard_sim, token_overlap, ref_contains_sup, sup_contains_ref, 
             number_match, normalized_lev_sim, room_type_match, bed_type_match, 
             view_match, view_conflict, capacity_people_match, capacity_bed_match],
            ref_embedding,
            sup_embedding
        ])
        
        return combined_features
    
    def train(self, reference_rooms, supplier_rooms, labeled_matches=None):
        """Train the model"""
        print("Training BERT+XGBoost room matching model...")
        start_time = time.time()
        
        # Get all room names
        ref_names = reference_rooms['processed_name'].tolist()
        sup_names = supplier_rooms['processed_name'].tolist()
        
        # Get BERT embeddings
        print("Getting BERT embeddings...")
        ref_embeddings = self.get_bert_embeddings(ref_names)
        sup_embeddings = self.get_bert_embeddings(sup_names)
        
        # Extract structured features
        print("Extracting structured features...")
        ref_features = [self.extract_features(name) for name in ref_names]
        sup_features = [self.extract_features(name) for name in sup_names]
        
        # Prepare training data
        X = []
        y = []
        
        # If there are labeled matches
        if labeled_matches is not None and len(labeled_matches) > 0:
            print("Using labeled match data...")
            
            # Positive samples
            for _, row in labeled_matches.iterrows():
                ref_idx = reference_rooms[reference_rooms['room_id'] == row['reference_room_id']].index[0]
                sup_idx = supplier_rooms[supplier_rooms['supplier_room_id'] == row['supplier_room_id']].index[0]
                
                X.append(self.create_pair_features(
                    ref_embeddings[ref_idx], sup_embeddings[sup_idx],
                    ref_features[ref_idx], sup_features[sup_idx],
                    ref_names[ref_idx], sup_names[sup_idx]
                ))
                y.append(1)
                
                # Generate several negative samples for each positive sample
                for _ in range(3):
                    neg_j = np.random.choice([j for j in range(len(sup_names)) if j != sup_idx])
                    X.append(self.create_pair_features(
                        ref_embeddings[ref_idx], sup_embeddings[neg_j],
                        ref_features[ref_idx], sup_features[neg_j],
                        ref_names[ref_idx], sup_names[neg_j]
                    ))
                    y.append(0)
        else:
            print("Generating synthetic training data...")
            # Generate synthetic training data
            for i in range(len(ref_names)):
                # Find the most similar room as a positive sample
                similarities = np.array([
                    np.dot(ref_embeddings[i], sup_embeddings[j]) / (
                        np.linalg.norm(ref_embeddings[i]) * np.linalg.norm(sup_embeddings[j])
                    ) for j in range(len(sup_names))
                ])
                
                best_j = np.argmax(similarities)
                best_similarity = similarities[best_j]
                
                # Only add as positive sample if similarity is greater than 0.99
                if best_similarity > 0.99:
                    # Add positive sample
                    X.append(self.create_pair_features(
                        ref_embeddings[i], sup_embeddings[best_j],
                        ref_features[i], sup_features[best_j],
                        ref_names[i], sup_names[best_j]
                    ))
                    y.append(1)
                
                # Add negative samples
                for _ in range(10):
                    # Select rooms with lower similarity as negative samples
                    neg_candidates = np.argsort(similarities)[:len(similarities)//2]
                    neg_j = np.random.choice(neg_candidates)
                    
                    X.append(self.create_pair_features(
                        ref_embeddings[i], sup_embeddings[neg_j],
                        ref_features[i], sup_features[neg_j],
                        ref_names[i], sup_names[neg_j]
                    ))
                    y.append(0)
        
        # Analyze feature relationships and remove useless features
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Define feature names
        feature_names = ['text_sim', 'jaccard_sim', 'token_overlap', 'ref_contains_sup', 'sup_contains_ref', 
                         'number_match', 'normalized_lev_sim', 'room_type_match', 'bed_type_match', 
                         'view_match', 'view_conflict', 'capacity_people_match', 'capacity_bed_match'] + \
                        [f'ref_emb_{i}' for i in range(ref_embeddings.shape[1])] + \
                        [f'sup_emb_{i}' for i in range(sup_embeddings.shape[1])]
        
        # Analyze correlation of the first 13 manual features
        print("\nAnalyzing feature correlations...")
        try:
            # Calculate correlation between features
            manual_features = X_array[:, :13]  # Only take manual features
            corr_matrix = np.corrcoef(manual_features.T)
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                        xticklabels=feature_names[:13], yticklabels=feature_names[:13])
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig('feature_correlation.png')
            print("Feature correlation heatmap saved as 'feature_correlation.png'")
            
            # Calculate mutual information between each feature and target variable
            mi_scores = mutual_info_classif(manual_features, y_array)
            mi_df = pd.DataFrame({'Feature': feature_names[:13], 'Mutual Information Score': mi_scores})
            mi_df = mi_df.sort_values('Mutual Information Score', ascending=False)
            
            print("\nMutual information scores between features and target variable:")
            print(mi_df)
            
            # Plot mutual information bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Mutual Information Score', y='Feature', data=mi_df)
            plt.title('Mutual Information Scores between Features and Target Variable')
            plt.tight_layout()
            plt.savefig('feature_importance_mi.png')
            print("Feature importance graph saved as 'feature_importance_mi.png'")
            
            # Identify highly correlated feature pairs
            print("\nHighly correlated feature pairs (|correlation coefficient| > 0.75):")
            high_corr_pairs = []
            for i in range(len(feature_names[:13])):
                for j in range(i+1, len(feature_names[:13])):
                    if abs(corr_matrix[i, j]) > 0.75:
                        print(f"{feature_names[i]} and {feature_names[j]}: {corr_matrix[i, j]:.3f}")
                        high_corr_pairs.append((i, j, corr_matrix[i, j], mi_scores[i], mi_scores[j]))
            
            # Identify features with low mutual information
            low_mi_threshold = 0.01
            low_mi_features = mi_df[mi_df['Mutual Information Score'] < low_mi_threshold]
            if not low_mi_features.empty:
                print(f"\nFeatures with low mutual information (MI < {low_mi_threshold}):")
                print(low_mi_features)
            
            # Create feature mask (True means keep the feature)
            feature_mask = np.ones(X_array.shape[1], dtype=bool)
            
            # Mark manual features with low mutual information as False
            for i in range(13):
                if mi_scores[i] < low_mi_threshold:
                    feature_mask[i] = False
                    print(f"Removing feature with low mutual information: {feature_names[i]}")
            
            # Handle highly correlated feature pairs - keep the one with higher mutual information
            features_to_remove = set()
            for i, j, corr, mi_i, mi_j in high_corr_pairs:
                # If neither feature has been marked for removal
                if feature_mask[i] and feature_mask[j]:
                    # Remove the feature with lower mutual information
                    if mi_i < mi_j:
                        features_to_remove.add(i)
                    else:
                        features_to_remove.add(j)
            
            # Apply removal of highly correlated features
            for i in features_to_remove:
                if feature_mask[i]:  # Ensure it hasn't been removed by previous rules
                    feature_mask[i] = False
                    print(f"Removing highly correlated feature: {feature_names[i]}")
            
            # Apply feature mask
            X_filtered = X_array[:, feature_mask]
            print(f"Feature count reduced from {X_array.shape[1]} to {X_filtered.shape[1]}")
            
            # Update feature names list
            filtered_feature_names = [feature_names[i] for i in range(len(feature_names)) if feature_mask[i]]
            print("\nRetained features:")
            for name in filtered_feature_names[:13]:  # Only print manual features
                print(f"- {name}")
            
        except ImportError as e:
            print(f"Cannot perform feature analysis: {e}")
            print("Please install required libraries: pip install matplotlib seaborn")
            X_filtered = X_array
            filtered_feature_names = feature_names
            feature_mask = np.ones(X_array.shape[1], dtype=bool)
        
        # Split training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_filtered, y_array, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        print(f"\nTraining XGBoost model with {len(X_train)} samples...")
        
        # Set XGBoost parameters - optimized for CPU
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',  # 'hist' is faster than 'exact'
            'nthread': -1  # Use all CPU cores
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        self.xgb_model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate model
        y_pred = (self.xgb_model.predict(dval) >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        
        print(f"Validation set performance: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
        
        # Save model and feature extractor
        model_data = {
            'xgb_model': self.xgb_model,
            'bert_model_name': self.bert_model_name,
            'feature_mask': feature_mask  # Save feature mask for prediction
        }
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to: {self.model_path}")
        
        # Feature importance
        importance_scores = self.xgb_model.get_score(importance_type='gain')
        print("\nTop 10 Feature Importance:")
        
        # Map feature indices to filtered feature names
        importance_dict = {}
        for fname, score in importance_scores.items():
            feature_idx = int(fname.replace('f', ''))
            if feature_idx < len(filtered_feature_names):
                importance_dict[filtered_feature_names[feature_idx]] = score
            else:
                importance_dict[f"Feature_{feature_idx}"] = score
        
        for i, (fname, score) in enumerate(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"  {fname}: {score}")
        
        # Plot feature importance
        try:
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            })
            importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('XGBoost Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig('xgboost_feature_importance.png')
            print("XGBoost feature importance graph saved as 'xgboost_feature_importance.png'")
        except Exception as e:
            print(f"Cannot plot feature importance: {e}")
        
        end_time = time.time()
        print(f"Training complete! Time taken: {end_time - start_time:.2f} seconds")
        
        # Save feature mask for prediction
        self.feature_mask = feature_mask
        
        return self
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.xgb_model = model_data['xgb_model']
            
            # Load feature mask
            if 'feature_mask' in model_data:
                self.feature_mask = model_data['feature_mask']
            else:
                # Compatibility with old models
                print("Warning: Loaded model has no feature mask information, will use all features")
                self.feature_mask = None
            
            # Check if BERT model matches
            if model_data['bert_model_name'] != self.bert_model_name:
                print(f"Warning: Loaded model used a different BERT model: {model_data['bert_model_name']}")
                print(f"Current model is: {self.bert_model_name}")
            
            # Load BERT model
            self.bert_model = SentenceTransformer(self.bert_model_name)
            
            return True
        return False
    
    def encode(self, texts):
        """
        
        """
        if self.bert_model is None:
            raise ValueError("BERT model not loaded. Call load_model() first.")
        
        return self.get_bert_embeddings(texts)
    
    def predict(self, reference_rooms, supplier_rooms, threshold=0.5):
        """Predict room matches"""
        print("Predicting room matches...")
        start_time = time.time()
        
        # Try to load model
        if self.xgb_model is None:
            if not self.load_model():
                raise ValueError("Model not trained, please call train() method first")
        
        # Get all room names
        ref_names = reference_rooms['processed_name'].tolist()
        sup_names = supplier_rooms['processed_name'].tolist()
        
        # Get BERT embeddings
        print("Getting BERT embeddings...")
        ref_embeddings = self.get_bert_embeddings(ref_names)
        sup_embeddings = self.get_bert_embeddings(sup_names)
        
        # Extract structured features
        print("Extracting structured features...")
        ref_features = [self.extract_features(name) for name in ref_names]
        sup_features = [self.extract_features(name) for name in sup_names]
        
        # Calculate features for each pair and predict
        matches = []
        
        print("Predicting matches...")
        for i in tqdm(range(len(reference_rooms)), desc="Predicting matches"):
            best_score = 0
            best_idx = -1
            
            # Create feature pairs for reference room with all supplier rooms
            pair_features = []
            for j in range(len(supplier_rooms)):
                features = self.create_pair_features(
                    ref_embeddings[i], sup_embeddings[j],
                    ref_features[i], sup_features[j],
                    ref_names[i], sup_names[j]
                )
                
                # Apply feature mask (if available)
                if self.feature_mask is not None:
                    features = features[self.feature_mask]
                
                pair_features.append(features)
            
            # Batch prediction
            if pair_features:
                dmatrix = xgb.DMatrix(np.array(pair_features))
                probs = self.xgb_model.predict(dmatrix)
                
                # Find best match
                best_idx = np.argmax(probs)
                best_score = probs[best_idx]
            
            # If best match exceeds threshold, add to results
            if best_score >= threshold and best_idx >= 0:
                matches.append({
                    'reference_hotel_id': reference_rooms.iloc[i]['hotel_id'],
                    'reference_room_id': reference_rooms.iloc[i]['room_id'],
                    'reference_room_name': reference_rooms.iloc[i]['room_name'],
                    'supplier_room_id': supplier_rooms.iloc[best_idx]['supplier_room_id'],
                    'supplier_room_name': supplier_rooms.iloc[best_idx]['supplier_room_name'],
                    'match_probability': float(best_score)
                })
        
        matches_df = pd.DataFrame(matches)
        
        end_time = time.time()
        print(f"Matching complete! Time taken: {end_time - start_time:.2f} seconds")
        print(f"Found {len(matches_df)} matches")
        
        return matches_df

    def tune_hyperparameters(self, labeled_data, new_model_path=None, n_iter=20):
        """Tune XGBoost hyperparameters using cross-validation"""
        print("Starting XGBoost hyperparameter tuning...")
        start_time = time.time()
        
        if new_model_path is None:
            new_model_path = self.model_path

        X = []
        y = []
        
        print("Preparing training data...")
        for i, row in tqdm(labeled_data.iterrows(), total=len(labeled_data), desc="处理标记数据"):
            ref_name = row['reference_processed_name']
            sup_name = row['supplier_processed_name']
            true_label = row['label']

            ref_embedding = self.get_bert_embeddings([ref_name])[0]
            sup_embedding = self.get_bert_embeddings([sup_name])[0]

            ref_feature = self.extract_features(ref_name)
            sup_feature = self.extract_features(sup_name)

            features = self.create_pair_features(
                ref_embedding, sup_embedding,
                ref_feature, sup_feature,
                ref_name, sup_name
            )
            
       
            X.append(features)
            y.append(true_label)
            
       
            if true_label == 1:
            
                neg_indices = [j for j in range(len(labeled_data)) if j != i]
                if len(neg_indices) >= 3:  # 确保有足够的样本
                    neg_samples = np.random.choice(neg_indices, 3, replace=False)
                    
                    for neg_idx in neg_samples:
                        neg_row = labeled_data.iloc[neg_idx]
                        neg_sup_name = neg_row['supplier_processed_name']
                        
                     
                        if neg_sup_name != sup_name:
                           
                            neg_sup_embedding = self.get_bert_embeddings([neg_sup_name])[0]

                            neg_sup_feature = self.extract_features(neg_sup_name)

                            neg_features = self.create_pair_features(
                                ref_embedding, neg_sup_embedding,
                                ref_feature, neg_sup_feature,
                                ref_name, neg_sup_name
                            )

                            X.append(neg_features)
                            y.append(0)  

        X_array = np.array(X)
        y_array = np.array(y)
        
        print(f"Preparation complete, total {len(X_array)} samples, with {np.sum(y_array)} positive samples")
        print(f"Feature count: {X_array.shape[1]}")
        
        ref_emb_dim = ref_embedding.shape[0]
        feature_names = ['text_sim', 'jaccard_sim', 'token_overlap', 'ref_contains_sup', 'sup_contains_ref', 
                         'number_match', 'normalized_lev_sim', 'room_type_match', 'bed_type_match', 
                         'view_match', 'view_conflict', 'capacity_people_match', 'capacity_bed_match'] + \
                        [f'ref_emb_{i}' for i in range(ref_emb_dim)] + \
                        [f'sup_emb_{i}' for i in range(ref_emb_dim)]
        
        # Feature selection - similar to train method
        print("\nPerforming feature selection...")
        try:
            
            manual_features = X_array[:, :13]
            mi_scores = mutual_info_classif(manual_features, y_array)
            mi_df = pd.DataFrame({'Feature': feature_names[:13], 'Mutual Information Score': mi_scores})
            mi_df = mi_df.sort_values('Mutual Information Score', ascending=False)
            
            print("\nMutual information scores between features and target variable:")
            print(mi_df)
            
            # Create feature mask (True means keep the feature)
            feature_mask = np.ones(X_array.shape[1], dtype=bool)
            
            # Mark manual features with low mutual information as False
            low_mi_threshold = 0.01
            for i in range(13):
                if mi_scores[i] < low_mi_threshold:
                    feature_mask[i] = False
                    print(f"Removing feature with low mutual information: {feature_names[i]}")
            
            # Apply feature mask
            X_filtered = X_array[:, feature_mask]
            print(f"Feature count reduced from {X_array.shape[1]} to {X_filtered.shape[1]}")
            
            # Update feature names list
            filtered_feature_names = [feature_names[i] for i in range(len(feature_names)) if i < 13 and feature_mask[i]]
            print("\nRetained manual features:")
            for name in filtered_feature_names:
                print(f"- {name}")
            
            # Save feature mask for later use
            self.feature_mask = feature_mask
            
        except ImportError as e:
            print(f"Cannot perform feature analysis: {e}")
            print("Please install required libraries: pip install scikit-learn")
            X_filtered = X_array
            self.feature_mask = None
        except Exception as e:
            print(f"Error during feature selection: {e}")
            X_filtered = X_array
            self.feature_mask = None
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_filtered, y_array, test_size=0.2, random_state=42)
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 4, 5, 6, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        # Use random search instead of grid search, more efficient
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.metrics import make_scorer, f1_score
        
        # Create XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            tree_method='hist',
            nthread=-1
        )
        
        # Create F1 scorer
        f1_scorer = make_scorer(f1_score)
        
        # Random search
        print(f"Starting random search, trying {n_iter} parameter combinations...")
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            n_iter=n_iter,  # Try n_iter parameter combinations
            scoring=f1_scorer,
            cv=3,  # 3-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1  # Use all CPUs
        )
        
        # Execute search
        random_search.fit(X_train, y_train)
        
        # Get best parameters
        best_params = random_search.best_params_
        print(f"Best parameters: {best_params}")
        print(f"Best F1 score: {random_search.best_score_:.4f}")
        
        # Train final model with best parameters
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': best_params['max_depth'],
            'eta': best_params['learning_rate'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'min_child_weight': best_params['min_child_weight'],
            'gamma': best_params['gamma'],
            'tree_method': 'hist',
            'nthread': -1
        }
        
        print("Training final model with best parameters...")
        self.xgb_model = xgb.train(
            final_params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Evaluate final model
        y_pred = (self.xgb_model.predict(dval) >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        
        print(f"Final model performance: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1:.4f}")
        
        # Save model and feature mask
        model_data = {
            'xgb_model': self.xgb_model,
            'bert_model_name': self.bert_model_name,
            'feature_mask': self.feature_mask,
            'best_params': best_params
        }
        joblib.dump(model_data, new_model_path)
        print(f"Optimized model saved to: {new_model_path}")
        
        # Feature importance
        try:
            importance_scores = self.xgb_model.get_score(importance_type='gain')
            print("\nTop 10 Feature Importance:")
            
            # Map feature indices to feature names
            importance_dict = {}
            for fname, score in importance_scores.items():
                feature_idx = int(fname.replace('f', ''))
                if feature_idx < len(filtered_feature_names):
                    importance_dict[filtered_feature_names[feature_idx]] = score
                else:
                    importance_dict[f"Feature_{feature_idx}"] = score
            
            for i, (fname, score) in enumerate(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]):
                print(f"  {fname}: {score}")
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            importance_df = pd.DataFrame({
                'Feature': list(importance_dict.keys()),
                'Importance': list(importance_dict.values())
            })
            importance_df = importance_df.sort_values('Importance', ascending=False).head(20)
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('XGBoost Feature Importance (Top 20)')
            plt.tight_layout()
            plt.savefig('xgboost_feature_importance.png')
            print("XGBoost feature importance graph saved as 'xgboost_feature_importance.png'")
        except Exception as e:
            print(f"Cannot plot feature importance: {e}")
        
        end_time = time.time()
        print(f"Hyperparameter tuning complete! Time taken: {end_time - start_time:.2f} seconds")
        
        return self

# # 使用示例
# if __name__ == "__main__":
#     # 示例数据
#     reference_rooms = pd.DataFrame({
#         'hotel_id': [1, 1, 1, 1, 1],
#         'room_id': [101, 102, 103, 104, 105],
#         'room_name': ['Deluxe Double Room', 'Superior Twin Room', 'Family Suite', 
#                       'Standard Single Room', 'Executive King Room with Sea View']
#     })
    
#     supplier_rooms = pd.DataFrame({
#         'supplier_id': ['A', 'A', 'A', 'A', 'A'],
#         'supplier_room_id': ['R1', 'R2', 'R3', 'R4', 'R5'],
#         'supplier_room_name': ['Superior Double Room', 'Deluxe Twin Room', 'Family Room', 
#                               'Standard Room', 'King Executive with Ocean View']
#     })
    
#     # 创建并训练匹配器
#     matcher = BertXGBoostRoomMatcher(bert_model_name='prajjwal1/bert-tiny')
#     matcher.train(reference_rooms, supplier_rooms)
    
#     # 预测匹配
#     matches = matcher.predict(reference_rooms, supplier_rooms, threshold=0.6)
#     print(matches)
