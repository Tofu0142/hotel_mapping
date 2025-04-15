import pandas as pd
import numpy as np
import re
import string
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_room_names(room_names, extract_features=True):
    """
    Comprehensive preprocessing of hotel room names without using NLTK
    
    Parameters:
    room_names (list): List of room names
    extract_features (bool): Whether to extract structured features
    
    Returns:
    processed_names (list): Preprocessed room names
    features (dict, optional): Extracted structured features
    """
    processed_names = []
    features = {} if extract_features else None
    
    # Custom English stopwords list
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                 'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
                 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                 'to', 'of', 'in', 'on', 'is', 'are', 'was', 'were'}
    
    # Room type standardization mapping
    room_type_mapping = {
        'double': 'double', 'twin': 'twin', 'single': 'single',
        'queen': 'queen', 'king': 'king', 'suite': 'suite',
        'apartment': 'apartment', 'studio': 'studio', 'deluxe': 'deluxe',
        'standard': 'standard', 'superior': 'superior', 'executive': 'executive',
        'family': 'family', 'junior': 'junior', 'presidential': 'presidential',
        'cottage': 'cottage', 'villa': 'villa', 'bungalow': 'bungalow',
        'dormitory': 'dormitory', 'dorm': 'dormitory'
    }
    
    # Bed type standardization mapping
    bed_type_mapping = {
        'queen': 'queen bed', 'king': 'king bed', 
        'single': 'single bed', 'double': 'double bed',
        'twin': 'twin bed', 'sofa': 'sofa bed'
    }
    
    # Capacity/occupancy patterns
    capacity_patterns = [
        r'(\d+)\s*person', r'(\d+)\s*people', r'(\d+)\s*bed',
        r'for\s*(\d+)', r'(\d+)\s*adult', r'(\d+)\s*pax'
    ]
    
    # Area patterns
    area_patterns = [
        r'(\d+)\s*m²', r'(\d+)\s*sqm', r'(\d+)\s*square\s*meter'
    ]
    
    # Simple lemmatization mapping
    lemma_mapping = {
        'rooms': 'room', 'beds': 'bed', 'bedrooms': 'bedroom',
        'apartments': 'apartment', 'suites': 'suite', 'villas': 'villa',
        'cottages': 'cottage', 'people': 'person', 'adults': 'adult',
        'children': 'child', 'views': 'view', 'windows': 'window'
    }
    
    if extract_features:
        features = {
            'room_type': [],
            'bed_type': [],
            'capacity': [],
            'area': [],
            'has_view': [],
            'has_balcony': [],
            'has_terrace': [],
            'is_smoking': [],
            'is_non_smoking': [],
            'has_breakfast': []
        }
    
    for i, name in enumerate(room_names):
        if name is None or pd.isna(name):
            processed_names.append("")
            if extract_features:
                for key in features:
                    features[key].append(None)
            continue
            
        # Convert to lowercase
        name = name.lower()
        
        # Extract features (if needed)
        if extract_features:
            # Initialize features for this room
            room_features = {k: None for k in features.keys()}
            
            # Extract room type
            for room_type in room_type_mapping:
                if room_type in name.split():
                    room_features['room_type'] = room_type_mapping[room_type]
                    break
            
            # Extract bed type
            for bed_type in bed_type_mapping:
                if bed_type in name.split() and 'bed' in name:
                    room_features['bed_type'] = bed_type_mapping[bed_type]
                    break
            
            # Extract capacity/occupancy
            for pattern in capacity_patterns:
                match = re.search(pattern, name)
                if match:
                    room_features['capacity'] = int(match.group(1))
                    break
            
            # Extract area
            for pattern in area_patterns:
                match = re.search(pattern, name)
                if match:
                    room_features['area'] = int(match.group(1))
                    break
            
            # Check other features
            room_features['has_view'] = any(view in name for view in ['view', 'sea', 'ocean', 'mountain', 'garden', 'harbor'])
            room_features['has_balcony'] = 'balcony' in name
            room_features['has_terrace'] = 'terrace' in name
            room_features['is_smoking'] = 'smoking' in name and 'non' not in name and 'no' not in name
            room_features['is_non_smoking'] = ('non-smoking' in name) or ('no smoking' in name)
            room_features['has_breakfast'] = 'breakfast' in name
            
            # Add features to the main features dictionary
            for k in features.keys():
                features[k].append(room_features[k])
        
        # Clean text
        # Remove punctuation
        name = name.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        
        # Normalize common abbreviations and variants
        name = name.replace('w/', 'with')
        name = name.replace('w/o', 'without')
        name = name.replace('sq m', 'sqm')
        name = name.replace('sq. m', 'sqm')
        
        # Standardize room types
        for room_type, standard_type in room_type_mapping.items():
            name = re.sub(r'\b' + room_type + r'\b', standard_type, name)
        
        # Standardize bed types
        for bed_type, standard_bed in bed_type_mapping.items():
            if re.search(r'\b' + bed_type + r'\s+bed\b', name):
                name = re.sub(r'\b' + bed_type + r'\s+bed\b', standard_bed, name)
        
        # Custom tokenization - simple space splitting
        tokens = name.split()
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]
        
        # Simple lemmatization
        tokens = [lemma_mapping.get(token, token) for token in tokens]
        
        # Recombine into text
        processed_name = ' '.join(tokens)
        
        processed_names.append(processed_name)
    
    if extract_features:
        return processed_names, features
    else:
        return processed_names

def enhanced_room_matching(reference_rooms, supplier_rooms, model, similarity_threshold=0.6, feature_weight=0.3):
    """
    Room matching using a combination of text similarity and structured features
    
    Parameters:
    reference_rooms (DataFrame): Reference room data
    supplier_rooms (DataFrame): Supplier room data
    model: Loaded Sentence-BERT model
    similarity_threshold (float): Similarity threshold
    feature_weight (float): Weight for feature similarity
    
    Returns:
    matches (DataFrame): Matching results
    """
    # Extract text embeddings
    ref_embeddings = model.encode(reference_rooms['processed_name'].tolist())
    sup_embeddings = model.encode(supplier_rooms['processed_name'].tolist())
    
    # Calculate text similarity matrix
    text_similarity = cosine_similarity(ref_embeddings, sup_embeddings)
    
   
    # Find best matches
    matches = []
    for i in range(len(reference_rooms)):
        best_match_idx = np.argmax(text_similarity[i])
        best_match_score = text_similarity[i][best_match_idx]
        
        if best_match_score >= similarity_threshold:
            matches.append({
                'reference_hotel_id': reference_rooms['hotel_id'].iloc[i],
                'reference_room_id': reference_rooms['room_id'].iloc[i],
                'reference_room_name': reference_rooms['room_name'].iloc[i],
                'reference_processed_name': reference_rooms['processed_name'].iloc[i],
                'supplier_room_id': supplier_rooms['supplier_room_id'].iloc[best_match_idx],
                'supplier_room_name': supplier_rooms['supplier_room_name'].iloc[best_match_idx],
                'supplier_processed_name': supplier_rooms['processed_name'].iloc[best_match_idx],
                'text_similarity': text_similarity[i][best_match_idx],
            })
    
    return pd.DataFrame(matches)


def main(reference_rooms, supplier_rooms):
    """
    Main function to process reference and supplier room data
    
    Parameters:
    reference_rooms (DataFrame): Reference room data
    supplier_rooms (DataFrame): Supplier room data
    
    Returns:
    tuple: Processed reference and supplier room data
    """
    
    # Preprocess reference room names and extract features
    processed_ref_names, ref_features = preprocess_room_names(reference_rooms['room_name'].tolist(), extract_features=True)
    
    # Preprocess supplier room names and extract features
    processed_sup_names, sup_features = preprocess_room_names(supplier_rooms['supplier_room_name'].tolist(), extract_features=True)
    
    # Add processed names to original dataframes
    reference_rooms['processed_name'] = processed_ref_names
    supplier_rooms['processed_name'] = processed_sup_names
    
    # Add extracted features to original dataframes
    for feature_name, feature_values in ref_features.items():
        reference_rooms[feature_name] = feature_values
    
    for feature_name, feature_values in sup_features.items():
        supplier_rooms[feature_name] = feature_values
    
    return reference_rooms, supplier_rooms


if __name__ == "__main__":
    main()