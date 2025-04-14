import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Define a simple list of English stopwords
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
    'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
    'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', 'should', 'now'
}

# Simple tokenizer function
def simple_tokenize(text):
    text = str(text).lower()
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text)
    return words

def analysis(reference_data, core_data):
    print("=== EXPLORATORY DATA ANALYSIS FOR ROOM MATCHING API ===\n")

    # 1. Basic Dataset Information
    print("=== DATASET OVERVIEW ===")
    print("\nReference Data Shape:", reference_data.shape)
    print("\nReference Data Sample:")
    print(reference_data.head())
    print("\nReference Data Info:")
    print(reference_data.info())
    print("\nReference Data Description:")
    print(reference_data.describe(include='all'))

    print("\n\nCore Data Shape:", core_data.shape)
    print("\nCore Data Sample:")
    print(core_data.head())
    print("\nCore Data Info:")
    print(core_data.info())
    print("\nCore Data Description:")
    print(core_data.describe(include='all'))

    # 2. Check for missing values
    print("\n=== MISSING VALUES ===")
    print("\nMissing values in Reference Data:")
    print(reference_data.isnull().sum())
    print("\nMissing values in Core Data:")
    print(core_data.isnull().sum())

    # 3. Analyze room name patterns
    print("\n=== ROOM NAME ANALYSIS ===")

    # Function to extract room name features
    def extract_room_features(room_name):
        room_name = str(room_name).lower()
        features = {
            'length': len(room_name),
            'word_count': len(room_name.split()),
            'has_numeric': bool(re.search(r'\d', room_name)),
            'has_bed_mention': any(bed in room_name for bed in ['bed', 'twin', 'double', 'single', 'king', 'queen']),
            'has_room_type': any(room_type in room_name for room_type in ['room', 'suite', 'apartment', 'house', 'cottage', 'dormitory']),
            'has_quality_indicator': any(quality in room_name for quality in ['deluxe', 'superior', 'standard', 'premium', 'luxury', 'budget', 'classic']),
            'has_feature': any(feature in room_name for feature in ['balcony', 'view', 'terrace', 'sea', 'harbor', 'beach'])
        }
        return features

    # Apply feature extraction to both datasets
    reference_features = reference_data['room_name'].apply(extract_room_features).apply(pd.Series)
    reference_features = pd.concat([reference_data[['room_id', 'room_name']], reference_features], axis=1)

    core_features = core_data['supplier_room_name'].apply(extract_room_features).apply(pd.Series)
    core_features = pd.concat([core_data[['supplier_room_id', 'supplier_name', 'supplier_room_name']], core_features], axis=1)

    print("\nReference Room Name Features:")
    print(reference_features.head())

    print("\nSupplier Room Name Features:")
    print(core_features.head())

    # 4. Analyze word frequency in room names
    print("\n=== WORD FREQUENCY ANALYSIS ===")

    def get_word_frequency(text_series):
        all_words = ' '.join(text_series.astype(str)).lower()
        # Tokenize and filter stopwords
        words = simple_tokenize(all_words)
        filtered_words = [word for word in words if word not in STOPWORDS]
        return Counter(filtered_words)

    reference_word_freq = get_word_frequency(reference_data['room_name'])
    supplier_word_freq = get_word_frequency(core_data['supplier_room_name'])

    print("\nTop 10 words in Reference Room Names:")
    print(reference_word_freq.most_common(10))

    print("\nTop 10 words in Supplier Room Names:")
    print(supplier_word_freq.most_common(10))

    # 5. Analyze room name length distribution
    print("\n=== ROOM NAME LENGTH ANALYSIS ===")

    reference_data['name_length'] = reference_data['room_name'].astype(str).apply(len)
    reference_data['word_count'] = reference_data['room_name'].astype(str).apply(lambda x: len(x.split()))

    core_data['name_length'] = core_data['supplier_room_name'].astype(str).apply(len)
    core_data['word_count'] = core_data['supplier_room_name'].astype(str).apply(lambda x: len(x.split()))

    print("\nReference Room Name Length Statistics:")
    print(reference_data[['name_length', 'word_count']].describe())

    print("\nSupplier Room Name Length Statistics:")
    print(core_data[['name_length', 'word_count']].describe())

    # 6. Analyze supplier distribution
    print("\n=== SUPPLIER DISTRIBUTION ===")
    supplier_counts = core_data['supplier_name'].value_counts()
    print(supplier_counts)

    # 7. Text similarity analysis - Optimized version
    print("\n=== TEXT SIMILARITY ANALYSIS ===")

    # Function to preprocess text
    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    # Preprocess room names
    reference_data['processed_name'] = reference_data['room_name'].apply(preprocess_text)
    core_data['processed_name'] = core_data['supplier_room_name'].apply(preprocess_text)

    # Create TF-IDF vectors with optimized parameters
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features to top 1000 terms
        min_df=2,           # Ignore terms that appear in fewer than 2 documents
        max_df=0.9,         # Ignore terms that appear in more than 90% of documents
        ngram_range=(1, 2)  # Use unigrams and bigrams
    )
    
    # Use a sample of the data if it's very large
    max_rows = 1000  # Adjust based on your system's capacity
    ref_sample = reference_data.head(min(len(reference_data), max_rows))
    core_sample = core_data.head(min(len(core_data), max_rows))
    
    
    # Fit on combined corpus to ensure consistent vocabulary
    combined_corpus = list(ref_sample['processed_name']) + list(core_sample['processed_name'])
    tfidf_vectorizer.fit(combined_corpus)
    
    # Transform separately
    reference_tfidf = tfidf_vectorizer.transform(ref_sample['processed_name'])
    supplier_tfidf = tfidf_vectorizer.transform(core_sample['processed_name'])
    
    print("Calculating similarity matrix...")
    
    # Calculate similarity matrix in batches if needed
    batch_size = 100  # Adjust based on your system's capacity
    
    if len(ref_sample) > batch_size or len(core_sample) > batch_size:
        # Initialize empty similarity matrix
        similarity_matrix = np.zeros((len(ref_sample), len(core_sample)))
        
        # Process in batches
        for i in range(0, len(ref_sample), batch_size):
            end_i = min(i + batch_size, len(ref_sample))
            ref_batch = reference_tfidf[i:end_i]
            
            for j in range(0, len(core_sample), batch_size):
                end_j = min(j + batch_size, len(core_sample))
                core_batch = supplier_tfidf[j:end_j]
                
                # Calculate similarity for this batch
                batch_sim = cosine_similarity(ref_batch, core_batch)
                
                # Store in the full matrix
                similarity_matrix[i:end_i, j:end_j] = batch_sim
                
    else:
        # For smaller datasets, calculate the full matrix at once
        similarity_matrix = cosine_similarity(reference_tfidf, supplier_tfidf)
    
    # Create a DataFrame to show similarity scores
    # Only show a subset if the matrix is large
    max_display = 20  # Maximum number of rows/columns to display
    display_rows = min(len(ref_sample), max_display)
    display_cols = min(len(core_sample), max_display)
    
    display_matrix = similarity_matrix[:display_rows, :display_cols]
    display_ref_names = ref_sample['room_name'].iloc[:display_rows]
    display_core_names = core_sample['supplier_room_name'].iloc[:display_cols]
    
    similarity_df = pd.DataFrame(
        display_matrix, 
        index=display_ref_names,
        columns=display_core_names
    )

    # print("\nText Similarity Matrix (TF-IDF Cosine Similarity) - Sample:")
    # print(similarity_df)
    
    # Save full matrix to file instead of displaying
    if len(ref_sample) > max_display or len(core_sample) > max_display:
        full_similarity_df = pd.DataFrame(
            similarity_matrix,
            index=ref_sample['room_name'],
            columns=core_sample['supplier_room_name']
        )
        full_similarity_df.to_csv('full_similarity_matrix.csv')
        print(f"Full similarity matrix saved to 'full_similarity_matrix.csv'")


    # 9. Visualizations
    print("\n=== VISUALIZATIONS ===")

    # Set up the figure for visualizations
    plt.figure(figsize=(15, 10))

    # Plot 1: Room name length distribution
    plt.subplot(2, 2, 1)
    sns.histplot(reference_data['name_length'], color='blue', label='Reference', alpha=0.5, kde=True)
    sns.histplot(core_data['name_length'], color='red', label='Supplier', alpha=0.5, kde=True)
    plt.title('Room Name Length Distribution')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 2: Word count distribution
    plt.subplot(2, 2, 2)
    sns.histplot(reference_data['word_count'], color='blue', label='Reference', alpha=0.5, kde=True)
    sns.histplot(core_data['word_count'], color='red', label='Supplier', alpha=0.5, kde=True)
    plt.title('Word Count Distribution in Room Names')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 3: Supplier distribution
    plt.subplot(2, 2, 3)
    supplier_counts.plot(kind='bar', color='skyblue')
    plt.title('Supplier Distribution')
    plt.xlabel('Supplier')
    plt.ylabel('Count')
    plt.xticks(rotation=45)

    # Plot 4: Heatmap of similarity matrix
    plt.subplot(2, 2, 4)
    sns.heatmap(similarity_df, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Room Name Similarity Heatmap')
    plt.tight_layout()

    # Save the figure
    plt.savefig('room_matching_eda.png')
    print("Visualizations saved to 'room_matching_eda.png'")

    # 10. Feature importance for matching
    print("\n=== FEATURE IMPORTANCE FOR MATCHING ===")

    # Combine all features for analysis
    all_features = pd.concat([
        reference_features.drop(['room_id', 'room_name'], axis=1).mean(),
        core_features.drop(['supplier_room_id', 'supplier_name', 'supplier_room_name'], axis=1).mean()
    ], axis=1)
    all_features.columns = ['Reference', 'Supplier']

    print("\nFeature Presence Comparison:")
    print(all_features)

    # 11. Potential matching pairs based on similarity
    print("\n=== POTENTIAL MATCHING PAIRS ===")

    # Find the best match for each reference room
    best_matches = []
    
    # Make sure we only process rooms that are in our similarity matrix
    ref_sample_size = min(len(reference_data), max_rows)
    core_sample_size = min(len(core_data), max_rows)
    
    for i, ref_room in enumerate(reference_data['room_name'][:ref_sample_size]):
        if i < similarity_matrix.shape[0]:  # Ensure index is within bounds
            # Find best match index, ensuring it's within bounds
            best_match_idx = np.argmax(similarity_matrix[i])
            
            if best_match_idx < core_sample_size:  # Ensure supplier index is valid
                best_match_score = similarity_matrix[i, best_match_idx]
                best_match_room = core_data.iloc[best_match_idx]['supplier_room_name']
                best_match_supplier = core_data.iloc[best_match_idx]['supplier_name']
                
                best_matches.append({
                    'reference_room': ref_room,
                    'best_match': best_match_room,
                    'supplier': best_match_supplier,
                    'similarity_score': best_match_score
                })
    
    best_matches_df = pd.DataFrame(best_matches)
    if not best_matches_df.empty:
        print(best_matches_df)
    else:
        print("No matches found in the analyzed sample.")

    # 12. Summary statistics
    print("\n=== SUMMARY STATISTICS ===")

    print("\nAverage similarity score:", best_matches_df['similarity_score'].mean())
    print("Minimum similarity score:", best_matches_df['similarity_score'].min())
    print("Maximum similarity score:", best_matches_df['similarity_score'].max())

    