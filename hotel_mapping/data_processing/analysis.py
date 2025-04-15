import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import nltk
from nltk.util import ngrams
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Set plot style with improved aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Define a comprehensive list of English stopwords
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

# Enhanced tokenizer function
def enhanced_tokenize(text):
    """Improved tokenization with better handling of special cases"""
    text = str(text).lower()
    # Remove punctuation but keep hyphens within words
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Split into words
    words = text.split()
    # Filter out stopwords and very short words
    words = [word for word in words if word not in STOPWORDS and len(word) > 1]
    return words

def extract_room_features(room_name):
    """Extract structured features from room names with improved pattern recognition"""
    room_name = str(room_name).lower()
    
    # Define more comprehensive patterns
    bed_patterns = {
        'single': r'single|1 bed|one bed',
        'double': r'double|2 bed|two bed',
        'twin': r'twin|2 single|two single',
        'queen': r'queen',
        'king': r'king',
        'super_king': r'super king|superking',
        'california_king': r'california king|cal king',
        'sofa_bed': r'sofa bed|pull-out|pullout|sofabed',
        'bunk': r'bunk',
        'multiple': r'3 bed|three bed|4 bed|four bed|multiple bed'
    }
    
    room_type_patterns = {
        'standard': r'standard|classic|basic',
        'deluxe': r'deluxe|luxury|premium|superior',
        'suite': r'suite',
        'apartment': r'apartment|flat|condo',
        'studio': r'studio',
        'villa': r'villa|bungalow|cottage',
        'family': r'family',
        'executive': r'executive|business|club',
        'accessible': r'accessible|disability|handicap',
        'connecting': r'connecting|adjoining'
    }
    
    view_patterns = {
        'sea': r'sea view|ocean view|beach view',
        'mountain': r'mountain view',
        'city': r'city view',
        'garden': r'garden view',
        'pool': r'pool view',
        'lake': r'lake view',
        'river': r'river view',
        'park': r'park view'
    }
    
    amenity_patterns = {
        'balcony': r'balcony|terrace|patio',
        'kitchen': r'kitchen|kitchenette',
        'jacuzzi': r'jacuzzi|hot tub|spa bath',
        'fireplace': r'fireplace',
        'private_bathroom': r'private bathroom|en-?suite',
        'shared_bathroom': r'shared bathroom|common bathroom',
        'air_conditioning': r'air conditioning|a/?c|climate control',
        'free_wifi': r'free wi-?fi|complimentary wi-?fi'
    }
    
    # Extract features
    features = {
        'length': len(room_name),
        'word_count': len(room_name.split()),
        'has_numeric': bool(re.search(r'\d', room_name)),
    }
    
    # Add bed types
    for bed_type, pattern in bed_patterns.items():
        features[f'bed_{bed_type}'] = bool(re.search(pattern, room_name))
    
    # Add room types
    for room_type, pattern in room_type_patterns.items():
        features[f'room_{room_type}'] = bool(re.search(pattern, room_name))
    
    # Add view types
    for view_type, pattern in view_patterns.items():
        features[f'view_{view_type}'] = bool(re.search(pattern, room_name))
    
    # Add amenities
    for amenity, pattern in amenity_patterns.items():
        features[f'amenity_{amenity}'] = bool(re.search(pattern, room_name))
    
    # Extract capacity if available
    capacity_match = re.search(r'(\d+)\s*person', room_name)
    if capacity_match:
        features['capacity'] = int(capacity_match.group(1))
    else:
        features['capacity'] = None
    
    # Extract area if available (in square meters or feet)
    area_match = re.search(r'(\d+)\s*(?:sq\.?m|m2|square meters?|sq\.?ft|square feet)', room_name)
    if area_match:
        features['area'] = int(area_match.group(1))
    else:
        features['area'] = None
    
    return features

# Custom similarity function implementations
def jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between two texts"""
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0

def levenshtein_distance(text1, text2):
    """Calculate Levenshtein distance between two texts"""
    if text1 == text2:
        return 0
    
    if len(text1) == 0:
        return len(text2)
    
    if len(text2) == 0:
        return len(text1)
    
    # Create distance matrix
    matrix = [[0 for _ in range(len(text2) + 1)] for _ in range(len(text1) + 1)]
    
    # Initialize first row and column
    for i in range(len(text1) + 1):
        matrix[i][0] = i
    
    for j in range(len(text2) + 1):
        matrix[0][j] = j
    
    # Fill the matrix
    for i in range(1, len(text1) + 1):
        for j in range(1, len(text2) + 1):
            cost = 0 if text1[i-1] == text2[j-1] else 1
            matrix[i][j] = min(
                matrix[i-1][j] + 1,      # deletion
                matrix[i][j-1] + 1,      # insertion
                matrix[i-1][j-1] + cost  # substitution
            )
    
    # Return the value in the bottom-right corner
    return matrix[len(text1)][len(text2)]

def levenshtein_similarity(text1, text2):
    """Calculate normalized Levenshtein similarity between two texts"""
    distance = levenshtein_distance(text1, text2)
    max_len = max(len(text1), len(text2))
    
    if max_len == 0:
        return 1.0
    
    return 1.0 - (distance / max_len)

def jaro_winkler_similarity(text1, text2):
    """Simplified Jaro-Winkler similarity, falls back to Levenshtein if unavailable"""
    # Since Jaro-Winkler algorithm is complex, we simplify by using Levenshtein similarity
    return levenshtein_similarity(text1, text2)

# Modified calculate_similarity_metrics function
def calculate_similarity_metrics(text1, text2):
    """Calculate multiple text similarity metrics"""
    text1 = str(text1).lower()
    text2 = str(text2).lower()
    
    # Calculate different similarity metrics
    jaccard_sim = jaccard_similarity(text1, text2)
    levenshtein_sim = levenshtein_similarity(text1, text2)
    jaro_sim = jaro_winkler_similarity(text1, text2)
    
    # Average similarity
    avg_sim = (jaccard_sim + levenshtein_sim + jaro_sim) / 3
    
    return {
        'jaccard': jaccard_sim,
        'levenshtein': levenshtein_sim,
        'jaro_winkler': jaro_sim,
        'average': avg_sim
    }

def analysis(reference_data, core_data, output_dir='./analysis_output'):
    """
    Comprehensive analysis of hotel room data with improved visualizations
    
    Parameters:
    reference_data (DataFrame): Reference hotel room data
    core_data (DataFrame): Supplier hotel room data
    output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== ENHANCED EXPLORATORY DATA ANALYSIS FOR ROOM MATCHING API ===\n")

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

    # 2. Check for missing values and data quality
    print("\n=== DATA QUALITY ANALYSIS ===")
    
    # Missing values
    ref_missing = reference_data.isnull().sum()
    core_missing = core_data.isnull().sum()
    
    print("\nMissing values in Reference Data:")
    print(ref_missing)
    print("\nMissing values in Core Data:")
    print(core_missing)
    
    # Check for duplicates
    ref_duplicates = reference_data.duplicated().sum()
    core_duplicates = core_data.duplicated().sum()
    
    print(f"\nDuplicate rows in Reference Data: {ref_duplicates} ({ref_duplicates/len(reference_data)*100:.2f}%)")
    print(f"Duplicate rows in Core Data: {core_duplicates} ({core_duplicates/len(core_data)*100:.2f}%)")
    
    # Check for empty room names
    ref_empty_names = (reference_data['room_name'].astype(str).str.strip() == '').sum()
    core_empty_names = (core_data['supplier_room_name'].astype(str).str.strip() == '').sum()
    
    print(f"\nEmpty room names in Reference Data: {ref_empty_names} ({ref_empty_names/len(reference_data)*100:.2f}%)")
    print(f"Empty room names in Core Data: {core_empty_names} ({core_empty_names/len(core_data)*100:.2f}%)")

    # 3. Analyze room name patterns with improved feature extraction
    print("\n=== ENHANCED ROOM NAME ANALYSIS ===")

    # Apply feature extraction to both datasets
    print("Extracting features from reference room names...")
    reference_features = reference_data['room_name'].apply(extract_room_features).apply(pd.Series)
    reference_features = pd.concat([reference_data[['room_id', 'room_name']], reference_features], axis=1)

    print("Extracting features from supplier room names...")
    core_features = core_data['supplier_room_name'].apply(extract_room_features).apply(pd.Series)
    core_features = pd.concat([core_data[['supplier_room_id', 'supplier_name', 'supplier_room_name']], core_features], axis=1)

    # Analyze feature distributions
    print("\nFeature distribution in Reference Room Names:")
    ref_feature_counts = {
        col: reference_features[col].sum() 
        for col in reference_features.columns 
        if col.startswith(('bed_', 'room_', 'view_', 'amenity_')) and reference_features[col].dtype == bool
    }
    ref_feature_df = pd.DataFrame.from_dict(ref_feature_counts, orient='index', columns=['count'])
    ref_feature_df['percentage'] = ref_feature_df['count'] / len(reference_features) * 100
    print(ref_feature_df.sort_values('count', ascending=False).head(15))

    print("\nFeature distribution in Supplier Room Names:")
    core_feature_counts = {
        col: core_features[col].sum() 
        for col in core_features.columns 
        if col.startswith(('bed_', 'room_', 'view_', 'amenity_')) and core_features[col].dtype == bool
    }
    core_feature_df = pd.DataFrame.from_dict(core_feature_counts, orient='index', columns=['count'])
    core_feature_df['percentage'] = core_feature_df['count'] / len(core_features) * 100
    print(core_feature_df.sort_values('count', ascending=False).head(15))

    # 4. Enhanced word frequency analysis with n-grams
    print("\n=== ENHANCED WORD FREQUENCY ANALYSIS ===")

    def get_ngram_frequency(text_series, n=1, top_n=20):
        """Get frequency of n-grams in text series"""
        all_words = []
        for text in text_series.astype(str):
            tokens = enhanced_tokenize(text)
            if n == 1:
                all_words.extend(tokens)
            else:
                all_words.extend([' '.join(gram) for gram in list(ngrams(tokens, n))])
        return Counter(all_words).most_common(top_n)

    # Unigrams (single words)
    ref_unigrams = get_ngram_frequency(reference_data['room_name'])
    core_unigrams = get_ngram_frequency(core_data['supplier_room_name'])
    
    print("\nTop 20 words in Reference Room Names:")
    print(ref_unigrams)
    
    print("\nTop 20 words in Supplier Room Names:")
    print(core_unigrams)
    
    # Bigrams (word pairs)
    ref_bigrams = get_ngram_frequency(reference_data['room_name'], n=2, top_n=15)
    core_bigrams = get_ngram_frequency(core_data['supplier_room_name'], n=2, top_n=15)
    
    print("\nTop 15 word pairs in Reference Room Names:")
    print(ref_bigrams)
    
    print("\nTop 15 word pairs in Supplier Room Names:")
    print(core_bigrams)

    # 5. Analyze room name length and word count distributions
    print("\n=== ROOM NAME LENGTH ANALYSIS ===")

    reference_data['name_length'] = reference_data['room_name'].astype(str).apply(len)
    reference_data['word_count'] = reference_data['room_name'].astype(str).apply(lambda x: len(x.split()))

    core_data['name_length'] = core_data['supplier_room_name'].astype(str).apply(len)
    core_data['word_count'] = core_data['supplier_room_name'].astype(str).apply(lambda x: len(x.split()))

    print("\nReference Room Name Length Statistics:")
    print(reference_data[['name_length', 'word_count']].describe())

    print("\nSupplier Room Name Length Statistics:")
    print(core_data[['name_length', 'word_count']].describe())

    # 6. Analyze supplier distribution with improved visualization
    print("\n=== SUPPLIER DISTRIBUTION ANALYSIS ===")
    
    supplier_counts = core_data['supplier_name'].value_counts()
    total_suppliers = len(supplier_counts)
    top_suppliers = min(10, total_suppliers)
    
    print(f"\nTotal unique suppliers: {total_suppliers}")
    print(f"\nTop {top_suppliers} suppliers by room count:")
    print(supplier_counts.head(top_suppliers))
    
    # Calculate supplier diversity metrics
    supplier_share = supplier_counts / supplier_counts.sum() * 100
    top_supplier_share = supplier_share.iloc[0]
    top_5_share = supplier_share.iloc[:5].sum() if len(supplier_share) >= 5 else supplier_share.sum()
    
    print(f"\nTop supplier market share: {top_supplier_share:.2f}%")
    print(f"Top 5 suppliers market share: {top_5_share:.2f}%")
    
    # Calculate Herfindahl-Hirschman Index (HHI) for supplier concentration
    hhi = (supplier_share ** 2).sum() / 100
    print(f"Supplier concentration (HHI): {hhi:.2f} (higher values indicate more concentration)")

    # 7. Enhanced similarity analysis with multiple metrics
    print("\n=== ENHANCED SIMILARITY ANALYSIS ===")
    
    # Sample data for similarity analysis
    max_sample = 1000  # Adjust based on your system's capacity
    ref_sample = reference_data.sample(min(len(reference_data), max_sample))
    core_sample = core_data.sample(min(len(core_data), max_sample))
    
    # Preprocess room names
    ref_sample['processed_name'] = ref_sample['room_name'].apply(lambda x: ' '.join(enhanced_tokenize(x)))
    core_sample['processed_name'] = core_sample['supplier_room_name'].apply(lambda x: ' '.join(enhanced_tokenize(x)))
    
    # Calculate similarity for a subset of pairs
    print("Calculating similarity metrics for sample pairs...")
    similarity_results = []
    
    # Limit the number of pairs to analyze
    max_pairs = 100
    pair_count = 0
    
    for i, ref_row in ref_sample.iterrows():
        if pair_count >= max_pairs:
            break
            
        for j, core_row in core_sample.iterrows():
            if pair_count >= max_pairs:
                break
                
            # Calculate similarity metrics
            sim_metrics = calculate_similarity_metrics(ref_row['processed_name'], core_row['processed_name'])
            
            similarity_results.append({
                'reference_room': ref_row['room_name'],
                'supplier_room': core_row['supplier_room_name'],
                'supplier': core_row['supplier_name'],
                'jaccard_sim': sim_metrics['jaccard'],
                'levenshtein_sim': sim_metrics['levenshtein'],
                'jaro_winkler_sim': sim_metrics['jaro_winkler'],
                'avg_sim': sim_metrics['average']
            })
            
            pair_count += 1
    
    # Convert to DataFrame
    similarity_df = pd.DataFrame(similarity_results)
    
    # Summary statistics for similarity metrics
    print("\nSimilarity Metrics Summary:")
    print(similarity_df[['jaccard_sim', 'levenshtein_sim', 'jaro_winkler_sim', 'avg_sim']].describe())
    
    # Find best matches based on average similarity
    print("\nTop 10 most similar room pairs:")
    top_matches = similarity_df.sort_values('avg_sim', ascending=False).head(10)
    for _, row in top_matches.iterrows():
        print(f"Reference: '{row['reference_room']}' ↔ Supplier: '{row['supplier_room']}' (Supplier: {row['supplier']}, Similarity: {row['avg_sim']:.4f})")

    # 8. Visualizations with improved aesthetics
    print("\n=== ENHANCED VISUALIZATIONS ===")

    # Figure 1: Room name length and word count distributions
    plt.figure(figsize=(16, 8))
    
    # Room name length distribution
    plt.subplot(1, 2, 1)
    sns.histplot(
        data=pd.DataFrame({
            'Reference': reference_data['name_length'],
            'Supplier': core_data['name_length']
        }).melt(var_name='Dataset', value_name='Character Count'),
        x='Character Count', hue='Dataset', kde=True, element='step',
        palette=['#4878D0', '#EE854A'], alpha=0.7, bins=30
    )
    plt.title('Room Name Length Distribution', fontweight='bold')
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Word count distribution
    plt.subplot(1, 2, 2)
    sns.histplot(
        data=pd.DataFrame({
            'Reference': reference_data['word_count'],
            'Supplier': core_data['word_count']
        }).melt(var_name='Dataset', value_name='Word Count'),
        x='Word Count', hue='Dataset', kde=True, element='step',
        palette=['#4878D0', '#EE854A'], alpha=0.7, bins=15
    )
    plt.title('Word Count Distribution in Room Names', fontweight='bold')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/room_name_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved room name distributions to {output_dir}/room_name_distributions.png")

    # Figure 2: Supplier distribution and feature comparison
    plt.figure(figsize=(16, 8))
    
    # Supplier distribution
    plt.subplot(1, 2, 1)
    top_n_suppliers = min(10, len(supplier_counts))
    supplier_plot = supplier_counts.head(top_n_suppliers).plot(
        kind='bar', color=sns.color_palette('viridis', top_n_suppliers)
    )
    plt.title('Top Suppliers by Room Count', fontweight='bold')
    plt.xlabel('Supplier')
    plt.ylabel('Number of Rooms')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    total = supplier_counts.sum()
    for i, count in enumerate(supplier_counts.head(top_n_suppliers)):
        percentage = count / total * 100
        supplier_plot.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Feature comparison
    plt.subplot(1, 2, 2)
    
    # Combine feature counts for comparison
    common_features = set(ref_feature_df.index) & set(core_feature_df.index)
    top_common_features = sorted(
        common_features, 
        key=lambda x: (ref_feature_df.loc[x, 'count'] + core_feature_df.loc[x, 'count']), 
        reverse=True
    )[:10]
    
    feature_comparison = pd.DataFrame({
        'Reference': [ref_feature_df.loc[f, 'percentage'] for f in top_common_features],
        'Supplier': [core_feature_df.loc[f, 'percentage'] for f in top_common_features]
    }, index=[f.replace('_', ' ').title() for f in top_common_features])
    
    feature_comparison.plot(kind='barh', color=['#4878D0', '#EE854A'])
    plt.title('Top Features in Room Names', fontweight='bold')
    plt.xlabel('Percentage of Rooms')
    plt.ylabel('Feature')
    plt.grid(True, alpha=0.3, axis='x')
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/supplier_feature_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved supplier and feature analysis to {output_dir}/supplier_feature_analysis.png")

    # Figure 3: Word clouds for reference and supplier room names
    try:
        from wordcloud import WordCloud
        
        plt.figure(figsize=(16, 8))
        
        # Reference word cloud
        plt.subplot(1, 2, 1)
        ref_text = ' '.join(reference_data['room_name'].astype(str))
        ref_wordcloud = WordCloud(
            width=800, height=400, background_color='white', 
            max_words=100, colormap='viridis', collocations=False
        ).generate(ref_text)
        
        plt.imshow(ref_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Reference Room Names Word Cloud', fontweight='bold', pad=20)
        
        # Supplier word cloud
        plt.subplot(1, 2, 2)
        sup_text = ' '.join(core_data['supplier_room_name'].astype(str))
        sup_wordcloud = WordCloud(
            width=800, height=400, background_color='white', 
            max_words=100, colormap='plasma', collocations=False
        ).generate(sup_text)
        
        plt.imshow(sup_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Supplier Room Names Word Cloud', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/room_name_wordclouds.png', dpi=300, bbox_inches='tight')
        print(f"Saved word clouds to {output_dir}/room_name_wordclouds.png")
    except ImportError:
        print("WordCloud package not installed. Skipping word cloud visualization.")

    # 9. Advanced analysis: Clustering room names
    print("\n=== ADVANCED ROOM NAME CLUSTERING ===")
    
    # Use TF-IDF vectorization for clustering
    tfidf_vectorizer = TfidfVectorizer(
        max_features=100,
        min_df=5,
        max_df=0.5,
        ngram_range=(1, 2)
    )
    
    # Process reference data
    ref_tfidf = tfidf_vectorizer.fit_transform(ref_sample['processed_name'])
    
    # Apply dimensionality reduction for visualization
    pca = PCA(n_components=2)
    ref_pca = pca.fit_transform(ref_tfidf.toarray())
    
    # Apply K-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    ref_clusters = kmeans.fit_predict(ref_tfidf)
    
    # Add cluster information to sample
    ref_sample['cluster'] = ref_clusters
    
    # Analyze clusters
    print("\nReference room name clusters:")
    for cluster in range(n_clusters):
        cluster_rooms = ref_sample[ref_sample['cluster'] == cluster]['room_name'].tolist()
        print(f"\nCluster {cluster+1} ({len(cluster_rooms)} rooms):")
        print(f"Sample rooms: {', '.join(cluster_rooms[:5])}")
        
        # Get most common words in this cluster
        cluster_text = ' '.join(ref_sample[ref_sample['cluster'] == cluster]['processed_name'])
        cluster_words = Counter(cluster_text.split()).most_common(5)
        print(f"Most common words: {', '.join([word for word, count in cluster_words])}")
    
    # Visualize clusters
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(ref_pca[:, 0], ref_pca[:, 1], c=ref_clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Room Name Clusters (PCA Visualization)', fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for a few points from each cluster
    for cluster in range(n_clusters):
        cluster_indices = np.where(ref_clusters == cluster)[0]
        if len(cluster_indices) > 0:
            # Select a few random points to annotate
            sample_indices = np.random.choice(cluster_indices, min(3, len(cluster_indices)), replace=False)
            for idx in sample_indices:
                plt.annotate(
                    ref_sample.iloc[idx]['room_name'][:20] + ('...' if len(ref_sample.iloc[idx]['room_name']) > 20 else ''),
                    (ref_pca[idx, 0], ref_pca[idx, 1]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
                )
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/room_name_clusters.png', dpi=300, bbox_inches='tight')
    print(f"Saved room name clusters visualization to {output_dir}/room_name_clusters.png")

    # 10. N-gram analysis for common phrases
    print("\n=== N-GRAM ANALYSIS ===")
    
    # Function to extract n-grams
    def get_ngrams(text_series, n):
        all_ngrams = []
        for text in text_series.astype(str):
            tokens = enhanced_tokenize(text)
            if len(tokens) >= n:
                all_ngrams.extend(list(ngrams(tokens, n)))
        return Counter(all_ngrams)
    
    # Get bigrams and trigrams for both datasets
    ref_bigrams = get_ngrams(reference_data['room_name'], 2)
    ref_trigrams = get_ngrams(reference_data['room_name'], 3)
    
    sup_bigrams = get_ngrams(core_data['supplier_room_name'], 2)
    sup_trigrams = get_ngrams(core_data['supplier_room_name'], 3)
    
    # Print top n-grams
    print("\nTop 10 bigrams in Reference Room Names:")
    for ngram, count in ref_bigrams.most_common(10):
        print(f"'{' '.join(ngram)}': {count}")
    
    print("\nTop 10 bigrams in Supplier Room Names:")
    for ngram, count in sup_bigrams.most_common(10):
        print(f"'{' '.join(ngram)}': {count}")
    
    print("\nTop 10 trigrams in Reference Room Names:")
    for ngram, count in ref_trigrams.most_common(10):
        print(f"'{' '.join(ngram)}': {count}")
    
    print("\nTop 10 trigrams in Supplier Room Names:")
    for ngram, count in sup_trigrams.most_common(10):
        print(f"'{' '.join(ngram)}': {count}")
    
    # Visualize top n-grams
    plt.figure(figsize=(16, 12))
    
    # Reference bigrams
    plt.subplot(2, 2, 1)
    ref_bigram_df = pd.DataFrame(ref_bigrams.most_common(10), columns=['Bigram', 'Count'])
    ref_bigram_df['Bigram'] = ref_bigram_df['Bigram'].apply(lambda x: ' '.join(x))
    sns.barplot(x='Count', y='Bigram', data=ref_bigram_df, palette='viridis')
    plt.title('Top Bigrams in Reference Room Names', fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Bigram')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Supplier bigrams
    plt.subplot(2, 2, 2)
    sup_bigram_df = pd.DataFrame(sup_bigrams.most_common(10), columns=['Bigram', 'Count'])
    sup_bigram_df['Bigram'] = sup_bigram_df['Bigram'].apply(lambda x: ' '.join(x))
    sns.barplot(x='Count', y='Bigram', data=sup_bigram_df, palette='plasma')
    plt.title('Top Bigrams in Supplier Room Names', fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Bigram')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Reference trigrams
    plt.subplot(2, 2, 3)
    ref_trigram_df = pd.DataFrame(ref_trigrams.most_common(10), columns=['Trigram', 'Count'])
    ref_trigram_df['Trigram'] = ref_trigram_df['Trigram'].apply(lambda x: ' '.join(x))
    sns.barplot(x='Count', y='Trigram', data=ref_trigram_df, palette='viridis')
    plt.title('Top Trigrams in Reference Room Names', fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Trigram')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Supplier trigrams
    plt.subplot(2, 2, 4)
    sup_trigram_df = pd.DataFrame(sup_trigrams.most_common(10), columns=['Trigram', 'Count'])
    sup_trigram_df['Trigram'] = sup_trigram_df['Trigram'].apply(lambda x: ' '.join(x))
    sns.barplot(x='Count', y='Trigram', data=sup_trigram_df, palette='plasma')
    plt.title('Top Trigrams in Supplier Room Names', fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Trigram')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ngram_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved n-gram analysis to {output_dir}/ngram_analysis.png")



    # 13. Summary and recommendations
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    # Calculate overall statistics
    avg_similarity = similarity_df['avg_sim'].mean()
    max_similarity = similarity_df['avg_sim'].max()
    min_similarity = similarity_df['avg_sim'].min()
    
    print(f"\nOverall average similarity: {avg_similarity:.4f}")
    print(f"Maximum similarity: {max_similarity:.4f}")
    print(f"Minimum similarity: {min_similarity:.4f}")
    
    # Generate recommendations based on analysis
    print("\nKey findings and recommendations:")
    
    # Data quality recommendations
    if ref_missing.sum() > 0 or core_missing.sum() > 0:
        print("- Data quality issues detected: Handle missing values before matching")
    
    if ref_empty_names > 0 or core_empty_names > 0:
        print("- Empty room names detected: Filter out or handle empty names")
    
    # Supplier-specific recommendations
    if len(supplier_counts) > 1:
        print("- Different suppliers use different naming conventions; consider supplier-specific matching rules")
    
    # Similarity threshold recommendations
    if avg_similarity < 0.3:
        print(f"- Low average similarity ({avg_similarity:.2f}) suggests using a lower threshold for matching")
    elif avg_similarity > 0.7:
        print(f"- High average similarity ({avg_similarity:.2f}) suggests using a higher threshold for matching")
    


    print("\nAnalysis complete! All visualizations saved to:", output_dir)
    return {
        'reference_features': reference_features,
        'core_features': core_features,
        'similarity_results': similarity_df,
        'output_dir': output_dir
    }