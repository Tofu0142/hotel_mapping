import pandas as pd
import numpy as np
import re
import string
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_room_names(room_names, extract_features=True):
    """
    对酒店房间名称进行全面预处理，不使用NLTK
    
    参数:
    room_names (list): 房间名称列表
    extract_features (bool): 是否提取结构化特征
    
    返回:
    processed_names (list): 预处理后的房间名称
    features (dict, optional): 提取的结构化特征
    """
    processed_names = []
    features = {} if extract_features else None
    
    # 自定义英文停用词列表
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                 'at', 'from', 'by', 'for', 'with', 'about', 'against', 'between',
                 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                 'to', 'of', 'in', 'on', 'is', 'are', 'was', 'were'}
    
    # 房间类型标准化映射
    room_type_mapping = {
        'double': 'double', 'twin': 'twin', 'single': 'single',
        'queen': 'queen', 'king': 'king', 'suite': 'suite',
        'apartment': 'apartment', 'studio': 'studio', 'deluxe': 'deluxe',
        'standard': 'standard', 'superior': 'superior', 'executive': 'executive',
        'family': 'family', 'junior': 'junior', 'presidential': 'presidential',
        'cottage': 'cottage', 'villa': 'villa', 'bungalow': 'bungalow',
        'dormitory': 'dormitory', 'dorm': 'dormitory'
    }
    
    # 床型标准化映射
    bed_type_mapping = {
        'queen': 'queen bed', 'king': 'king bed', 
        'single': 'single bed', 'double': 'double bed',
        'twin': 'twin bed', 'sofa': 'sofa bed'
    }
    
    # 容量/人数模式
    capacity_patterns = [
        r'(\d+)\s*person', r'(\d+)\s*people', r'(\d+)\s*bed',
        r'for\s*(\d+)', r'(\d+)\s*adult', r'(\d+)\s*pax'
    ]
    
    # 面积模式
    area_patterns = [
        r'(\d+)\s*m²', r'(\d+)\s*sqm', r'(\d+)\s*square\s*meter'
    ]
    
    # 简单的词形还原映射
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
            
        # 转换为小写
        name = name.lower()
        
        # 提取特征（如果需要）
        if extract_features:
            # 初始化该房间的特征
            room_features = {k: None for k in features.keys()}
            
            # 提取房间类型
            for room_type in room_type_mapping:
                if room_type in name.split():
                    room_features['room_type'] = room_type_mapping[room_type]
                    break
            
            # 提取床型
            for bed_type in bed_type_mapping:
                if bed_type in name.split() and 'bed' in name:
                    room_features['bed_type'] = bed_type_mapping[bed_type]
                    break
            
            # 提取容量/人数
            for pattern in capacity_patterns:
                match = re.search(pattern, name)
                if match:
                    room_features['capacity'] = int(match.group(1))
                    break
            
            # 提取面积
            for pattern in area_patterns:
                match = re.search(pattern, name)
                if match:
                    room_features['area'] = int(match.group(1))
                    break
            
            # 检查其他特征
            room_features['has_view'] = any(view in name for view in ['view', 'sea', 'ocean', 'mountain', 'garden', 'harbor'])
            room_features['has_balcony'] = 'balcony' in name
            room_features['has_terrace'] = 'terrace' in name
            room_features['is_smoking'] = 'smoking' in name and 'non' not in name and 'no' not in name
            room_features['is_non_smoking'] = ('non-smoking' in name) or ('no smoking' in name)
            room_features['has_breakfast'] = 'breakfast' in name
            
            # 将特征添加到总特征字典中
            for k in features.keys():
                features[k].append(room_features[k])
        
        # 清理文本
        # 移除标点符号
        name = name.translate(str.maketrans('', '', string.punctuation))
        
        # 标准化空格
        name = re.sub(r'\s+', ' ', name).strip()
        
        # 标准化常见缩写和变体
        name = name.replace('w/', 'with')
        name = name.replace('w/o', 'without')
        name = name.replace('sq m', 'sqm')
        name = name.replace('sq. m', 'sqm')
        
        # 标准化房间类型
        for room_type, standard_type in room_type_mapping.items():
            name = re.sub(r'\b' + room_type + r'\b', standard_type, name)
        
        # 标准化床型
        for bed_type, standard_bed in bed_type_mapping.items():
            if re.search(r'\b' + bed_type + r'\s+bed\b', name):
                name = re.sub(r'\b' + bed_type + r'\s+bed\b', standard_bed, name)
        
        # 自定义分词 - 简单按空格分割
        tokens = name.split()
        
        # 移除停用词
        tokens = [token for token in tokens if token not in stop_words]
        
        # 简单的词形还原
        tokens = [lemma_mapping.get(token, token) for token in tokens]
        
        # 重新组合为文本
        processed_name = ' '.join(tokens)
        
        processed_names.append(processed_name)
    
    if extract_features:
        return processed_names, features
    else:
        return processed_names

def enhanced_room_matching(reference_rooms, supplier_rooms, model, similarity_threshold=0.6, feature_weight=0.3):
    """
    使用文本相似度和结构化特征的组合方法进行房间匹配
    
    参数:
    reference_rooms (DataFrame): 参考房间数据
    supplier_rooms (DataFrame): 供应商房间数据
    model: 已加载的Sentence-BERT模型
    similarity_threshold (float): 相似度阈值
    feature_weight (float): 特征相似度的权重
    
    返回:
    matches (DataFrame): 匹配结果
    """
    # 提取文本嵌入
    ref_embeddings = model.encode(reference_rooms['processed_name'].tolist())
    sup_embeddings = model.encode(supplier_rooms['processed_name'].tolist())
    
    # 计算文本相似度矩阵
    text_similarity = cosine_similarity(ref_embeddings, sup_embeddings)
    
   
    # 找到最佳匹配
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

    
    # 预处理参考房间名称并提取特征
    processed_ref_names, ref_features = preprocess_room_names(reference_rooms['room_name'].tolist(), extract_features=True)
    
    # 预处理供应商房间名称并提取特征
    processed_sup_names, sup_features = preprocess_room_names(supplier_rooms['supplier_room_name'].tolist(), extract_features=True)
    
    # 将处理后的名称添加到原始数据框中
    reference_rooms['processed_name'] = processed_ref_names
    supplier_rooms['processed_name'] = processed_sup_names
    
    # 将提取的特征添加到原始数据框中
    for feature_name, feature_values in ref_features.items():
        reference_rooms[feature_name] = feature_values
    
    for feature_name, feature_values in sup_features.items():
        supplier_rooms[feature_name] = feature_values
    return reference_rooms, supplier_rooms



if __name__ == "__main__":
    main()