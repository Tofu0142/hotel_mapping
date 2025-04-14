import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 在导入模块之前模拟依赖
from unittest.mock import MagicMock, patch

# 模拟 matplotlib
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# 模拟 sentence_transformers
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()

# 现在导入模块
from hotel_mapping.data_processing.data_processing import preprocess_room_names, enhanced_room_matching
import pytest

# 创建模拟的 BertXGBoostRoomMatcher 类
class MockBertXGBoostRoomMatcher:
    def __init__(self, *args, **kwargs):
        pass
        
    def load_model(self):
        return None
        
    def predict_similarity(self, *args, **kwargs):
        return np.array([0.9, 0.8, 0.3])

# 替换真实的类
sys.modules['hotel_mapping.models.bert_xgb'] = MagicMock()
sys.modules['hotel_mapping.models.bert_xgb'].BertXGBoostRoomMatcher = MockBertXGBoostRoomMatcher

class TestDataProcessing(unittest.TestCase):
    
    def test_preprocess_room_names(self):
        # 测试预处理函数
        room_names = ["Superior Room, Mountain View", "Deluxe King Room with Balcony"]
        processed_names, features = preprocess_room_names(room_names, extract_features=True)
        
        # 验证预处理结果
        self.assertEqual(len(processed_names), 2)
        self.assertIn("superior", processed_names[0].lower())
        self.assertIn("mountain", processed_names[0].lower())
        self.assertIn("view", processed_names[0].lower())
        
        # 验证特征提取
        self.assertEqual(features["room_type"][0].lower(), "superior")
        self.assertEqual(features["has_view"][0], True)
    
    def test_enhanced_room_matching(self):
        # 创建测试数据
        reference_rooms = pd.DataFrame({
            'hotel_id': ['hotel1', 'hotel1'],
            'room_id': ['room1', 'room2'],
            'room_name': ['Superior Room, Mountain View', 'Deluxe King Room with Balcony'],
            'processed_name': ['superior room mountain view', 'deluxe king room balcony']
        })
        
        supplier_rooms = pd.DataFrame({
            'supplier_room_id': ['sup1', 'sup2', 'sup3'],
            'supplier_room_name': ['Superior Mountain View', 'King Deluxe with Balcony', 'Standard Double Room'],
            'processed_name': ['superior mountain view', 'king deluxe balcony', 'standard double room']
        })
        
        # 使用模拟对象替代实际模型
        mock_model = MagicMock()
        mock_model.predict_similarity.return_