import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_processing.data_processing import preprocess_room_names, enhanced_room_matching
from models.bert_xgb import BertXGBoostRoomMatcher
import pytest

class TestDataProcessing(unittest.TestCase):
    
    def test_preprocess_room_names(self):
        # 测试预处理函数
        room_names = ["Superior Room, Mountain View", "Deluxe King Room with Balcony"]
        processed_names, features = preprocess_room_names(room_names, extract_features=True)
        
        # 验证预处理结果
        self.assertEqual(len(processed_names), 2)
        self.assertIn("superior", processed_names[0])
        self.assertIn("mountain", processed_names[0])
        self.assertIn("view", processed_names[0])
        
        # 验证特征提取
        self.assertEqual(features["room_type"][0], "superior")
        self.assertEqual(features["has_view"][0], True)
        
    @pytest.mark.skipif(not os.path.exists("trained_model/fine_tuned_model.joblib"), 
                       reason="Model file not found")
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
        
        # 初始化模型
        model = BertXGBoostRoomMatcher(
            bert_model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_path="trained_model/fine_tuned_model.joblib", 
            batch_size=32
        )
        
        # 模拟模型行为，避免实际加载模型
        model.bert_model = MockBertModel()
        
        # 执行匹配
        matches = enhanced_room_matching(reference_rooms, supplier_rooms, model, similarity_threshold=0.5)
        
        # 验证匹配结果
        self.assertGreater(len(matches), 0)
        
class MockBertModel:
    def encode(self, texts, batch_size=32):
        # 返回模拟的嵌入向量
        return np.random.rand(len(texts), 384)

if __name__ == '__main__':
    unittest.main() 