import unittest
import json
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 在导入 app 之前模拟依赖
import pytest
from unittest.mock import patch, MagicMock

# 模拟 SentenceTransformer
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()


mock_model = MagicMock()
mock_model.load_model.return_value = None
mock_model.predict_similarity.return_value = [0.9]


from hotel_mapping.app import app

import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


app.model = mock_model

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_index_route(self):
        # 测试首页路由
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    @patch('hotel_mapping.app.enhanced_room_matching')
    @patch('hotel_mapping.app.preprocess_room_names')
    def test_match_rooms_api(self, mock_preprocess, mock_match):
        # 模拟预处理函数
        mock_preprocess.side_effect = [
            (["superior mountain view"], {"room_type": ["superior"], "has_view": [True]}),
            (["king deluxe balcony"], {"room_type": ["deluxe"], "has_balcony": [True]})
        ]
        
        # 模拟匹配函数
        mock_match.return_value = pd.DataFrame({
            'reference_hotel_id': ['hotel123'],
            'reference_room_id': ['ref001'],
            'reference_room_name': ['Superior Room, Mountain View'],
            'reference_processed_name': ['superior mountain view'],
            'supplier_room_id': ['sup001'],
            'supplier_room_name': ['Superior Mountain View'],
            'supplier_processed_name': ['superior mountain view'],
            'text_similarity': [0.85]
        })
        
        # 发送测试请求
        payload = {
            "referenceCatalog": {
                "propertyId": "hotel123",
                "rooms": [
                    {"roomId": "ref001", "roomName": "Superior Room, Mountain View"}
                ]
            },
            "inputCatalog": {
                "rooms": [
                    {"roomId": "sup001", "roomName": "Superior Mountain View"}
                ]
            }
        }
        
        response = self.client.post('/api/match-rooms', 
                                json=payload)
        
        # 验证响应
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['propertyId'], 'hotel123')
        self.assertEqual(len(data['matches']), 1)
        self.assertEqual(data['matches'][0]['referenceRoomId'], 'ref001')
        self.assertEqual(data['matches'][0]['supplierRoomId'], 'sup001')

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_match_rooms():
    test_data = {
        "referenceCatalog": {
            "propertyId": "hotel123",
            "rooms": [
                {"roomId": "ref001", "roomName": "Superior Room, Mountain View"}
            ]
        },
        "inputCatalog": {
            "rooms": [
                {"roomId": "sup001", "roomName": "Superior Mountain View"}
            ]
        },
        "similarityThreshold": 0.6,
        "featureWeight": 0.3
    }
    
    # 这里可能需要模拟模型的行为，取决于你的应用架构
    with patch('hotel_mapping.app.enhanced_room_matching') as mock_match:
        mock_match.return_value = pd.DataFrame({
            'reference_room_id': ['ref001'],
            'reference_room_name': ['Superior Room, Mountain View'],
            'supplier_room_id': ['sup001'],
            'supplier_room_name': ['Superior Mountain View'],
            'text_similarity': [0.85]
        })
        response = client.post("/api/match-rooms", json=test_data)
        assert response.status_code == 200
        # 添加更多断言来验证响应内容

if __name__ == '__main__':
    unittest.main() 