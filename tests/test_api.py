import unittest
import json
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies before importing app
import pytest
from unittest.mock import patch, MagicMock

# Mock SentenceTransformer
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()

# Mock hotel_mapping.models.bert_xgb
sys.modules['hotel_mapping.models.bert_xgb'] = MagicMock()
mock_model_class = MagicMock()
mock_model = MagicMock()
mock_model.load_model.return_value = None
mock_model.predict_similarity.return_value = [0.9]
mock_model_class.return_value = mock_model
sys.modules['hotel_mapping.models.bert_xgb'].BertXGBoostRoomMatcher = mock_model_class

# Create a proper mock for preprocess_room_names
mock_preprocess = MagicMock()
mock_preprocess.return_value = (
    ["superior mountain view"], 
    {"room_type": ["superior"], "has_view": [True]}
)

# Mock hotel_mapping.data_processing.data_processing
sys.modules['hotel_mapping.data_processing.data_processing'] = MagicMock()
sys.modules['hotel_mapping.data_processing.data_processing'].preprocess_room_names = mock_preprocess
sys.modules['hotel_mapping.data_processing.data_processing'].enhanced_room_matching = MagicMock()

# Now import app
from hotel_mapping.app import app

import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Replace app's model
app.model = mock_model

class TestAPI(unittest.TestCase):
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_index_route(self):
        # Test home route
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    @patch('hotel_mapping.app.enhanced_room_matching')
    @patch('hotel_mapping.app.preprocess_room_names')
    def test_match_rooms_api(self, mock_preprocess, mock_match):
        # Mock preprocessing function
        mock_preprocess.side_effect = [
            (["superior mountain view"], {"room_type": ["superior"], "has_view": [True]}),
            (["king deluxe balcony"], {"room_type": ["deluxe"], "has_balcony": [True]})
        ]
        
        # Mock matching function
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
        
        # Send test request
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
        
        # Verify response
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

@patch('hotel_mapping.app.preprocess_room_names')
@patch('hotel_mapping.app.enhanced_room_matching')
def test_match_rooms(mock_match, mock_preprocess):
    # Set up mocks
    mock_preprocess.side_effect = [
        (["superior mountain view"], {"room_type": ["superior"], "has_view": [True]}),
        (["king deluxe balcony"], {"room_type": ["deluxe"], "has_balcony": [True]})
    ]
    
    mock_match.return_value = pd.DataFrame({
        'reference_room_id': ['ref001'],
        'reference_room_name': ['Superior Room, Mountain View'],
        'supplier_room_id': ['sup001'],
        'supplier_room_name': ['Superior Mountain View'],
        'text_similarity': [0.85]
    })
    
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
    
    response = client.post("/api/match-rooms", json=test_data)
    assert response.status_code == 200
    # Add more assertions to verify response content

if __name__ == '__main__':
    unittest.main() 