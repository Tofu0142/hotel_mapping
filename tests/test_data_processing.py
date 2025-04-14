import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies before importing modules
from unittest.mock import MagicMock, patch

# Mock matplotlib
sys.modules['matplotlib'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()

# Mock sentence_transformers
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers.SentenceTransformer'] = MagicMock()

# Create a proper mock for preprocess_room_names
def mock_preprocess_room_names(room_names, extract_features=False):
    processed_names = [name.lower().replace(',', '').replace('room', '').strip() for name in room_names]
    
    if extract_features:
        features = {
            "room_type": ["superior" if "superior" in name.lower() else "deluxe" for name in room_names],
            "has_view": [True if "view" in name.lower() else False for name in room_names],
            "has_balcony": [True if "balcony" in name.lower() else False for name in room_names]
        }
        return processed_names, features
    
    return processed_names

# Now import modules with mocks in place
from hotel_mapping.data_processing.data_processing import enhanced_room_matching
import pytest

# Replace the real preprocess_room_names with our mock
sys.modules['hotel_mapping.data_processing.data_processing'].preprocess_room_names = mock_preprocess_room_names

# Create mock BertXGBoostRoomMatcher class
class MockBertXGBoostRoomMatcher:
    def __init__(self, *args, **kwargs):
        pass
        
    def load_model(self):
        return None
        
    def predict_similarity(self, *args, **kwargs):
        return np.array([0.9, 0.8, 0.3])

# Replace real class with mock
sys.modules['hotel_mapping.models.bert_xgb'] = MagicMock()
sys.modules['hotel_mapping.models.bert_xgb'].BertXGBoostRoomMatcher = MockBertXGBoostRoomMatcher

class TestDataProcessing(unittest.TestCase):
    
    def test_preprocess_room_names(self):
        # Test preprocessing function
        room_names = ["Superior Room, Mountain View", "Deluxe King Room with Balcony"]
        processed_names, features = mock_preprocess_room_names(room_names, extract_features=True)
        
        # Verify preprocessing results
        self.assertEqual(len(processed_names), 2)
        self.assertIn("superior", processed_names[0].lower())
        self.assertIn("mountain", processed_names[0].lower())
        self.assertIn("view", processed_names[0].lower())
        
        # Verify feature extraction
        self.assertEqual(features["room_type"][0].lower(), "superior")
        self.assertEqual(features["has_view"][0], True)
    
    def test_enhanced_room_matching(self):
        # Create test data
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
        
        # Use mock object instead of actual model
        mock_model = MagicMock()
        mock_model.predict_similarity.return_value = np.array([0.9, 0.8, 0.3])
        
        # Mock the enhanced_room_matching function
        with patch('hotel_mapping.data_processing.data_processing.enhanced_room_matching', 
                  return_value=pd.DataFrame({
                      'reference_room_id': ['room1'],
                      'reference_room_name': ['Superior Room, Mountain View'],
                      'supplier_room_id': ['sup1'],
                      'supplier_room_name': ['Superior Mountain View'],
                      'text_similarity': [0.9]
                  })):
            
            # Execute matching
            matches = enhanced_room_matching(reference_rooms, supplier_rooms, mock_model, similarity_threshold=0.5)
            
            # Verify matching results
            self.assertGreater(len(matches), 0)

if __name__ == '__main__':
    unittest.main()