# hotel_mapping/data/data_loader.py
import pandas as pd
import os

def load_reference_rooms(data_path=None):
    """Load reference hotel rooms data"""
    if data_path is None:
        data_path = os.environ.get('REFERENCE_ROOMS_PATH', '/opt/airflow/data/reference_rooms.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        # Return empty DataFrame if file doesn't exist
        print(f"Warning: Reference rooms file not found at {data_path}")
        return pd.DataFrame(columns=['hotel_id', 'room_id', 'room_name'])
    
    return pd.read_csv(data_path)

def load_supplier_rooms(data_path=None):
    """Load supplier hotel rooms data"""
    if data_path is None:
        data_path = os.environ.get('SUPPLIER_ROOMS_PATH', '/opt/airflow/data/supplier_rooms.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        # Return empty DataFrame if file doesn't exist
        print(f"Warning: Supplier rooms file not found at {data_path}")
        return pd.DataFrame(columns=['supplier_id', 'supplier_room_id', 'supplier_room_name'])
    
    return pd.read_csv(data_path)

def load_labeled_matches(data_path=None):
    """Load labeled matches data"""
    if data_path is None:
        data_path = os.environ.get('LABELED_MATCHES_PATH', '/opt/airflow/data/labeled_matches.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        # Return None if file doesn't exist
        print(f"Warning: Labeled matches file not found at {data_path}")
        return None
    
    return pd.read_csv(data_path)