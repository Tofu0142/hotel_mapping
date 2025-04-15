from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append('/path/to/your/project')

from hotel_mapping.models.bert_xgb import BertXGBoostRoomMatcher
from hotel_mapping.pipelines.data_loader import load_reference_rooms, load_supplier_rooms, load_labeled_matches

# Default arguments
default_args = {
    'owner': 'hotel_mapping',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'room_matcher_pipeline',
    default_args=default_args,
    description='Hotel room matching pipeline',
    schedule_interval=timedelta(days=7),  # Weekly
    catchup=False
)

# Define tasks
def load_and_preprocess_data(**kwargs):
    """Load and preprocess hotel room data"""
    # Load data
    reference_rooms = load_reference_rooms()
    supplier_rooms = load_supplier_rooms()
    labeled_matches = load_labeled_matches()
    
    # Preprocess room names
    reference_rooms['processed_name'] = reference_rooms['room_name'].str.lower()
    supplier_rooms['processed_name'] = supplier_rooms['supplier_room_name'].str.lower()
    
    # Save to temp location for next tasks
    reference_rooms.to_csv('/tmp/reference_rooms.csv', index=False)
    supplier_rooms.to_csv('/tmp/supplier_rooms.csv', index=False)
    if labeled_matches is not None:
        labeled_matches.to_csv('/tmp/labeled_matches.csv', index=False)
    
    return {'reference_count': len(reference_rooms), 
            'supplier_count': len(supplier_rooms),
            'labeled_count': len(labeled_matches) if labeled_matches is not None else 0}

def train_model(**kwargs):
    """Train the BERT-XGBoost model"""
    # Load preprocessed data
    reference_rooms = pd.read_csv('/tmp/reference_rooms.csv')
    supplier_rooms = pd.read_csv('/tmp/supplier_rooms.csv')
    
    try:
        labeled_matches = pd.read_csv('/tmp/labeled_matches.csv')
    except:
        labeled_matches = None
    
    # Initialize model
    model = BertXGBoostRoomMatcher(
        model_path='/opt/airflow/models/bert_xgb_room_matcher.joblib'
    )
    
    # Train model
    model.train(reference_rooms, supplier_rooms, labeled_matches)
    
    return {'model_path': model.model_path}

def evaluate_model(**kwargs):
    """Evaluate model performance on test data"""
    # Load test data
    test_reference = pd.read_csv('/opt/airflow/data/test_reference_rooms.csv')
    test_supplier = pd.read_csv('/opt/airflow/data/test_supplier_rooms.csv')
    test_matches = pd.read_csv('/opt/airflow/data/test_matches.csv')
    
    # Preprocess
    test_reference['processed_name'] = test_reference['room_name'].str.lower()
    test_supplier['processed_name'] = test_supplier['supplier_room_name'].str.lower()
    
    # Load model
    model = BertXGBoostRoomMatcher(
        model_path='/opt/airflow/models/bert_xgb_room_matcher.joblib'
    )
    model.load_model()
    
    # Predict matches
    predicted_matches = model.predict(test_reference, test_supplier)
    
    # Calculate metrics
    # (Simplified - in practice you'd want more comprehensive evaluation)
    correct_matches = 0
    for _, pred_row in predicted_matches.iterrows():
        for _, true_row in test_matches.iterrows():
            if (pred_row['reference_room_id'] == true_row['reference_room_id'] and
                pred_row['supplier_room_id'] == true_row['supplier_room_id']):
                correct_matches += 1
                break
    
    precision = correct_matches / len(predicted_matches) if len(predicted_matches) > 0 else 0
    recall = correct_matches / len(test_matches) if len(test_matches) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Log metrics
    with open('/opt/airflow/logs/model_metrics.txt', 'w') as f:
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
    
    return {'precision': precision, 'recall': recall, 'f1': f1}

def deploy_model(**kwargs):
    """Deploy the model to production"""
    # Copy model to production location
    import shutil
    shutil.copy(
        '/opt/airflow/models/bert_xgb_room_matcher.joblib',
        '/opt/airflow/production/bert_xgb_room_matcher.joblib'
    )
    
    # Update version file
    with open('/opt/airflow/production/model_version.txt', 'w') as f:
        f.write(f"Model deployed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"F1 Score: {kwargs['ti'].xcom_pull(task_ids='evaluate_model')['f1']}")
    
    return {'deployment_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Create tasks
load_data_task = PythonOperator(
    task_id='load_and_preprocess_data',
    python_callable=load_and_preprocess_data,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

deploy_model_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Define task dependencies
load_data_task >> train_model_task >> evaluate_model_task >> deploy_model_task
