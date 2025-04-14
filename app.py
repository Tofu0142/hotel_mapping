from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer
from models.bert_xgb import BertXGBoostRoomMatcher
import os
from data_processing.data_processing import preprocess_room_names, enhanced_room_matching

app = Flask(__name__)

# Load the pre-trained model
model_path = "trained_model/fine_tuned_model.joblib"  # Update with your model path
model = BertXGBoostRoomMatcher(bert_model_name='sentence-transformers/all-MiniLM-L6-v2',model_path=model_path, batch_size=32) 

# 添加错误处理
try:
    model.load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # 提供一个备用方案或友好的错误消息

@app.route('/', methods=['GET'])
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Room Matching API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            h1 { color: #333; }
            pre { background: #f4f4f4; padding: 15px; border-radius: 5px; }
            .endpoint { margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <h1>Room Matching API</h1>
        <p>This API provides room matching functionality between reference and supplier catalogs.</p>
        
        <div class="endpoint">
            <h2>Match Rooms</h2>
            <p><strong>Endpoint:</strong> POST /api/match-rooms</p>
            <p><strong>Description:</strong> Match rooms between reference and supplier catalogs</p>
            <h3>Example Request:</h3>
            <pre>
{
  "referenceCatalog": {
    "propertyId": "hotel123",
    "rooms": [
      {"roomId": "ref001", "roomName": "Superior Room, Mountain View"},
      {"roomId": "ref002", "roomName": "Deluxe King Room with Balcony"}
    ]
  },
  "inputCatalog": {
    "rooms": [
      {"roomId": "sup001", "roomName": "Superior Mountain View"},
      {"roomId": "sup002", "roomName": "King Deluxe with Balcony"},
      {"roomId": "sup003", "roomName": "Standard Double Room"}
    ]
  },
  "similarityThreshold": 0.6,
  "featureWeight": 0.3
}
            </pre>
        </div>
    </body>
    </html>
    """

@app.route('/api/match-rooms', methods=['POST'])
def match_rooms():
    try:
        data = request.get_json()
        
        # Extract data from request
        reference_catalog = data.get('referenceCatalog', {})
        input_catalog = data.get('inputCatalog', {})
        
        # Validate input
        if not reference_catalog or not input_catalog:
            return jsonify({"error": "Missing required data"}), 400
            
        hotel_id = reference_catalog.get('propertyId')
        reference_rooms_data = reference_catalog.get('rooms', [])
        supplier_rooms_data = input_catalog.get('rooms', [])
        
        if not hotel_id or not reference_rooms_data or not supplier_rooms_data:
            return jsonify({"error": "Missing required fields"}), 400
        
        # Convert to DataFrames
        reference_rooms = pd.DataFrame(reference_rooms_data)
        reference_rooms['hotel_id'] = hotel_id
        reference_rooms = reference_rooms.rename(columns={'roomId': 'room_id', 'roomName': 'room_name'})
        
        supplier_rooms = pd.DataFrame(supplier_rooms_data)
        supplier_rooms = supplier_rooms.rename(columns={'roomId': 'supplier_room_id', 'roomName': 'supplier_room_name'})
        
        # Preprocess room names and extract features
        processed_ref_names, ref_features = preprocess_room_names(reference_rooms['room_name'].tolist(), extract_features=True)
        processed_sup_names, sup_features = preprocess_room_names(supplier_rooms['supplier_room_name'].tolist(), extract_features=True)
        
        # Add processed names to DataFrames
        reference_rooms['processed_name'] = processed_ref_names
        supplier_rooms['processed_name'] = processed_sup_names
        
        # Add extracted features to DataFrames
        for feature_name, feature_values in ref_features.items():
            reference_rooms[feature_name] = feature_values
        
        for feature_name, feature_values in sup_features.items():
            supplier_rooms[feature_name] = feature_values
        
        # Perform room matching
        similarity_threshold = data.get('similarityThreshold', 0.8)
        feature_weight = data.get('featureWeight', 0.3)
        
        matches = enhanced_room_matching(
            reference_rooms, 
            supplier_rooms, 
            model, 
            similarity_threshold=similarity_threshold,
            feature_weight=feature_weight
        )
        
        # Format the response
        results = []
        for _, match in matches.iterrows():
            results.append({
                'referenceRoomId': match['reference_room_id'],
                'referenceRoomName': match['reference_room_name'],
                'supplierRoomId': match['supplier_room_id'],
                'supplierRoomName': match['supplier_room_name'],
                'similarityScore': float(match['text_similarity']),
            })
        
        return jsonify({
            'propertyId': hotel_id,
            'matches': results
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 