from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import pandas as pd
from sentence_transformers import SentenceTransformer
from models.bert_xgb import BertXGBoostRoomMatcher
import os
from data_processing.data_processing import preprocess_room_names, enhanced_room_matching
import uvicorn
import traceback

# Define Pydantic models for request validation
class Room(BaseModel):
    roomId: str
    roomName: str

class SupplierRoom(BaseModel):
    roomId: str
    roomName: str

class ReferenceCatalog(BaseModel):
    propertyId: str
    rooms: List[Room]

class InputCatalog(BaseModel):
    rooms: List[SupplierRoom]

class MatchRoomsRequest(BaseModel):
    referenceCatalog: ReferenceCatalog
    inputCatalog: InputCatalog
    similarityThreshold: Optional[float] = 0.8
    featureWeight: Optional[float] = 0.3

class MatchResult(BaseModel):
    referenceRoomId: str
    referenceRoomName: str
    supplierRoomId: str
    supplierRoomName: str
    similarityScore: float

class MatchResponse(BaseModel):
    propertyId: str
    matches: List[MatchResult]

app = FastAPI(title="Room Matching API", 
              description="API for matching rooms between reference and supplier catalogs")

# Load the pre-trained model
model_path = "trained_model/fine_tuned_model.joblib"  # Update with your model path
model = BertXGBoostRoomMatcher(bert_model_name='sentence-transformers/all-MiniLM-L6-v2',model_path=model_path, batch_size=32) 

# Add error handling
try:
    model.load_model()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Provide a fallback or friendly error message

@app.get("/", response_class=HTMLResponse)
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

@app.post("/api/match-rooms", response_model=MatchResponse)
async def match_rooms(request: MatchRoomsRequest):
    try:
        # Extract data from request
        reference_catalog = request.referenceCatalog
        input_catalog = request.inputCatalog
        
        hotel_id = reference_catalog.propertyId
        reference_rooms_data = [{"roomId": room.roomId, "roomName": room.roomName} for room in reference_catalog.rooms]
        supplier_rooms_data = [{"roomId": room.roomId, "roomName": room.roomName} for room in input_catalog.rooms]
        
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
        similarity_threshold = request.similarityThreshold
        feature_weight = request.featureWeight
        
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
            results.append(MatchResult(
                referenceRoomId=match['reference_room_id'],
                referenceRoomName=match['reference_room_name'],
                supplierRoomId=match['supplier_room_id'],
                supplierRoomName=match['supplier_room_name'],
                similarityScore=float(match['text_similarity']),
            ))
        
        return MatchResponse(
            propertyId=hotel_id,
            matches=results
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())  # Print full stack trace
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 