# Hotel Room Mapping API

A machine learning-powered API for matching hotel room descriptions between reference catalogs and supplier catalogs. This project uses a combination of NLP techniques and machine learning to accurately map similar room types across different naming conventions.

## Overview

The Hotel Room Mapping API provides an intelligent solution for the hospitality industry to standardize room descriptions across multiple suppliers. By leveraging BERT embeddings and XGBoost, the system can identify matching rooms even when the naming conventions differ significantly.

## Features

- **Intelligent Room Matching**: Uses a hybrid approach combining semantic similarity and feature extraction
- **High Performance**: Optimized for both accuracy and speed with batch processing
- **REST API**: Simple integration with existing systems via RESTful endpoints
- **Explainable Results**: Provides similarity scores and feature importance analysis
- **Scalable Architecture**: Containerized deployment with Cloud Run support

## Architecture

The system consists of several key components:

1. **Data Processing Pipeline**: Cleans and normalizes room descriptions, extracts structured features
2. **BERT+XGBoost Model**: Combines semantic embeddings with feature engineering for accurate matching
3. **FastAPI Service**: Provides RESTful endpoints for room matching
4. **CI/CD Pipeline**: Automated testing and deployment to Google Cloud Platform

## API Usage

### Match Rooms Endpoint

**Endpoint**: `POST /api/match-rooms`

**Request Format**:

```
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
```

**Response Format**:
```json
{
  "propertyId": "hotel123",
  "matches": [
    {
      "referenceRoomId": "ref001",
      "referenceRoomName": "Superior Room, Mountain View",
      "supplierRoomId": "sup001",
      "supplierRoomName": "Superior Mountain View",
      "similarityScore": 0.85
    },
    {
      "referenceRoomId": "ref002",
      "referenceRoomName": "Deluxe King Room with Balcony",
      "supplierRoomId": "sup002",
      "supplierRoomName": "King Deluxe with Balcony",
      "similarityScore": 0.78
    }
  ]
}
```

## Technical Implementation

### Data Processing

The data processing pipeline includes:

1. **Text Preprocessing**:
   - Lowercase conversion
   - Punctuation removal
   - Stopword removal
   - Simple lemmatization for common hotel terms

2. **Feature Extraction**:
   - Room type identification (deluxe, superior, standard, etc.)
   - Bed type detection (king, queen, twin, etc.)
   - View type identification (sea, mountain, garden, etc.)
   - Capacity extraction (number of people, beds)
   - Amenity detection (balcony, terrace, etc.)

### Model Architecture

The room matching model uses a hybrid approach:

1. **BERT Embeddings**: Captures semantic meaning of room descriptions using pre-trained language models
2. **Feature Engineering**: Extracts structured features from room descriptions
3. **XGBoost Classifier**: Combines semantic similarity with feature matching to predict room matches

### Model Training Process

1. **Data Collection**: Gathering labeled pairs of matching rooms from different suppliers
2. **Feature Generation**: Creating feature vectors from room pairs (both matching and non-matching)
3. **Model Training**: Training XGBoost on the combined features
4. **Hyperparameter Tuning**: Optimizing model parameters for best performance
5. **Evaluation**: Measuring precision, recall, F1 score, and ROC AUC

### Model Explainability

The system provides several explainability features:

1. **Feature Importance Analysis**: Identifies which features contribute most to matching decisions
2. **Correlation Analysis**: Detects redundant features
3. **Visualization**: Generates ROC curves and precision-recall curves
4. **Error Analysis**: Examines false positives and false negatives

## Deployment

The application is containerized using Docker and deployed to Google Cloud Run:

1. **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment
2. **Containerization**: Docker image built and pushed to Google Artifact Registry
3. **Cloud Deployment**: Serverless deployment on Google Cloud Run
4. **Scaling**: Automatic scaling based on request load

## Development Setup

### Prerequisites

- Python 3.9+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Tofu0142/hotel_mapping.git
cd hotel_mapping
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application locally:
```bash
uvicorn hotel_mapping.app:app --reload
```

### Testing

Run the test suite:
```bash
pytest
```

## Future Improvements

- **Multilingual Support**: Extend the model to handle room descriptions in multiple languages
- **Incremental Learning**: Implement feedback loop to improve model over time
- **Additional Features**: Support for more room attributes and amenities
- **Performance Optimization**: Further optimize for speed and resource usage

## License

[MIT License](LICENSE)