import pandas as pd

def evaluate_model(matcher, labeled_data, threshold=0.5):
    """Evaluate model performance using labeled data"""
    print("Starting model evaluation...")
    
    # Prepare evaluation data
    eval_data = []
    
    # Make predictions for each labeled sample
    for i, row in labeled_data.iterrows():
        ref_id = row['reference_room_id']
        sup_id = row['supplier_room_id']
        true_label = row['label']
        
        # Get room names directly from labeled data
        ref_name = row['reference_processed_name']
        sup_name = row['supplier_processed_name']
        
        # Use model to predict match probability
        # Get BERT embeddings
        ref_embedding = matcher.get_bert_embeddings([ref_name])[0]
        sup_embedding = matcher.get_bert_embeddings([sup_name])[0]
        
        # Extract structured features
        ref_feature = matcher.extract_features(ref_name)
        sup_feature = matcher.extract_features(sup_name)
        
        # Create feature pair
        features = matcher.create_pair_features(
            ref_embedding, sup_embedding,
            ref_feature, sup_feature,
            ref_name, sup_name
        )
        
        # Apply feature mask (if available)
        if hasattr(matcher, 'feature_mask') and matcher.feature_mask is not None:
            features = features[matcher.feature_mask]
        
        # Predict
        import xgboost as xgb
        import numpy as np
        dmatrix = xgb.DMatrix(np.array([features]))
        prob = float(matcher.xgb_model.predict(dmatrix)[0])
        
        # Determine predicted label
        pred_label = 1 if prob >= threshold else 0
        
        # Add to evaluation data
        eval_data.append({
            'reference_id': ref_id,
            'reference_name': row['reference_room_name'],
            'supplier_id': sup_id,
            'supplier_name': row['supplier_room_name'],
            'true_label': true_label,
            'pred_prob': prob,
            'pred_label': pred_label,
            'correct': true_label == pred_label
        })
    
    # Convert to DataFrame
    eval_df = pd.DataFrame(eval_data)
    
    # Calculate evaluation metrics
    total = len(eval_df)
    correct = eval_df['correct'].sum()
    accuracy = correct / total
    
    # Calculate precision, recall and F1 score
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        eval_df['true_label'], 
        eval_df['pred_label'], 
        average='binary'
    )
    
    # Print evaluation results
    print("\nModel evaluation results:")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    
    # Analyze incorrect predictions
    errors = eval_df[eval_df['correct'] == False]
    false_positives = errors[errors['pred_label'] == 1]
    false_negatives = errors[errors['pred_label'] == 0]
    
    print(f"\nFalse positives (predicted match but actually not matching): {len(false_positives)}")
    print(f"False negatives (predicted no match but actually matching): {len(false_negatives)}")
    
    # Show some error examples
    if len(false_positives) > 0:
        print("\nFalse positive examples:")
        for i, row in false_positives.head(3).iterrows():
            print(f"Reference room: {row['reference_name']}")
            print(f"Supplier room: {row['supplier_name']}")
            print(f"Prediction probability: {row['pred_prob']:.4f}")
            print("---")
    
    if len(false_negatives) > 0:
        print("\nFalse negative examples:")
        for i, row in false_negatives.head(3).iterrows():
            print(f"Reference room: {row['reference_name']}")
            print(f"Supplier room: {row['supplier_name']}")
            print(f"Prediction probability: {row['pred_prob']:.4f}")
            print("---")
    
    # Plot ROC curve and PR curve
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        # ROC curve
        fpr, tpr, _ = roc_curve(eval_df['true_label'], eval_df['pred_prob'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        # PR curve
        precision_curve, recall_curve, _ = precision_recall_curve(eval_df['true_label'], eval_df['pred_prob'])
        pr_auc = auc(recall_curve, precision_curve)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png')
        print("\nEvaluation curves saved as 'model_evaluation.png'")
    except Exception as e:
        print(f"Unable to plot evaluation curves: {e}")
    
    # Return evaluation metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc if 'roc_auc' in locals() else None,
        'pr_auc': pr_auc if 'pr_auc' in locals() else None
    }
    
    return metrics, eval_df