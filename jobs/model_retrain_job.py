#!/usr/bin/env python3
"""
Model Retraining Job
Updates AI models with new data to improve accuracy
Schedule: Weekly on Sunday at 1 AM
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from models.stock_predictor import StockPredictor
from models.sentiment_analyzer import SentimentAnalyzer
from models.ner_trainer import CompanyNERTrainer
from data.stock_processor import StockDataProcessor
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MODEL_RETRAIN] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def model_retraining_job():
    """
    Main model retraining job
    Updates ML models with new data and validates performance
    """
    try:
        start_time = datetime.now()
        logger.info("Starting Model Retraining Job...")
        
        # Create backup directory
        backup_dir = Config.MODELS_DIR / "backups" / start_time.strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Track retraining results
        retraining_results = {
            'timestamp': start_time.isoformat(),
            'models_retrained': [],
            'performance_metrics': {},
            'backup_location': str(backup_dir),
            'success': False
        }
        
        # 1. STOCK PREDICTOR MODEL RETRAINING
        logger.info("Starting Stock Predictor Model Retraining...")
        
        try:
            # Backup existing model
            existing_model_path = Config.MODELS_DIR / "trained" / "stock_predictor.joblib"
            if existing_model_path.exists():
                backup_model_path = backup_dir / "stock_predictor_backup.joblib"
                joblib.dump(joblib.load(existing_model_path), backup_model_path)
                logger.info(f"Backed up existing model to: {backup_model_path}")
            
            # Load recent data for retraining (last 90 days)
            engine = create_engine(Config.get_database_url())
            
            # Get processed messages with sentiment data
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        pm.from_name,
                        pm.sentiment_score,
                        pm.confidence,
                        pm.company_mentions,
                        pm.timestamp_formatted,
                        sd.symbol,
                        sd.close as current_price,
                        sd.date as stock_date
                    FROM processed_messages pm
                    JOIN stock_data sd ON 
                        JSON_EXTRACT_PATH_TEXT(pm.company_mentions::json, '0', 'ticker') = sd.symbol
                        AND DATE(pm.timestamp_formatted) <= sd.date
                        AND sd.date <= DATE(pm.timestamp_formatted) + INTERVAL '7 days'
                    WHERE 
                        pm.is_financial = true 
                        AND pm.timestamp_formatted >= NOW() - INTERVAL '90 days'
                        AND pm.sentiment_score IS NOT NULL
                    ORDER BY pm.timestamp_formatted DESC
                    LIMIT 10000
                """))
                
                training_data = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if len(training_data) >= 100:  # Minimum data requirement
                logger.info(f"Loaded {len(training_data)} training samples")
                
                # Initialize predictor
                predictor = StockPredictor()
                
                # Prepare features and targets
                features_df = predictor.create_features(
                    stock_data=training_data.rename(columns={'current_price': 'close', 'stock_date': 'date'}),
                    sentiment_data=training_data.rename(columns={'timestamp_formatted': 'date'}),
                    prediction_horizon=3
                )
                
                if len(features_df) >= 50:  # Minimum for training
                    X = features_df.drop(columns=['symbol', 'date', 'target', 'target_return'], errors='ignore')
                    y = features_df['target']
                    
                    # Clean features
                    X = predictor._clean_features(X)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Train new model
                    logger.info("Training new Stock Predictor model...")
                    training_results = predictor.train_model(
                        X_train, y_train, 
                        tune_hyperparameters=True,
                        dates=features_df.loc[X_train.index, 'date'] if 'date' in features_df.columns else None
                    )
                    
                    # Validate performance
                    predictions = predictor.predict(X_test)['predictions']
                    
                    # Calculate metrics
                    test_accuracy = accuracy_score(y_test, predictions)
                    test_f1 = f1_score(y_test, predictions, average='weighted')
                    
                    logger.info("Model Performance:")
                    logger.info(f"   Test Accuracy: {test_accuracy:.3f}")
                    logger.info(f"   Test F1-Score: {test_f1:.3f}")
                    
                    # Performance thresholds (from document requirements)
                    baseline_accuracy = 0.78
                    baseline_f1 = 0.75
                    
                    # Deploy new model if performance improved
                    if test_accuracy >= baseline_accuracy and test_f1 >= baseline_f1:
                        predictor.save_model(str(existing_model_path))
                        
                        retraining_results['models_retrained'].append('stock_predictor')
                        retraining_results['performance_metrics']['stock_predictor'] = {
                            'accuracy': test_accuracy,
                            'f1_score': test_f1,
                            'training_samples': len(X_train),
                            'test_samples': len(X_test)
                        }
                        
                        logger.info("Stock Predictor model deployed successfully!")
                    else:
                        logger.warning(f"New model performance below baseline (Acc: {test_accuracy:.3f} < {baseline_accuracy}, F1: {test_f1:.3f} < {baseline_f1})")
                        logger.info("Keeping existing model")
                        
                        # Restore backup if performance is worse
                        if existing_model_path.exists() and backup_model_path.exists():
                            joblib.dump(joblib.load(backup_model_path), existing_model_path)
                
                else:
                    logger.warning("Insufficient feature data for stock predictor training")
            
            else:
                logger.warning("Insufficient training data for stock predictor")
                
        except Exception as e:
            logger.error(f"Stock Predictor retraining failed: {e}")
        
        # 2. NER MODEL RETRAINING
        logger.info("Starting NER Model Retraining...")
        
        try:
            # Backup existing NER model
            ner_model_path = Config.MODELS_DIR / "trained" / "company_ner"
            if ner_model_path.exists():
                backup_ner_path = backup_dir / "company_ner_backup"
                import shutil
                shutil.copytree(ner_model_path, backup_ner_path)
                logger.info(f"Backed up existing NER model to: {backup_ner_path}")
            
            # Initialize NER trainer
            ner_trainer = CompanyNERTrainer()
            
            # Load existing model or create new one
            if not ner_trainer.load_existing_model():
                ner_trainer.create_blank_model()
            
            # Load company data for training
            with open(Config.KEYWORDS_FILE, 'r') as f:
                company_data = json.load(f)
            
            # Prepare training data
            training_data = ner_trainer.prepare_training_data_from_companies(company_data)
            
            if len(training_data) >= 20:
                logger.info(f"Training NER model with {len(training_data)} examples...")
                
                # Train model
                training_stats = ner_trainer.train_model(
                    training_data, 
                    iterations=20,  # Reduced for weekly updates
                    dropout=0.5
                )
                
                # Save improved model
                ner_trainer.save_model(str(ner_model_path))
                
                retraining_results['models_retrained'].append('company_ner')
                retraining_results['performance_metrics']['company_ner'] = {
                    'final_loss': training_stats['final_loss'],
                    'training_examples': len(training_data),
                    'iterations': 20
                }
                
                logger.info(f"NER model retrained successfully! Final loss: {training_stats['final_loss']:.4f}")
                
            else:
                logger.warning("Insufficient NER training data")
                
        except Exception as e:
            logger.error(f"NER model retraining failed: {e}")
        
        # 3. SENTIMENT MODEL CALIBRATION
        logger.info("Calibrating Sentiment Analysis...")
        
        try:
            # Load recent sentiment predictions vs actual outcomes
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        pm.sentiment_score,
                        pm.sentiment,
                        pm.confidence,
                        CASE 
                            WHEN sd_future.close > sd_current.close THEN 'positive'
                            WHEN sd_future.close < sd_current.close THEN 'negative'
                            ELSE 'neutral'
                        END as actual_outcome
                    FROM processed_messages pm
                    JOIN stock_data sd_current ON 
                        JSON_EXTRACT_PATH_TEXT(pm.company_mentions::json, '0', 'ticker') = sd_current.symbol
                        AND DATE(pm.timestamp_formatted) = sd_current.date
                    JOIN stock_data sd_future ON 
                        sd_current.symbol = sd_future.symbol
                        AND sd_future.date = sd_current.date + INTERVAL '3 days'
                    WHERE 
                        pm.is_financial = true 
                        AND pm.timestamp_formatted >= NOW() - INTERVAL '60 days'
                        AND pm.sentiment IS NOT NULL
                    LIMIT 5000
                """))
                
                sentiment_validation = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if len(sentiment_validation) >= 50:
                # Calculate sentiment accuracy
                correct_predictions = (sentiment_validation['sentiment'] == sentiment_validation['actual_outcome']).sum()
                sentiment_accuracy = correct_predictions / len(sentiment_validation)
                
                logger.info(f"Sentiment Analysis Accuracy: {sentiment_accuracy:.3f}")
                
                retraining_results['performance_metrics']['sentiment_analysis'] = {
                    'accuracy': sentiment_accuracy,
                    'validation_samples': len(sentiment_validation),
                    'correct_predictions': int(correct_predictions)
                }
                
                retraining_results['models_retrained'].append('sentiment_calibration')
                
                # Target accuracy from document: 85%
                if sentiment_accuracy >= 0.85:
                    logger.info("Sentiment analysis meets target accuracy!")
                else:
                    logger.info(f"Sentiment accuracy improving (current: {sentiment_accuracy:.1%}, target: 85%)")
                    
        except Exception as e:
            logger.warning(f"Sentiment calibration failed: {e}")
        
        # 4. Save retraining report
        report_file = Config.DATA_DIR / "processed" / f"model_retrain_report_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        retraining_results['success'] = len(retraining_results['models_retrained']) > 0
        
        with open(report_file, 'w') as f:
            json.dump(retraining_results, f, indent=2, default=str)
        
        # Log final results
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Model Retraining Job completed!")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Models retrained: {len(retraining_results['models_retrained'])}")
        
        for model in retraining_results['models_retrained']:
            logger.info(f"   Model: {model}")
        
        logger.info(f"   Performance report: {report_file}")
        logger.info(f"   Model backups: {backup_dir}")
        
        if not retraining_results['success']:
            logger.warning("No models were successfully retrained")
        
    except Exception as e:
        logger.error(f"Model Retraining Job failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # For testing purposes
    model_retraining_job()