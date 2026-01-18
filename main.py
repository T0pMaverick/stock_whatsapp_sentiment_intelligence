#!/usr/bin/env python3

import sys
import os
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from data.whatsapp_parser import WhatsAppDataParser
from data.stock_processor import StockDataProcessor
from data.quality_validator import DataQualityValidator
from processing.text_preprocessor import MessagePreprocessor
from models.ner_trainer import CompanyNERTrainer
from models.sentiment_analyzer import SentimentAnalyzer
from models.stock_predictor import StockPredictor
from models.prediction_reasoning import PredictionReasoning

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def save_processed_messages_to_db(financial_messages: pd.DataFrame) -> dict:
    """
    Save processed messages with sentiment analysis to PostgreSQL database
    
    Args:
        financial_messages: DataFrame with processed messages and sentiment analysis
        
    Returns:
        Dictionary with save statistics
    """
    logger.info("Saving processed messages to database...")
    
    stats = {
        'total_messages': len(financial_messages),
        'saved': 0,
        'updated': 0,
        'failed': 0,
        'errors': []
    }
    
    if financial_messages.empty:
        logger.warning("No financial messages to save")
        return stats
    
    try:
        # Create database engine
        engine = create_engine(Config.get_database_url())
        
        with engine.connect() as conn:
            for idx, row in financial_messages.iterrows():
                try:
                    # Prepare company_mentions as JSON string
                    company_mentions_json = json.dumps(
                        row.get('company_mentions', [])
                    ) if row.get('company_mentions') else json.dumps([])
                    
                    # Prepare record for insertion
                    record = {
                        'message_id': str(row.get('message_id', row.get('id', idx))),
                        'message_body': str(row.get('message_body', '')),
                        'cleaned_text': str(row.get('cleaned_text', '')),
                        'from_name': str(row.get('from_name', 'Unknown')),
                        'timestamp_formatted': row.get('timestamp_formatted', datetime.now()),
                        'group_name': str(row.get('group_name', 'Unknown')),
                        'sentiment': str(row.get('sentiment', 'neutral')),
                        'sentiment_score': float(row.get('sentiment_score', 0.0)),
                        'confidence': float(row.get('confidence', 0.0)),
                        'is_financial': True,
                        'company_mentions': company_mentions_json
                    }
                    
                    # Insert or update using ON CONFLICT
                    result = conn.execute(text("""
                        INSERT INTO processed_messages 
                        (message_id, message_body, cleaned_text, from_name, timestamp_formatted,
                         group_name, sentiment, sentiment_score, confidence, is_financial, company_mentions)
                        VALUES 
                        (:message_id, :message_body, :cleaned_text, :from_name, :timestamp_formatted,
                         :group_name, :sentiment, :sentiment_score, :confidence, :is_financial, 
                         CAST(:company_mentions AS jsonb))
                        ON CONFLICT (message_id) DO UPDATE SET
                            message_body = EXCLUDED.message_body,
                            cleaned_text = EXCLUDED.cleaned_text,
                            sentiment = EXCLUDED.sentiment,
                            sentiment_score = EXCLUDED.sentiment_score,
                            confidence = EXCLUDED.confidence,
                            company_mentions = EXCLUDED.company_mentions,
                            processed_at = CURRENT_TIMESTAMP
                        RETURNING (xmax = 0) AS inserted
                    """), record)
                    
                    # Check if it was an insert or update
                    inserted = result.fetchone()[0]
                    if inserted:
                        stats['saved'] += 1
                    else:
                        stats['updated'] += 1
                    
                    # Log progress every 100 messages
                    if (stats['saved'] + stats['updated']) % 100 == 0:
                        logger.info(f"Processed {stats['saved'] + stats['updated']}/{len(financial_messages)} messages...")
                    
                except Exception as e:
                    stats['failed'] += 1
                    error_msg = f"Error saving message {idx}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)
                    continue
            
            # Commit all changes
            conn.commit()
            
        logger.info(f"✅ Database save complete: {stats['saved']} saved, {stats['updated']} updated, {stats['failed']} failed")
        
    except SQLAlchemyError as e:
        logger.error(f"Database error: {e}")
        stats['errors'].append(f"Database connection error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error saving to database: {e}")
        stats['errors'].append(f"Unexpected error: {str(e)}")
    
    return stats

def aggregate_sentiment_data(financial_messages: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentiment data by company and date"""
    sentiment_data = []
    
    # Get unique companies mentioned
    all_companies = set()
    for _, row in financial_messages.iterrows():
        if row.get('company_mentions'):
            for mention in row['company_mentions']:
                all_companies.add(mention['ticker'])
    
    # Aggregate sentiment by company and date
    for company in all_companies:
        company_messages = financial_messages[
            financial_messages['company_mentions'].apply(
                lambda mentions: any(mention['ticker'] == company for mention in mentions)
            )
        ].copy()
        
        if not company_messages.empty:
            # Use timestamp_formatted instead of timestamp_dt
            date_column = 'timestamp_dt' if 'timestamp_dt' in company_messages.columns else 'timestamp_formatted'
            
            daily_sentiment = company_messages.groupby(
                pd.to_datetime(company_messages[date_column]).dt.date
            ).agg({
                'sentiment_score': ['mean', 'std', 'count'],
                'confidence': 'mean'
            }).reset_index()
            
            daily_sentiment.columns = ['date', 'avg_sentiment', 'sentiment_std', 'message_count', 'avg_confidence']
            daily_sentiment['symbol'] = company
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            sentiment_data.append(daily_sentiment)
    
    if sentiment_data:
        return pd.concat(sentiment_data, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    """Main execution function"""
    logger.info("Starting Sentiment Stock Agent with Full Pipeline")
    
    # Create necessary directories and setup early
    Config.create_directories()
    output_dir = Config.DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Load WhatsApp data
        logger.info("Step 1: Loading WhatsApp data...")
        with WhatsAppDataParser() as parser:
            messages = parser.get_recent_messages(days=30, group_name="Wealth Builders")
        logger.info(f"Loaded {len(messages)} messages")
        
        # 2. Load stock data  
        logger.info("Step 2: Loading stock data...")
        stock_processor = StockDataProcessor()
        
        # Use Config.KEYWORDS_FILE instead of hardcoded path
        with open(Config.KEYWORDS_FILE, 'r') as f:
            symbols_to_load = [item['ticker'] for item in json.load(f)]
            
        stock_data = {}
        
        for symbol in symbols_to_load:
            try:
                data = stock_processor.fetch_ohlcv_data(symbol, n_bars=100)
                if data is not None:
                    data = stock_processor.add_technical_indicators(data)
                    stock_data[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                
        logger.info(f"Loaded stock data for {len(stock_data)} symbols")
        
        # 3. Validate data quality
        logger.info("Step 3: Validating data quality...")
        validator = DataQualityValidator()
        whatsapp_quality = validator.validate_whatsapp_data(messages)
        logger.info(f"WhatsApp data quality score: {whatsapp_quality['quality_score']:.2f}/100")
        
        # 4. Preprocess messages
        logger.info("Step 4: Preprocessing messages...")
        preprocessor = MessagePreprocessor()
        processed_messages = preprocessor.process_messages(messages)
        logger.info(f"Processed {len(processed_messages)} messages")
        financial_messages = processed_messages[processed_messages.get('is_financial', False)]
        logger.info(f"Found {len(financial_messages)} financial messages")
        
        # 5. Train NER model
        logger.info("Step 5: Training NER model...")
        try:
            ner_trainer = CompanyNERTrainer()
            
            with open(Config.KEYWORDS_FILE, 'r') as f:
                company_data = json.load(f)
            
            if not ner_trainer.load_existing_model():
                ner_trainer.create_blank_model()
            
            training_data = ner_trainer.prepare_training_data_from_companies(company_data)
            training_stats = ner_trainer.train_model(training_data)
            logger.info(f"NER training completed. Final loss: {training_stats['final_loss']:.4f}")
            
            # Save NER model
            ner_save_path = Config.MODELS_DIR / "trained" / "company_ner"
            ner_trainer.save_model(str(ner_save_path))
        except Exception as e:
            logger.warning(f"NER training failed: {e}. Continuing without NER.")
            training_stats = {'final_loss': 0}
        
        # 6. Analyze sentiment
        logger.info("Step 6: Analyzing sentiment...")
        analyzer = SentimentAnalyzer()
        
        # Analyze sentiment for financial messages
        sentiment_results = []
        for _, row in financial_messages.iterrows():
            result = analyzer.analyze_sentiment(row['cleaned_text'])
            sentiment_results.append(result)
            
        # Fix pandas warning by creating proper copy
        financial_messages = financial_messages.copy()
        financial_messages['sentiment_analysis'] = sentiment_results
        financial_messages['sentiment'] = financial_messages['sentiment_analysis'].apply(lambda x: x['sentiment'])
        financial_messages['sentiment_score'] = financial_messages['sentiment_analysis'].apply(lambda x: x['scores']['compound'])
        financial_messages['confidence'] = financial_messages['sentiment_analysis'].apply(lambda x: x['confidence'])
        
        logger.info(f"Completed sentiment analysis for {len(financial_messages)} messages")
        
        # 6.5. Save processed messages to database
        logger.info("Step 6.5: Saving processed messages to database...")
        db_save_stats = save_processed_messages_to_db(financial_messages)
        logger.info(f"Database save stats: {db_save_stats['saved']} new, {db_save_stats['updated']} updated, {db_save_stats['failed']} failed")
        
        # 7. Aggregate sentiment data
        logger.info("Step 7: Aggregating sentiment data...")
        aggregated_sentiment = aggregate_sentiment_data(financial_messages)
        logger.info(f"Generated sentiment data for {len(aggregated_sentiment)} company-date combinations")
        print(f"\n\n\nFinancial Messages : {financial_messages}\n\n\n\n")
        
        # 7.5. Analyze sender impact
        logger.info("Step 7.5: Analyzing sender impact...")
        sender_analysis = {}
        
        try:
            from models.sender_impact_analyzer import SenderImpactAnalyzer
            
            sender_analyzer = SenderImpactAnalyzer()
            sender_analysis = sender_analyzer.analyze_sender_impact(
                messages_df=financial_messages,
                stock_data=stock_data,
                analysis_window_days=7,
                save_to_db=True
            )
            
            if sender_analysis and 'analysis_summary' in sender_analysis:
                logger.info(f"✅ Sender impact analysis complete:")
                logger.info(f"   - Total participants: {sender_analysis['analysis_summary']['total_participants']}")
                logger.info(f"   - Classified: {sender_analysis['analysis_summary']['classified_participants']}")
                logger.info(f"   - Dominant group: {sender_analysis['analysis_summary']['dominant_group']}")
                
                # Save sender analysis to JSON
                sender_analysis_file = output_dir / f"sender_analysis_{timestamp}.json"
                with open(sender_analysis_file, 'w') as f:
                    # Convert any non-serializable objects
                    serializable_analysis = {
                        'session_id': sender_analysis.get('session_id'),
                        'analysis_summary': sender_analysis.get('analysis_summary'),
                        'sender_rankings': [
                            {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                             for k, v in sender.items()}
                            for sender in sender_analysis.get('sender_rankings', [])[:20]  # Top 20
                        ],
                        'group_statistics': sender_analysis.get('group_statistics')
                    }
                    json.dump(serializable_analysis, f, indent=2, default=str)
                logger.info(f"   - Saved to: {sender_analysis_file}")
        except Exception as e:
            logger.warning(f"Sender impact analysis failed: {e}. Continuing without sender analysis.")
        
        # 8. Combine stock and sentiment data
        logger.info("Step 8: Combining stock and sentiment data...")
        combined_stock_df = pd.DataFrame()
        
        if stock_data:
            # Combine all stock data
            stock_dfs = []
            for symbol, data in stock_data.items():
                data_copy = data.copy()
                data_copy['symbol'] = symbol
                stock_dfs.append(data_copy)
            
            if stock_dfs:
                combined_stock_df = pd.concat(stock_dfs, ignore_index=True)
                logger.info(f"Combined stock data: {len(combined_stock_df)} records")
        
        # 9. Feature engineering for ML
        logger.info("Step 9: Engineering features for ML prediction...")
        predictor = StockPredictor()
        training_results = None
        predictions_df = pd.DataFrame()
        
        if not combined_stock_df.empty and not aggregated_sentiment.empty:
            try:
                # Create features
                features_df = predictor.create_features(
                    stock_data=combined_stock_df,
                    sentiment_data=aggregated_sentiment,
                    prediction_horizon=3  # Predict 3 days ahead
                )
                
                logger.info(f"Created {len(features_df)} feature rows")
                
                # 10. Train ML model
                if len(features_df) >= 20:  # Need minimum data
                    logger.info("Step 10: Training XGBoost prediction model...")
                    
                    # Prepare training data
                    X = features_df.drop(columns=['symbol', 'date', 'target', 'target_return'], errors='ignore')
                    y = features_df['target']
                    
                    # Clean data
                    X = predictor._clean_features(X)
                    
                    # Train model
                    training_results = predictor.train_model(
                        X, y,
                        tune_hyperparameters=True,
                        dates=features_df['date']
                    )
                    
                    logger.info(f"Model training completed:")
                    logger.info(f"  - Test Accuracy: {training_results['metrics']['test']['accuracy']:.3f}")
                    logger.info(f"  - Test F1 Score: {training_results['metrics']['test']['f1_score']:.3f}")
                    
                    # Save model
                    model_path = Config.MODELS_DIR / "trained" / "stock_predictor.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    predictor.save_model(str(model_path))
                    logger.info(f"Model saved to {model_path}")
                    
                    # Make predictions on latest data
                    latest_features = features_df.groupby('symbol').last().reset_index()
                    latest_features_clean = predictor._clean_features(
                        latest_features.drop(columns=['symbol', 'date', 'target', 'target_return'], errors='ignore')
                    )
                    
                    if not latest_features_clean.empty:
                        prediction_results = predictor.predict(latest_features_clean)
                        
                        # Create prediction dataframe
                        predictions_df = latest_features[['symbol', 'date', 'close']].copy()
                        predictions_df['prediction'] = prediction_results['predictions']
                        predictions_df['confidence'] = prediction_results.get('confidence', [0.5] * len(predictions_df))
                        predictions_df['direction'] = predictions_df['prediction'].map({1: 'UP', 0: 'DOWN'})
                        
                        logger.info(f"Generated {len(predictions_df)} predictions")
                    else:
                        logger.warning("No valid data for predictions")
                else:
                    logger.warning(f"Insufficient data for training ({len(features_df)} samples). Need at least 20.")
                    
            except Exception as e:
                logger.error(f"Error in ML prediction: {e}")
                
        else:
            logger.warning("No sentiment data available for modeling")
        
        # 11. Generate predictions with reasoning
        logger.info("Step 11: Generating predictions with reasoning...")
        reasoning_engine = PredictionReasoning()
        
        # Get unique companies from processed messages
        all_companies = set()
        for _, row in financial_messages.iterrows():
            if row.get('company_mentions'):
                for mention in row['company_mentions']:
                    all_companies.add(mention['ticker'])
        
        logger.info(f"Found {len(all_companies)} companies mentioned: {list(all_companies)}")
        
        prediction_outputs = []
        
        for company_ticker in all_companies:
            logger.info(f"Processing predictions for {company_ticker}...")
            
            # Filter messages for this company
            company_messages = financial_messages[
                financial_messages['company_mentions'].apply(
                    lambda mentions: any(mention['ticker'] == company_ticker for mention in mentions)
                )
            ].copy()
            
            if len(company_messages) < 2:
                logger.info(f"Skipping {company_ticker} - only {len(company_messages)} messages")
                continue
            
            # Calculate sentiment for this company
            company_sentiment_scores = [r['scores']['compound'] for r in company_messages['sentiment_analysis']]
            avg_sentiment = np.mean(company_sentiment_scores)
            avg_confidence = np.mean([r['confidence'] for r in company_messages['sentiment_analysis']])
            
            # Get ML prediction if available
            if not predictions_df.empty and company_ticker in predictions_df['symbol'].values:
                ml_prediction = predictions_df[predictions_df['symbol'] == company_ticker].iloc[0]
                prediction = ml_prediction['direction']
                ml_confidence = ml_prediction['confidence']
                final_confidence = (avg_confidence + ml_confidence) / 2
            else:
                # Fallback to sentiment-based prediction
                if avg_sentiment > 0.2:
                    prediction = "UP"
                elif avg_sentiment < -0.2:
                    prediction = "DOWN"
                else:
                    prediction = "HOLD"
                final_confidence = avg_confidence
            
            # Generate reasoning
            company_stock_data = stock_data.get(company_ticker)
            reasoning_analysis = reasoning_engine.generate_reasoning(
                company_messages=company_messages,
                sentiment_score=avg_sentiment,
                confidence=final_confidence,
                prediction=prediction,
                stock_data=company_stock_data
            )
            
            # Create prediction output
            prediction_output = reasoning_engine.create_prediction_output(
                company=company_ticker.replace('.N0000', ''),
                prediction=prediction,
                confidence=final_confidence,
                sentiment_score=avg_sentiment,
                reasoning_analysis=reasoning_analysis
            )
            
            prediction_outputs.append(prediction_output)
            
            # Print individual prediction
            print(f"\n📊 Prediction for {company_ticker}:")
            print(json.dumps(prediction_output, indent=2))
        
        # 12. Save all results
        logger.info("Step 12: Saving results...")
        
        # Save processed messages to CSV (in addition to database)
        processed_file = output_dir / f"processed_messages_{timestamp}.csv"
        processed_messages.to_csv(processed_file, index=False)
        
        # Save stock data
        if not combined_stock_df.empty:
            stock_file = output_dir / f"stock_data_{timestamp}.csv"
            combined_stock_df.to_csv(stock_file, index=False)
        
        # Save sentiment data
        if not aggregated_sentiment.empty:
            sentiment_file = output_dir / f"sentiment_data_{timestamp}.csv"
            aggregated_sentiment.to_csv(sentiment_file, index=False)
        
        # Save ML predictions
        if not predictions_df.empty:
            ml_predictions_file = output_dir / f"ml_predictions_{timestamp}.csv"
            predictions_df.to_csv(ml_predictions_file, index=False)
        
        # Save predictions with reasoning
        if prediction_outputs:
            reasoning_file = output_dir / f"predictions_with_reasoning_{timestamp}.json"
            with open(reasoning_file, 'w') as f:
                json.dump(prediction_outputs, f, indent=2)
        
        logger.info("Pipeline completed successfully!")
        
        # Print comprehensive summary
        print(f"\n{'='*80}")
        print("SENTIMENT STOCK AGENT - COMPLETE EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"✅ WhatsApp messages loaded: {len(messages):,}")
        print(f"✅ Financial messages found: {len(financial_messages):,}")
        print(f"✅ Messages saved to database: {db_save_stats['saved']} new, {db_save_stats['updated']} updated")
        print(f"✅ Companies analyzed: {len(prediction_outputs)}")
        print(f"✅ Stock symbols processed: {len(stock_data)}")
        print(f"✅ Data quality score: {whatsapp_quality['quality_score']:.1f}/100")
        print(f"✅ NER model trained and saved")
        print(f"✅ Sentiment analysis completed")
        print(f"✅ Sentiment data aggregated: {len(aggregated_sentiment):,} records")
        
        if training_results:
            print(f"✅ XGBoost model trained (F1: {training_results['metrics']['test']['f1_score']:.3f})")
            print(f"✅ ML predictions generated: {len(predictions_df) if not predictions_df.empty else 0}")
        
        print(f"✅ Predictions with reasoning: {len(prediction_outputs)}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"💾 Database: {db_save_stats['saved'] + db_save_stats['updated']} messages in processed_messages table")
        print(f"{'='*80}")
        
        # Show prediction summary
        if prediction_outputs:
            print(f"\n🎯 PREDICTION SUMMARY:")
            for pred in prediction_outputs:
                direction_emoji = "📈" if pred['direction'] == "UP" else "📉" if pred['direction'] == "DOWN" else "➡️"
                print(f"{direction_emoji} {pred['company']}: {pred['direction']} "
                      f"(Confidence: {pred['confidence']:.2f}, Sentiment: {pred['sentiment_score']:.2f}, "
                      f"Hype: {pred['hype_score']:.2f})")
        
        # Show feature importance if model was trained
        if training_results and 'feature_importance' in training_results:
            print(f"\n🔍 TOP PREDICTIVE FEATURES:")
            for feature, importance in list(training_results['feature_importance'].items())[:5]:
                print(f"   {feature}: {importance:.3f}")
        
        # Show database save errors if any
        if db_save_stats['errors']:
            print(f"\n⚠️  DATABASE SAVE WARNINGS ({len(db_save_stats['errors'])} errors):")
            for error in db_save_stats['errors'][:5]:  # Show first 5 errors
                print(f"   {error}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")
        print("Please check the logs and configuration")

if __name__ == "__main__":
    main()