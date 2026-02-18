#!/usr/bin/env python3
"""
Message Processing Job
Automatically processes new WhatsApp messages for sentiment analysis
Schedule: Every hour at minute 10
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pandas as pd
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import Config
from data.whatsapp_parser import WhatsAppDataParser
from data.quality_validator import DataQualityValidator
from processing.text_preprocessor import MessagePreprocessor
from models.sentiment_analyzer import SentimentAnalyzer
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [MESSAGE_PROCESS] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_processed_messages_to_db(financial_messages: pd.DataFrame) -> dict:
    """Save processed messages with sentiment analysis to PostgreSQL database"""
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
                    
                except Exception as e:
                    stats['failed'] += 1
                    error_msg = f"Error saving message {idx}: {str(e)}"
                    stats['errors'].append(error_msg)
                    logger.error(error_msg)
                    continue
            
            conn.commit()
            
        logger.info(f"Database save complete: {stats['saved']} saved, {stats['updated']} updated, {stats['failed']} failed")
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        stats['errors'].append(f"Database connection error: {str(e)}")
    
    return stats

def message_processing_job():
    """
    Main message processing job
    Processes new WhatsApp messages for sentiment analysis
    """
    try:
        start_time = datetime.now()
        logger.info("Starting Message Processing Job...")
        
        # Create necessary directories
        Config.create_directories()
        output_dir = Config.DATA_DIR / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        
        # 1. Load WhatsApp data (last 2 hours to catch new messages)
        logger.info("Loading recent WhatsApp messages...")
        try:
            with WhatsAppDataParser() as parser:
                messages = parser.get_recent_messages(hours=2, group_name="Wealth Builders")
            logger.info(f"Loaded {len(messages)} recent messages")
        except Exception as e:
            logger.warning(f"WhatsApp data loading failed: {e}")
            messages = pd.DataFrame()  # Continue with empty messages
        
        if messages.empty:
            logger.info("No new messages to process")
            return
        
        # 2. Validate data quality
        logger.info("Validating data quality...")
        validator = DataQualityValidator()
        try:
            whatsapp_quality = validator.validate_whatsapp_data(messages)
            logger.info(f"WhatsApp data quality score: {whatsapp_quality['quality_score']:.2f}/100")
        except Exception as e:
            logger.warning(f"Data quality validation failed: {e}")
            whatsapp_quality = {'quality_score': 50}  # Default score
        
        # 3. Preprocess messages
        logger.info("Preprocessing messages...")
        preprocessor = MessagePreprocessor()
        
        try:
            processed_messages = preprocessor.process_messages(messages)
            logger.info(f"Processed {len(processed_messages)} messages")
            
            financial_messages = processed_messages[processed_messages.get('is_financial', False)]
            logger.info(f"Found {len(financial_messages)} financial messages")
        except Exception as e:
            logger.error(f"Message preprocessing failed: {e}")
            return
        
        if financial_messages.empty:
            logger.info("No financial messages found - job completed")
            return
        
        # 4. Analyze sentiment
        logger.info("Analyzing sentiment...")
        analyzer = SentimentAnalyzer()
        
        try:
            sentiment_results = []
            for _, row in financial_messages.iterrows():
                result = analyzer.analyze_sentiment(row['cleaned_text'])
                sentiment_results.append(result)
            
            # Add sentiment data to dataframe
            financial_messages = financial_messages.copy()
            financial_messages['sentiment_analysis'] = sentiment_results
            financial_messages['sentiment'] = financial_messages['sentiment_analysis'].apply(
                lambda x: x['sentiment']
            )
            financial_messages['sentiment_score'] = financial_messages['sentiment_analysis'].apply(
                lambda x: x['scores']['compound']
            )
            financial_messages['confidence'] = financial_messages['sentiment_analysis'].apply(
                lambda x: x['confidence']
            )
            
            logger.info(f"Completed sentiment analysis for {len(financial_messages)} messages")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return
        
        # 5. Save processed messages to database
        logger.info("Saving processed messages to database...")
        try:
            db_save_stats = save_processed_messages_to_db(financial_messages)
            logger.info(f"Database save stats: {db_save_stats['saved']} new, {db_save_stats['updated']} updated, {db_save_stats['failed']} failed")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            db_save_stats = {'saved': 0, 'updated': 0, 'failed': len(financial_messages)}
        
        # 6. Save backup CSV
        try:
            processed_file = output_dir / f"processed_messages_{timestamp}.csv"
            financial_messages.drop('sentiment_analysis', axis=1, errors='ignore').to_csv(processed_file, index=False)
            logger.info(f"Backup saved to: {processed_file}")
        except Exception as e:
            logger.warning(f"Backup CSV save failed: {e}")
        
        # 7. Generate summary statistics
        if not financial_messages.empty:
            sentiment_counts = financial_messages['sentiment'].value_counts().to_dict()
            avg_sentiment_score = financial_messages['sentiment_score'].mean()
            
            # Get unique companies mentioned
            all_companies = set()
            for _, row in financial_messages.iterrows():
                if row.get('company_mentions'):
                    for mention in row['company_mentions']:
                        all_companies.add(mention.get('ticker', 'Unknown'))
            
            logger.info("Processing Summary:")
            logger.info(f"   Messages processed: {len(financial_messages)}")
            logger.info(f"   Companies mentioned: {len(all_companies)}")
            logger.info(f"   Average sentiment score: {avg_sentiment_score:.3f}")
            logger.info(f"   Positive messages: {sentiment_counts.get('positive', 0)}")
            logger.info(f"   Neutral messages: {sentiment_counts.get('neutral', 0)}")  
            logger.info(f"   Negative messages: {sentiment_counts.get('negative', 0)}")
        
        # Log completion
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info("Message Processing Job completed!")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Database records: {db_save_stats['saved'] + db_save_stats['updated']}")
        logger.info(f"   Data quality: {whatsapp_quality['quality_score']:.1f}/100")
        
    except Exception as e:
        logger.error(f"Message Processing Job failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # For testing purposes
    message_processing_job()