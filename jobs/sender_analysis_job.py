#!/usr/bin/env python3
"""
Sender Impact Analysis Job
Analyzes which WhatsApp group members give the best stock predictions
Schedule: Daily at 2 AM
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
from models.sender_impact_analyzer import SenderImpactAnalyzer
from data.stock_processor import StockDataProcessor
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [SENDER_ANALYSIS] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sender_impact_analysis_job():
    """
    Main sender impact analysis job
    Updates sender performance correlations and classifications
    """
    try:
        start_time = datetime.now()
        logger.info("Starting Sender Impact Analysis Job...")
        
        # Initialize components
        analyzer = SenderImpactAnalyzer()
        stock_processor = StockDataProcessor()
        
        # Get recent messages from database (last 30 days)
        logger.info("Loading recent processed messages from database...")
        
        engine = create_engine(Config.get_database_url())
        
        # Load recent financial messages
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 
                    message_id,
                    message_body,
                    cleaned_text,
                    from_name,
                    timestamp_formatted,
                    group_name,
                    sentiment_score,
                    confidence,
                    company_mentions
                FROM processed_messages
                WHERE 
                    is_financial = true 
                    AND timestamp_formatted >= NOW() - INTERVAL '30 days'
                    AND from_name != 'Unknown'
                    AND from_name != ''
                ORDER BY timestamp_formatted DESC
            """))
            
            rows = result.fetchall()
            if not rows:
                logger.warning("No recent financial messages found")
                return
            
            # Convert to DataFrame
            messages_df = pd.DataFrame(rows, columns=result.keys())
            logger.info(f"Loaded {len(messages_df)} financial messages")
        
        # Load stock data for mentioned companies
        logger.info("Loading stock data for analysis...")
        
        # Extract unique companies from messages
        all_companies = set()
        for _, row in messages_df.iterrows():
            try:
                company_mentions = row.get('company_mentions', [])
                if isinstance(company_mentions, str):
                    company_mentions = json.loads(company_mentions)
                
                for mention in company_mentions:
                    if isinstance(mention, dict) and 'ticker' in mention:
                        all_companies.add(mention['ticker'])
            except Exception as e:
                logger.warning(f"Error parsing company mentions: {e}")
                continue
        
        logger.info(f"Found {len(all_companies)} unique companies mentioned")
        
        # Load stock data for these companies
        stock_data = {}
        successful_loads = 0
        
        for symbol in all_companies:
            try:
                data = stock_processor.fetch_ohlcv_data(symbol, n_bars=100)
                if data is not None:
                    data = stock_processor.add_technical_indicators(data)
                    stock_data[symbol] = data
                    successful_loads += 1
            except Exception as e:
                logger.warning(f"Failed to load stock data for {symbol}: {e}")
        
        logger.info(f"Successfully loaded stock data for {successful_loads} symbols")
        
        if not stock_data:
            logger.warning("No stock data available - skipping analysis")
            return
        
        # Run sender impact analysis
        logger.info("Running sender impact analysis...")
        
        try:
            analysis_results = analyzer.analyze_sender_impact(
                messages_df=messages_df,
                stock_data=stock_data,
                analysis_window_days=7,
                save_to_db=True
            )
            
            if analysis_results:
                summary = analysis_results.get('analysis_summary', {})
                logger.info("Sender impact analysis completed:")
                logger.info(f"   Total participants: {summary.get('total_participants', 0)}")
                logger.info(f"   Classified participants: {summary.get('classified_participants', 0)}")
                logger.info(f"   Dominant group: {summary.get('dominant_group', 'Unknown')}")
                
                # Log top performers
                rankings = analysis_results.get('sender_rankings', [])
                if rankings:
                    logger.info("Top 5 performers:")
                    for i, sender in enumerate(rankings[:5], 1):
                        classification = sender.get('classification', 'Unknown')
                        influence_score = sender.get('influence_score', 0)
                        sender_name = sender.get('sender_name', 'Unknown')
                        
                        logger.info(f"   {i}. {sender_name}: "
                                  f"{classification.upper()} "
                                  f"(Score: {influence_score:.3f})")
                
                # Save analysis results to JSON backup
                try:
                    timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
                    output_dir = Config.DATA_DIR / "processed"
                    sender_analysis_file = output_dir / f"sender_analysis_{timestamp_str}.json"
                    
                    # Convert any non-serializable objects
                    serializable_analysis = {
                        'session_id': analysis_results.get('session_id'),
                        'analysis_summary': analysis_results.get('analysis_summary'),
                        'sender_rankings': [
                            {k: float(v) if isinstance(v, (pd.Series, pd.DataFrame)) or 
                                   (isinstance(v, (int, float)) and not isinstance(v, bool)) 
                                   else str(v) if not isinstance(v, (dict, list, str, bool, type(None))) 
                                   else v
                             for k, v in sender.items()}
                            for sender in analysis_results.get('sender_rankings', [])[:20]  # Top 20
                        ],
                        'group_statistics': analysis_results.get('group_statistics'),
                        'timestamp': start_time.isoformat()
                    }
                    
                    with open(sender_analysis_file, 'w') as f:
                        json.dump(serializable_analysis, f, indent=2, default=str)
                    
                    logger.info(f"Analysis backup saved to: {sender_analysis_file}")
                except Exception as e:
                    logger.warning(f"Failed to save analysis backup: {e}")
                
                # Cleanup old analysis records (keep last 90 days)
                logger.info("Cleaning up old analysis records...")
                try:
                    cleanup_count = analyzer.cleanup_old_analysis(days_to_keep=90)
                    logger.info(f"Cleaned up {cleanup_count} old analysis records")
                except Exception as e:
                    logger.warning(f"Cleanup failed: {e}")
                
                # Generate insights for top performers
                logger.info("Analysis Insights:")
                group_stats = analysis_results.get('group_statistics', {})
                
                if 'mover' in group_stats:
                    mover_count = group_stats['mover'].get('count', 0)
                    mover_avg_accuracy = group_stats['mover'].get('avg_accuracy', 0)
                    logger.info(f"   Movers: {mover_count} members (avg accuracy: {mover_avg_accuracy:.2%})")
                
                if 'seller' in group_stats:
                    seller_count = group_stats['seller'].get('count', 0)
                    seller_avg_accuracy = group_stats['seller'].get('avg_accuracy', 0)
                    logger.info(f"   Sellers: {seller_count} members (avg accuracy: {seller_avg_accuracy:.2%})")
                
                logger.info("Sender impact analysis update completed successfully!")
                
            else:
                logger.warning("Sender impact analysis returned no results")
                
        except Exception as e:
            logger.error(f"Sender analysis execution failed: {e}")
            return
        
        # Verify database update
        logger.info("Verifying database updates...")
        try:
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM sender_analysis 
                    WHERE created_at > NOW() - INTERVAL '6 hours'
                """))
                recent_count = result.fetchone()[0]
                
                if recent_count > 0:
                    logger.info(f"Database verification passed: {recent_count} recent analysis records")
                else:
                    logger.warning("No recent analysis records found in database")
        except Exception as e:
            logger.warning(f"Database verification failed: {e}")
        
        # Log completion
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Sender Impact Analysis Job completed!")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        logger.info(f"   Messages analyzed: {len(messages_df)}")
        logger.info(f"   Companies processed: {successful_loads}")
        if analysis_results:
            logger.info(f"   Participants classified: {summary.get('classified_participants', 0)}")
        
    except Exception as e:
        logger.error(f"Sender Impact Analysis Job failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # For testing purposes
    sender_impact_analysis_job()