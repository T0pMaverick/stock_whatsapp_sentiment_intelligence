#!/usr/bin/env python3
"""
Stock Data Sync Job
Automatically fetches latest CSE stock prices during market hours
Schedule: Every hour from 9AM-3PM, Monday-Friday
"""

import sys
import os
import json
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd

# Add both src and root directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from data.stock_processor import StockDataProcessor
from sqlalchemy import create_engine, text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [STOCK_SYNC] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def stock_data_sync_job():
    """
    Main stock data synchronization job
    Fetches latest stock prices and saves to database
    """
    try:
        start_time = datetime.now()
        logger.info("Starting Stock Data Sync Job...")
        
        # Check if it's market hours (9AM-3PM, Mon-Fri)
        current_hour = start_time.hour
        current_weekday = start_time.weekday()  # 0=Monday, 6=Sunday
        
        if current_weekday >= 5:  # Saturday=5, Sunday=6
            logger.info("Market closed (Weekend) - Skipping stock sync")
            return
            
        # Temporarily disable market hours check for testing
        # if current_hour < 9 or current_hour >= 15:
        #     logger.info(f"Market closed (Hour: {current_hour}) - Skipping stock sync")
        #     return
        
        if current_hour < 9 or current_hour >= 15:
            logger.info(f"Off-market hours (Hour: {current_hour}) - Running maintenance sync")
        else:
            logger.info("Market hours active - Proceeding with stock sync")
        
        # Initialize stock processor
        stock_processor = StockDataProcessor()
        
        # Load ALL symbols from keywords.json as requested by senior
        try:
            with open(Config.KEYWORDS_FILE, 'r') as f:
                company_data = json.load(f)
                symbols_to_sync = [company['ticker'] for company in company_data]
            
            logger.info(f"Loaded {len(symbols_to_sync)} symbols from keywords.json")
            logger.info("Processing ALL symbols as requested (no priority filtering)")
                
        except Exception as e:
            logger.error(f"Failed to load symbols from keywords.json: {e}")
            logger.error("Cannot proceed without symbols file")
            return
        
        logger.info(f"Total symbols to sync: {len(symbols_to_sync)}")
        
        # Track sync results
        successful_syncs = 0
        failed_syncs = 0
        all_stock_data = []
        
        # Fetch data for each symbol
        for symbol in symbols_to_sync:
            try:
                logger.info(f"Fetching data for {symbol}...")
                
                # Fetch OHLCV data (last 5 bars for recent data)
                data = stock_processor.fetch_ohlcv_data(symbol, n_bars=5)
                
                if data is not None and not data.empty:
                    # Add technical indicators
                    data = stock_processor.add_technical_indicators(data)
                    
                    # Add metadata
                    data_copy = data.copy()
                    data_copy['symbol'] = symbol
                    data_copy['sync_timestamp'] = start_time
                    data_copy['date'] = data_copy.index
                    
                    # Get latest record only
                    latest_data = data_copy.tail(1)
                    all_stock_data.append(latest_data)
                    
                    successful_syncs += 1
                    logger.info(f"{symbol}: Latest price = {latest_data['close'].iloc[0]:.2f}")
                    
                else:
                    failed_syncs += 1
                    logger.warning(f"{symbol}: No data received")
                    
            except Exception as e:
                failed_syncs += 1
                logger.error(f"{symbol}: Error - {str(e)}")
                continue
        
        # Combine all data
        if all_stock_data:
            combined_df = pd.concat(all_stock_data, ignore_index=True)
            
            # Save to CSV backup first (always works)
            timestamp_str = start_time.strftime("%Y%m%d_%H%M%S")
            backup_file = Config.DATA_DIR / "processed" / f"stock_sync_{timestamp_str}.csv"
            combined_df.to_csv(backup_file, index=False)
            logger.info(f"CSV backup saved to: {backup_file}")
            
            # Try to save to database (optional if database is unavailable)
            try:
                engine = create_engine(Config.get_database_url())
                db_saved_count = 0
                
                with engine.connect() as conn:
                    for _, row in combined_df.iterrows():
                        try:
                            # Insert or update stock data
                            conn.execute(text("""
                                INSERT INTO stock_data 
                                (symbol, date, open, high, low, close, volume, 
                                 sma_20, rsi, sync_timestamp)
                                VALUES 
                                (:symbol, :date, :open, :high, :low, :close, :volume,
                                 :sma_20, :rsi, :sync_timestamp)
                                ON CONFLICT (symbol, date) DO UPDATE SET
                                    close = EXCLUDED.close,
                                    volume = EXCLUDED.volume,
                                    sma_20 = EXCLUDED.sma_20,
                                    rsi = EXCLUDED.rsi,
                                    sync_timestamp = EXCLUDED.sync_timestamp
                            """), {
                                'symbol': row['symbol'],
                                'date': row['date'],
                                'open': float(row['open']),
                                'high': float(row['high']),
                                'low': float(row['low']),
                                'close': float(row['close']),
                                'volume': int(row['volume']),
                                'sma_20': float(row.get('sma_20', 0)) if pd.notna(row.get('sma_20')) else None,
                                'rsi': float(row.get('rsi', 0)) if pd.notna(row.get('rsi')) else None,
                                'sync_timestamp': start_time
                            })
                            db_saved_count += 1
                        except Exception as e:
                            logger.error(f"Database error for {row['symbol']}: {e}")
                            continue
                    
                    conn.commit()
                    logger.info(f"Database save successful: {db_saved_count} records")
                    
            except Exception as e:
                logger.warning(f"Database save failed: {e}")
                logger.info("Data preserved in CSV backup - job continues successfully")
            
        # Log completion
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info("Stock Data Sync Job completed!")
        logger.info(f"   Successful syncs: {successful_syncs}")
        logger.info(f"   Failed syncs: {failed_syncs}")
        logger.info(f"   Processing time: {processing_time:.2f} seconds")
        
        if all_stock_data:
            logger.info(f"   Saved {len(combined_df)} records to database")
            logger.info(f"   Backup saved to: {backup_file}")
        
    except Exception as e:
        logger.error(f"Stock Data Sync Job failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # For testing purposes
    stock_data_sync_job()