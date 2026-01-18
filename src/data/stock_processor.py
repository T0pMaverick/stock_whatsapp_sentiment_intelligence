import json
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
import os
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import ta
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class StockDataProcessor:
    """Processes stock market data using TradingView DataFeed"""
    
    def __init__(self, keywords_file: Optional[str] = None):
        self.keywords_file = keywords_file or Config.KEYWORDS_FILE
        self.tv = TvDatafeed()
        self.symbols_data = self.load_symbols()
        
    def load_symbols(self) -> List[Dict]:
        """Load symbols and company names from keywords.json"""
        try:
            with open(self.keywords_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} symbols from keywords file")
            return data
        except Exception as e:
            logger.error(f"Error loading keywords file: {e}")
            return []
    
    def fetch_ohlcv_data(
        self, 
        symbol: str, 
        exchange: str = None,
        n_bars: int = None,
        interval: str = "daily"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data for a given symbol
        
        Args:
            symbol: Stock symbol
            exchange: Exchange name (default from config)
            n_bars: Number of bars to fetch (default from config)
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        try:
            exchange = exchange or Config.TV_EXCHANGE
            n_bars = n_bars or Config.TV_N_BARS
            
            # Map interval string to TvDatafeed Interval
            interval_map = {
                "1min": Interval.in_1_minute,
                "5min": Interval.in_5_minute,
                "15min": Interval.in_15_minute,
                "30min": Interval.in_30_minute,
                "1h": Interval.in_1_hour,
                "2h": Interval.in_2_hour,
                "4h": Interval.in_4_hour,
                "daily": Interval.in_daily,
                "weekly": Interval.in_weekly,
                "monthly": Interval.in_monthly
            }
            
            tv_interval = interval_map.get(interval, Interval.in_daily)
            
            logger.info(f"Fetching data for {symbol}...")
            data = self.tv.get_hist(
                symbol=symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=n_bars
            )
            
            if data is None or data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
                
            logger.info(f"Successfully fetched {len(data)} entries for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            df_copy = df.copy()
            
            # Price-based indicators
            df_copy['sma_5'] = ta.trend.sma_indicator(df_copy['close'], window=5)
            df_copy['sma_10'] = ta.trend.sma_indicator(df_copy['close'], window=10)
            df_copy['sma_20'] = ta.trend.sma_indicator(df_copy['close'], window=20)
            df_copy['ema_12'] = ta.trend.ema_indicator(df_copy['close'], window=12)
            df_copy['ema_26'] = ta.trend.ema_indicator(df_copy['close'], window=26)
            
            # MACD
            macd = ta.trend.MACD(df_copy['close'])
            df_copy['macd'] = macd.macd()
            df_copy['macd_signal'] = macd.macd_signal()
            df_copy['macd_histogram'] = macd.macd_diff()
            
            # RSI
            df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=14)
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df_copy['close'], window=20)
            df_copy['bb_upper'] = bollinger.bollinger_hband()
            df_copy['bb_lower'] = bollinger.bollinger_lband()
            df_copy['bb_middle'] = bollinger.bollinger_mavg()
            df_copy['bb_width'] = df_copy['bb_upper'] - df_copy['bb_lower']
            
            # Volume indicators
            # Volume indicators
            df_copy['volume_sma'] = df_copy['volume'].rolling(window=20).mean()
            
            # Price changes
            df_copy['price_change'] = df_copy['close'].pct_change()
            df_copy['price_change_abs'] = df_copy['close'].diff()
            df_copy['volume_change'] = df_copy['volume'].pct_change()
            
            # Volatility
            df_copy['volatility'] = df_copy['price_change'].rolling(window=20).std()
            
            # Support and Resistance levels
            df_copy['high_20'] = df_copy['high'].rolling(window=20).max()
            df_copy['low_20'] = df_copy['low'].rolling(window=20).min()
            
            # Average True Range (ATR)
            df_copy['atr'] = ta.volatility.average_true_range(
                df_copy['high'], df_copy['low'], df_copy['close'], window=14
            )
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(
                df_copy['high'], df_copy['low'], df_copy['close']
            )
            df_copy['stoch_k'] = stoch.stoch()
            df_copy['stoch_d'] = stoch.stoch_signal()
            
            logger.info("Technical indicators added successfully")
            return df_copy
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def fetch_all_symbols_data(
        self, 
        n_bars: int = None,
        add_indicators: bool = True,
        save_individual: bool = True,
        save_combined: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols in keywords file
        
        Args:
            n_bars: Number of bars to fetch per symbol
            add_indicators: Whether to add technical indicators
            save_individual: Save individual CSV files
            save_combined: Save combined CSV file
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        if not self.symbols_data:
            logger.error("No symbols found in keywords file")
            return {}
        
        n_bars = n_bars or Config.TV_N_BARS
        all_data = {}
        successful_downloads = 0
        failed_downloads = 0
        
        # Create output directory
        output_dir = Path(Config.OHLCV_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting data fetch for {len(self.symbols_data)} symbols")
        
        # Process each symbol
        for i, item in enumerate(self.symbols_data, 1):
            symbol = item['ticker']
            company_name = item['company_name']
            
            logger.info(f"Processing {i}/{len(self.symbols_data)}: {symbol} - {company_name}")
            
            # Fetch OHLCV data
            data = self.fetch_ohlcv_data(symbol, n_bars=n_bars)
            
            if data is not None:
                # Add symbol and company information
                data_copy = data.copy()
                data_copy['symbol'] = symbol
                data_copy['company_name'] = company_name
                data_copy['date'] = data_copy.index
                
                # Add technical indicators if requested
                if add_indicators:
                    data_copy = self.add_technical_indicators(data_copy)
                
                # Reorder columns
                base_columns = ['symbol', 'company_name', 'date', 'open', 'high', 'low', 'close', 'volume']
                other_columns = [col for col in data_copy.columns if col not in base_columns]
                data_copy = data_copy[base_columns + other_columns]
                
                # Save individual file
                if save_individual:
                    filename = output_dir / f"{symbol}_ohlcv.csv"
                    data_copy.to_csv(filename, index=False)
                    logger.info(f"Saved {symbol} data to {filename}")
                
                all_data[symbol] = data_copy
                successful_downloads += 1
                
            else:
                failed_downloads += 1
            
            # Add delay to avoid overwhelming the API
            time.sleep(0.5)
        
        # Save combined file
        if save_combined and all_data:
            combined_df = pd.concat(all_data.values(), ignore_index=True)
            combined_filename = output_dir / "all_stocks_ohlcv_data.csv"
            combined_df.to_csv(combined_filename, index=False)
            logger.info(f"Saved combined data to {combined_filename}")
        
        logger.info(f"Data fetch completed. Success: {successful_downloads}, Failed: {failed_downloads}")
        return all_data