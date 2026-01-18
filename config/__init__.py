import os
from pathlib import Path
from typing import Dict, List
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration"""
    
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432') 
    DB_NAME = os.getenv('DB_NAME', 'whatsapp_analysis')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'password')
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    SRC_DIR = PROJECT_ROOT / "src"
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CONFIG_DIR = PROJECT_ROOT / "config"
    
    # Database configuration
    DB_CONFIG = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "whatsapp_data"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
    }
    
    # Model configuration
    FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(MODELS_DIR / "cache"))
    
    # Data configuration
    OHLCV_DATA_DIR = os.getenv("OHLCV_DATA_DIR", str(DATA_DIR / "raw" / "ohlcv"))
    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", str(DATA_DIR / "processed"))
    KEYWORDS_FILE = os.getenv("KEYWORDS_FILE", str(CONFIG_DIR / "keywords.json"))
    
    # TradingView configuration
    TV_EXCHANGE = os.getenv("TV_EXCHANGE", "CSELK")
    TV_INTERVAL = os.getenv("TV_INTERVAL", "daily")
    TV_N_BARS = int(os.getenv("TV_N_BARS", 500))
    
    # Prediction configuration
    PREDICTION_HORIZON_DAYS = int(os.getenv("PREDICTION_HORIZON_DAYS", 5))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    SENTIMENT_WINDOW_DAYS = int(os.getenv("SENTIMENT_WINDOW_DAYS", 3))
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", str(LOGS_DIR / "sentiment_agent.log"))
    
    # Financial keywords for sentiment enhancement
    FINANCIAL_KEYWORDS = {
        "positive": [
            "profit", "gain", "bull", "bullish", "buy", "long", "up", "rise", "surge",
            "boom", "rally", "growth", "strong", "good", "excellent", "positive",
            "breakout", "momentum", "support", "target", "uptrend"
        ],
        "negative": [
            "loss", "bear", "bearish", "sell", "short", "down", "fall", "crash",
            "dump", "decline", "weak", "bad", "terrible", "negative", "breakdown",
            "resistance", "downtrend", "correction", "panic", "fear"
        ],
        "neutral": [
            "analysis", "report", "update", "news", "neutral", "stable", "flat",
            "sideways", "consolidation", "range", "volume", "technical", "chart"
        ]
    }
    
    # NER training configuration
    NER_CONFIG = {
        "model_name": "en_core_web_sm",
        "iterations": 30,
        "dropout": 0.5,
        "batch_size": 16,
        "learning_rate": 0.001
    }
    
    # XGBoost configuration
    XGBOOST_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.DATA_DIR / "raw",
            cls.DATA_DIR / "processed",
            cls.DATA_DIR / "external",
            cls.MODELS_DIR / "trained",
            cls.MODELS_DIR / "checkpoints",
            cls.LOGS_DIR,
            cls.CONFIG_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database connection URL"""
        config = cls.DB_CONFIG
        return f"postgresql://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}"

# Create directories on import
Config.create_directories()