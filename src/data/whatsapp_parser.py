import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.config = Config.DB_CONFIG
        self.engine = None
        self.connection = None
        
    def connect(self):
        """Establish database connection"""
        try:
            database_url = Config.get_database_url()
            self.engine = create_engine(database_url)
            self.connection = self.engine.connect()
            logger.info("Database connection established successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        try:
            if self.connection:
                self.connection.close()
            if self.engine:
                self.engine.dispose()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
    
    def execute_query(self, query: str, params: Dict = None) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        try:
            if params:
                df = pd.read_sql_query(text(query), self.engine, params=params)
            else:
                df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return pd.DataFrame()

class WhatsAppDataParser:
    """Parses WhatsApp data from PostgreSQL database"""
    
    def __init__(self, table_name: str = "messages"):
        self.db = DatabaseManager()
        self.table_name = table_name
        
    def __enter__(self):
        self.db.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.disconnect()
    
    def get_messages_by_date_range(
        self, 
        start_date: str, 
        end_date: str,
        group_name: Optional[str] = None
    ) -> pd.DataFrame:

        query = f"""
        SELECT 
            m.id AS message_id,
            m.group_id,
            g.name AS group_name,
            m.message_body,
            m.message_type,
            m.timestamp,
            m.from_number,
            m.from_name,
            m.author,
            m.is_from_me,
            m.has_media,
            m.media_path,
            m.ack,
            m.scraped_at,
            m.timestamp_formatted,
            m.author_phone
        FROM messages m
        INNER JOIN groups g ON m.group_id = g.id
        ORDER BY m.timestamp DESC;
        """
        
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        
        
        df = self.db.execute_query(query)
        print(f"Retrieved Messages Size : {df.shape}")
        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp_dt'] = pd.to_datetime(df['timestamp_formatted'])
            df['date'] = df['timestamp_dt'].dt.date
            df['hour'] = df['timestamp_dt'].dt.hour
            
            logger.info(f"Retrieved {len(df)} messages from {start_date} to {end_date}")
        
        return df
    
    def get_recent_messages(self, days: int = 30, group_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get messages from the last N days
        
        Args:
            days: Number of days to look back
            group_name: Optional group name filter
            
        Returns:
            DataFrame with recent messages
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_messages_by_date_range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            group_name
        )
    
    def get_messages_with_keywords(
        self, 
        keywords: List[str], 
        days: int = 30,
        case_sensitive: bool = False
    ) -> pd.DataFrame:
        """
        Get messages containing specific keywords
        
        Args:
            keywords: List of keywords to search for
            days: Number of days to look back
            case_sensitive: Whether search is case sensitive
            
        Returns:
            DataFrame with filtered messages
        """
        if not keywords:
            return pd.DataFrame()
        
        # Build keyword search conditions
        keyword_conditions = []
        params = {}
        
        for i, keyword in enumerate(keywords):
            param_name = f"keyword_{i}"
            if case_sensitive:
                keyword_conditions.append(f"message_body LIKE %(param_name)s")
            else:
                keyword_conditions.append(f"LOWER(message_body) LIKE LOWER(%(param_name)s)")
            params[param_name] = f"%{keyword}%"
        
        keyword_clause = " OR ".join(keyword_conditions)
        
        # Date filter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        params["start_date"] = start_date.strftime('%Y-%m-%d')
        params["end_date"] = end_date.strftime('%Y-%m-%d')
        
        query = f"""
        SELECT 
            m.id AS message_id,
            m.group_id,
            g.name AS group_name,
            m.message_body,
            m.message_type,
            m.timestamp,
            m.from_number,
            m.from_name,
            m.author,
            m.is_from_me,
            m.has_media,
            m.media_path,
            m.ack,
            m.scraped_at,
            m.timestamp_formatted,
            m.author_phone
        FROM messages m
        INNER JOIN groups g ON m.group_id = g.id
        WHERE timestamp_formatted >= %(start_date)s
        AND timestamp_formatted <= %(end_date)s
        AND message_body IS NOT NULL
        AND LENGTH(TRIM(message_body)) > 0
        AND ({keyword_clause})
        ORDER BY m.timestamp DESC;
        """
        
        df = self.db.execute_query(query, params)
        
        if not df.empty:
            df['timestamp_dt'] = pd.to_datetime(df['timestamp_formatted'])
            df['date'] = df['timestamp_dt'].dt.date
            
            logger.info(f"Found {len(df)} messages containing keywords: {', '.join(keywords)}")
        
        return df