import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class DataQualityValidator:
    """Validates data quality for both WhatsApp messages and stock data"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_whatsapp_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate WhatsApp message data quality
        
        Args:
            df: DataFrame with WhatsApp messages
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_records": len(df),
            "validation_time": datetime.now().isoformat(),
            "issues": [],
            "warnings": [],
            "quality_score": 0.0
        }
        
        if df.empty:
            results["issues"].append("DataFrame is empty")
            return results
        
        # Check required columns
        required_columns = [
            'message_body', 'from_name', 'timestamp_formatted', 'group_name'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results["issues"].append(f"Missing required columns: {missing_columns}")
        
        # Check for null values in critical columns
        null_checks = {}
        for col in required_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_percentage = (null_count / len(df)) * 100
                null_checks[col] = {
                    "null_count": int(null_count),
                    "null_percentage": round(null_percentage, 2)
                }
                
                if null_percentage > 10:
                    results["issues"].append(
                        f"High null percentage in {col}: {null_percentage:.1f}%"
                    )
                elif null_percentage > 5:
                    results["warnings"].append(
                        f"Moderate null percentage in {col}: {null_percentage:.1f}%"
                    )
        
        results["null_analysis"] = null_checks
        
        # Check message body quality
        if 'message_body' in df.columns:
            message_quality = self._analyze_message_quality(df['message_body'])
            results["message_quality"] = message_quality
            
            if message_quality["empty_message_percentage"] > 20:
                results["issues"].append(
                    f"High percentage of empty messages: {message_quality['empty_message_percentage']:.1f}%"
                )
        
        # Check timestamp consistency
        if 'timestamp_formatted' in df.columns:
            timestamp_quality = self._analyze_timestamp_quality(df['timestamp_formatted'])
            results["timestamp_quality"] = timestamp_quality
            
            if timestamp_quality["invalid_timestamps"] > 0:
                results["issues"].append(
                    f"Found {timestamp_quality['invalid_timestamps']} invalid timestamps"
                )
        
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        results["duplicate_records"] = int(duplicate_count)
        if duplicate_count > 0:
            results["warnings"].append(f"Found {duplicate_count} duplicate records")
        
        # Check user activity patterns
        if 'from_name' in df.columns:
            user_activity = self._analyze_user_activity(df)
            results["user_activity"] = user_activity
            
            if user_activity["single_user_dominance"] > 80:
                results["warnings"].append(
                    f"Single user dominance: {user_activity['single_user_dominance']:.1f}%"
                )
        
        # Calculate overall quality score
        results["quality_score"] = self._calculate_quality_score(results)
        
        logger.info(f"WhatsApp data validation completed. Quality score: {results['quality_score']:.2f}")
        return results
    
    def _analyze_message_quality(self, messages: pd.Series) -> Dict[str, Any]:
        """Analyze message content quality"""
        total_messages = len(messages)
        
        # Filter out null messages
        valid_messages = messages.dropna()
        
        # Calculate various metrics
        empty_messages = valid_messages.str.strip().str.len() == 0
        short_messages = valid_messages.str.len() < 5
        long_messages = valid_messages.str.len() > 500
        
        return {
            "total_messages": total_messages,
            "valid_messages": len(valid_messages),
            "empty_message_count": int(empty_messages.sum()),
            "empty_message_percentage": round((empty_messages.sum() / total_messages) * 100, 2),
            "short_message_count": int(short_messages.sum()),
            "short_message_percentage": round((short_messages.sum() / total_messages) * 100, 2),
            "long_message_count": int(long_messages.sum()),
            "average_message_length": round(valid_messages.str.len().mean(), 2),
            "median_message_length": round(valid_messages.str.len().median(), 2)
        }
    
    def _analyze_timestamp_quality(self, timestamps: pd.Series) -> Dict[str, Any]:
        """Analyze timestamp data quality"""
        try:
            # Convert to datetime
            timestamp_dt = pd.to_datetime(timestamps, errors='coerce')
            invalid_timestamps = timestamp_dt.isnull().sum()
            
            if len(timestamp_dt.dropna()) == 0:
                return {"invalid_timestamps": len(timestamps), "date_range": None}
            
            valid_timestamps = timestamp_dt.dropna()
            
            # Fix timezone comparison issue
            current_time = pd.Timestamp.now(tz='UTC')  # Make timezone-aware
            
            return {
                "invalid_timestamps": int(invalid_timestamps),
                "date_range": {
                    "start": valid_timestamps.min().isoformat(),
                    "end": valid_timestamps.max().isoformat(),
                    "span_days": (valid_timestamps.max() - valid_timestamps.min()).days
                },
                "future_timestamps": int((valid_timestamps > current_time).sum()),
                "duplicate_timestamps": int(valid_timestamps.duplicated().sum())
            }
        except Exception as e:
            logger.error(f"Error analyzing timestamps: {e}")
            return {"invalid_timestamps": 0, "date_range": None, "future_timestamps": 0, "duplicate_timestamps": 0}
    
    def _analyze_user_activity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user activity patterns"""
        if 'from_name' not in df.columns:
            return {}
        
        user_counts = df['from_name'].value_counts()
        total_messages = len(df)
        
        return {
            "unique_users": len(user_counts),
            "most_active_user": {
                "name": user_counts.index[0],
                "message_count": int(user_counts.iloc[0]),
                "percentage": round((user_counts.iloc[0] / total_messages) * 100, 2)
            },
            "single_user_dominance": round((user_counts.iloc[0] / total_messages) * 100, 2),
            "users_with_single_message": int((user_counts == 1).sum()),
            "average_messages_per_user": round(user_counts.mean(), 2)
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for issues
        score -= len(results.get("issues", [])) * 10
        
        # Deduct points for warnings
        score -= len(results.get("warnings", [])) * 5
        
        # Additional deductions based on specific metrics
        if "message_quality" in results:
            empty_pct = results["message_quality"].get("empty_message_percentage", 0)
            score -= max(0, (empty_pct - 5) * 2)  # Deduct after 5% empty messages
        
        if "null_analysis" in results:
            for col, analysis in results["null_analysis"].items():
                null_pct = analysis.get("null_percentage", 0)
                score -= max(0, (null_pct - 2) * 1)  # Deduct after 2% null values
        
        return max(0.0, min(100.0, score))