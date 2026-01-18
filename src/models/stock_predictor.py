import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import joblib
import json
from pathlib import Path
from config import Config
import ta

logger = logging.getLogger(__name__)

class StockPredictor:
    """XGBoost-based stock price prediction using sentiment and technical analysis"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.is_trained = False
        self.model_metadata = {}
        
    def create_features(
        self,
        stock_data: pd.DataFrame,
        sentiment_data: pd.DataFrame,
        prediction_horizon: int = None
    ) -> pd.DataFrame:
        """
        Create features for prediction model
        
        Args:
            stock_data: DataFrame with OHLCV data
            sentiment_data: DataFrame with sentiment scores
            prediction_horizon: Days to predict ahead
            
        Returns:
            DataFrame with engineered features
        """
        prediction_horizon = prediction_horizon or Config.PREDICTION_HORIZON_DAYS
        
        # Ensure date columns are datetime
        if 'date' in stock_data.columns:
            stock_data = stock_data.copy()
            stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        if not sentiment_data.empty and 'date' in sentiment_data.columns:
            sentiment_data = sentiment_data.copy()
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Start with stock data
        merged_data = stock_data.copy()
        
        # Merge with sentiment data if available
        if not sentiment_data.empty:
            merged_data = pd.merge(
                merged_data, sentiment_data,
                on=['date', 'symbol'],
                how='left'
            )
        
        # Fill missing sentiment values
        sentiment_columns = ['avg_sentiment', 'sentiment_std', 'message_count', 'avg_confidence']
        for col in sentiment_columns:
            if col not in merged_data.columns:
                merged_data[col] = 0
            merged_data[col] = merged_data[col].fillna(0)
        
        # Sort by symbol and date
        merged_data = merged_data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Create technical indicators
        merged_data = self._add_technical_features(merged_data)
        
        # Create sentiment features
        merged_data = self._add_sentiment_features(merged_data)
        
        # Create price targets (what we're predicting)
        merged_data = self._create_targets(merged_data, prediction_horizon)
        
        # Create time-based features
        merged_data = self._add_time_features(merged_data)
        
        # Create rolling features
        merged_data = self._add_rolling_features(merged_data)
        
        return merged_data
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean features by handling infinity and extreme values"""
        df_clean = df.copy()
        
        # Replace infinity with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Replace very large numbers
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Cap values at 99th percentile to handle outliers
            upper_limit = df_clean[col].quantile(0.99)
            lower_limit = df_clean[col].quantile(0.01)
            
            if pd.notna(upper_limit) and pd.notna(lower_limit):
                df_clean[col] = df_clean[col].clip(lower=lower_limit, upper=upper_limit)
        
        # Fill remaining NaN with 0
        df_clean = df_clean.fillna(0)
        
        return df_clean
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        df_copy = df.copy()
        
        try:
            # Group by symbol to calculate indicators per stock
            for symbol in df_copy['symbol'].unique():
                mask = df_copy['symbol'] == symbol
                symbol_data = df_copy[mask].copy()
                
                if len(symbol_data) < 20:  # Need enough data
                    continue
                
                # Price-based features
                df_copy.loc[mask, 'returns_1d'] = symbol_data['close'].pct_change()
                df_copy.loc[mask, 'returns_5d'] = symbol_data['close'].pct_change(periods=5)
                df_copy.loc[mask, 'returns_10d'] = symbol_data['close'].pct_change(periods=10)
                
                # Volatility
                returns_1d = symbol_data['close'].pct_change()
                df_copy.loc[mask, 'volatility_5d'] = returns_1d.rolling(5).std()
                df_copy.loc[mask, 'volatility_10d'] = returns_1d.rolling(10).std()
                
                # Moving averages
                df_copy.loc[mask, 'sma_5'] = symbol_data['close'].rolling(5).mean()
                df_copy.loc[mask, 'sma_10'] = symbol_data['close'].rolling(10).mean()
                df_copy.loc[mask, 'sma_20'] = symbol_data['close'].rolling(20).mean()
                
                # Moving average ratios
                sma_5 = symbol_data['close'].rolling(5).mean()
                sma_10 = symbol_data['close'].rolling(10).mean()
                sma_20 = symbol_data['close'].rolling(20).mean()
                
                df_copy.loc[mask, 'price_sma5_ratio'] = symbol_data['close'] / sma_5
                df_copy.loc[mask, 'price_sma10_ratio'] = symbol_data['close'] / sma_10
                df_copy.loc[mask, 'sma5_sma20_ratio'] = sma_5 / sma_20
                
                # RSI (if we have enough data)
                if len(symbol_data) >= 14:
                    try:
                        rsi = ta.momentum.rsi(symbol_data['close'], window=14)
                        df_copy.loc[mask, 'rsi'] = rsi
                    except:
                        df_copy.loc[mask, 'rsi'] = 50  # Neutral RSI
                
                # Volume features
                df_copy.loc[mask, 'volume_change'] = symbol_data['volume'].pct_change()
                df_copy.loc[mask, 'volume_sma_5'] = symbol_data['volume'].rolling(5).mean()
                
                volume_sma = symbol_data['volume'].rolling(5).mean()
                df_copy.loc[mask, 'volume_ratio'] = symbol_data['volume'] / volume_sma
                
                # High-Low spread
                df_copy.loc[mask, 'hl_spread'] = (symbol_data['high'] - symbol_data['low']) / symbol_data['low']
                df_copy.loc[mask, 'oc_spread'] = (symbol_data['close'] - symbol_data['open']) / symbol_data['open']
            
        except Exception as e:
            logger.error(f"Error adding technical features: {e}")
        
        return df_copy
    
    def _add_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sentiment-based features"""
        df_copy = df.copy()
        
        try:
            # Group by symbol for sentiment features
            for symbol in df_copy['symbol'].unique():
                mask = df_copy['symbol'] == symbol
                symbol_data = df_copy[mask].copy()
                
                if len(symbol_data) < 3:
                    continue
                
                # Rolling sentiment features
                df_copy.loc[mask, 'sentiment_ma_3'] = symbol_data['avg_sentiment'].rolling(3).mean()
                df_copy.loc[mask, 'sentiment_ma_7'] = symbol_data['avg_sentiment'].rolling(7).mean()
                
                # Sentiment momentum
                sentiment_change = symbol_data['avg_sentiment'].diff()
                df_copy.loc[mask, 'sentiment_change'] = sentiment_change
                df_copy.loc[mask, 'sentiment_momentum'] = sentiment_change.rolling(3).sum()
                
                # Message volume features
                df_copy.loc[mask, 'message_change'] = symbol_data['message_count'].pct_change()
                df_copy.loc[mask, 'message_ma_3'] = symbol_data['message_count'].rolling(3).mean()
                
                # Sentiment-price correlation features
                returns_1d = symbol_data['close'].pct_change()
                volume_change = symbol_data['volume'].pct_change()
                
                df_copy.loc[mask, 'sentiment_price_product'] = symbol_data['avg_sentiment'] * returns_1d
                df_copy.loc[mask, 'sentiment_volume_product'] = symbol_data['avg_sentiment'] * volume_change
                
        except Exception as e:
            logger.error(f"Error adding sentiment features: {e}")
        
        return df_copy
    
    def _create_targets(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Create prediction targets"""
        df_copy = df.copy()
        
        try:
            # Group by symbol to create targets
            for symbol in df_copy['symbol'].unique():
                mask = df_copy['symbol'] == symbol
                symbol_data = df_copy[mask].copy()
                
                # Create future price targets
                future_close = symbol_data['close'].shift(-horizon)
                future_return = (future_close - symbol_data['close']) / symbol_data['close']
                
                # Create binary target (up/down)
                target_direction = (future_return > 0).astype(int)
                
                # Create magnitude target (small/medium/large moves)
                target_magnitude = pd.cut(
                    future_return.abs(),
                    bins=[0, 0.02, 0.05, float('inf')],
                    labels=[0, 1, 2]  # Use numeric labels
                )
                
                # Assign back to main dataframe
                df_copy.loc[mask, 'future_close'] = future_close
                df_copy.loc[mask, 'future_return'] = future_return
                df_copy.loc[mask, 'target_direction'] = target_direction
                df_copy.loc[mask, 'target_magnitude'] = target_magnitude
                df_copy.loc[mask, 'target_return'] = future_return
                
        except Exception as e:
            logger.error(f"Error creating targets: {e}")
        
        return df_copy
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df_copy = df.copy()
        
        try:
            # Day of week
            df_copy['day_of_week'] = df_copy['date'].dt.dayofweek
            df_copy['is_monday'] = (df_copy['day_of_week'] == 0).astype(int)
            df_copy['is_friday'] = (df_copy['day_of_week'] == 4).astype(int)
            
            # Month
            df_copy['month'] = df_copy['date'].dt.month
            
            # Quarter
            df_copy['quarter'] = df_copy['date'].dt.quarter
            
        except Exception as e:
            logger.error(f"Error adding time features: {e}")
        
        return df_copy
    
    def _add_rolling_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add rolling statistical features"""
        if windows is None:
            windows = [3, 5, 10]
        
        df_copy = df.copy()
        
        try:
            for symbol in df_copy['symbol'].unique():
                mask = df_copy['symbol'] == symbol
                symbol_data = df_copy[mask].copy()
                
                for window in windows:
                    if len(symbol_data) < window:
                        continue
                    
                    # Price rolling statistics
                    df_copy.loc[mask, f'price_mean_{window}d'] = symbol_data['close'].rolling(window).mean()
                    df_copy.loc[mask, f'price_std_{window}d'] = symbol_data['close'].rolling(window).std()
                    
                    # Volume rolling statistics
                    df_copy.loc[mask, f'volume_mean_{window}d'] = symbol_data['volume'].rolling(window).mean()
                    
                    # Sentiment rolling statistics
                    if 'avg_sentiment' in symbol_data.columns:
                        df_copy.loc[mask, f'sentiment_mean_{window}d'] = symbol_data['avg_sentiment'].rolling(window).mean()
                        df_copy.loc[mask, f'sentiment_std_{window}d'] = symbol_data['avg_sentiment'].rolling(window).std()
            
        except Exception as e:
            logger.error(f"Error adding rolling features: {e}")
        
        return df_copy
    
    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'target_direction',
        remove_nulls: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.Series]:
        """
        Prepare data for model training
        
        Args:
            df: DataFrame with features and targets
            target_column: Name of target column
            remove_nulls: Remove rows with null values
            
        Returns:
            Tuple of (features, targets, feature_names, dates)
        """
        # Remove rows without targets
        clean_df = df.dropna(subset=[target_column]).copy()
        clean_df = self._clean_features(clean_df)
        # Select feature columns (exclude non-numeric and target columns)
        exclude_columns = [
            'symbol', 'company_name', 'date', 'future_close', 'future_return', 
            'target_direction', 'target_magnitude', 'target_return'
        ]
        
        feature_columns = []
        for col in clean_df.columns:
            if col not in exclude_columns:
                if clean_df[col].dtype in ['int64', 'float64']:
                    feature_columns.append(col)
        
        # Prepare features and targets
        X = clean_df[feature_columns].fillna(0)
        y = clean_df[target_column]
        dates = clean_df['date']
        
        # Handle categorical targets
        if y.dtype == 'object' or target_column == 'target_magnitude':
            y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        
        # Remove any remaining NaN values
        if remove_nulls:
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            dates = dates[mask]
        
        self.feature_columns = feature_columns
        logger.info(f"Prepared {len(X)} samples with {len(feature_columns)} features")
        
        return X, y, feature_columns, dates
    
    def time_based_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data based on time periods (chronological order)
        
        Args:
            X: Feature matrix
            y: Target vector  
            dates: Date series
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio
        
        # Sort by date to ensure chronological order
        sort_idx = dates.sort_values().index
        X_sorted = X.loc[sort_idx]
        y_sorted = y.loc[sort_idx]
        dates_sorted = dates.loc[sort_idx]
        
        # Calculate split indices
        n_samples = len(X_sorted)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split data chronologically
        X_train = X_sorted.iloc[:train_end]
        X_val = X_sorted.iloc[train_end:val_end]
        X_test = X_sorted.iloc[val_end:]
        
        y_train = y_sorted.iloc[:train_end]
        y_val = y_sorted.iloc[train_end:val_end]
        y_test = y_sorted.iloc[val_end:]
        
        train_dates = dates_sorted.iloc[:train_end]
        val_dates = dates_sorted.iloc[train_end:val_end]
        test_dates = dates_sorted.iloc[val_end:]
        
        logger.info(f"Time-based split:")
        logger.info(f"  Training: {len(X_train)} samples ({train_dates.min()} to {train_dates.max()})")
        logger.info(f"  Validation: {len(X_val)} samples ({val_dates.min()} to {val_dates.max()})")
        logger.info(f"  Testing: {len(X_test)} samples ({test_dates.min()} to {test_dates.max()})")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        hyperparameter_tuning: bool = False
    ) -> Dict[str, Any]:
        """
        Train XGBoost model using time-based splitting
        
        Args:
            X: Feature matrix
            y: Target vector
            dates: Date series for chronological splitting
            train_ratio: Proportion for training
            val_ratio: Proportion for validation  
            test_ratio: Proportion for testing
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Training results and metrics
        """
        if len(X) < 20:
            logger.error(f"Not enough data for training ({len(X)} samples). Need at least 20.")
            return {}
        
        # Time-based split
        X_train, X_val, X_test, y_train, y_val, y_test = self.time_based_split(
            X, y, dates, train_ratio, val_ratio, test_ratio
        )
        
        # Check if we have enough data in each split
        if len(X_train) < 10 or len(X_val) < 5 or len(X_test) < 5:
            logger.error("Insufficient data in one or more splits after time-based splitting")
            return {}
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get base parameters
        base_params = Config.XGBOOST_PARAMS.copy()
        
        if hyperparameter_tuning and len(X_train) > 50:
            # Hyperparameter tuning with time-series cross-validation
            param_grid = {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'n_estimators': [50, 100, 200]
            }
            
            # Use a simple validation approach for time series
            best_score = -float('inf')
            best_params = base_params
            
            for max_depth in param_grid['max_depth']:
                for lr in param_grid['learning_rate']:
                    for n_est in param_grid['n_estimators']:
                        params = base_params.copy()
                        params.update({
                            'max_depth': max_depth,
                            'learning_rate': lr,
                            'n_estimators': n_est
                        })
                        
                        model = xgb.XGBClassifier(**params)
                        model.fit(X_train_scaled, y_train)
                        val_pred = model.predict(X_val_scaled)
                        score = f1_score(y_val, val_pred, average='weighted', zero_division=0)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
            
            self.model = xgb.XGBClassifier(**best_params)
            self.model.fit(X_train_scaled, y_train)
            
        else:
            # Use base parameters
            self.model = xgb.XGBClassifier(**base_params)
            self.model.fit(X_train_scaled, y_train)
            best_params = base_params
        
        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        test_pred = self.model.predict(X_test_scaled)
        
        try:
            train_proba = self.model.predict_proba(X_train_scaled)
            val_proba = self.model.predict_proba(X_val_scaled)
            test_proba = self.model.predict_proba(X_test_scaled)
        except:
            # Fallback if predict_proba fails
            train_proba = np.column_stack([1-train_pred, train_pred])
            val_proba = np.column_stack([1-val_pred, val_pred])
            test_proba = np.column_stack([1-test_pred, test_pred])
        
        # Calculate metrics
        metrics = {
            'train': self._calculate_metrics(y_train, train_pred, train_proba),
            'validation': self._calculate_metrics(y_val, val_pred, val_proba),
            'test': self._calculate_metrics(y_test, test_pred, test_proba)
        }
        
        # Feature importance
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        ))
        
        # Store metadata
        self.model_metadata = {
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_columns),
            'training_samples': len(X_train),
            'best_params': best_params,
            'feature_importance': feature_importance,
            'split_method': 'time_based',
            'train_period': f"{dates.iloc[X_train.index].min()} to {dates.iloc[X_train.index].max()}",
            'test_period': f"{dates.iloc[X_test.index].min()} to {dates.iloc[X_test.index].max()}"
        }
        
        self.is_trained = True
        
        results = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'split_info': {
                'method': 'time_based',
                'train_period': self.model_metadata['train_period'],
                'test_period': self.model_metadata['test_period']
            }
        }
        
        logger.info(f"Model training completed with time-based split. Test F1 Score: {metrics['test']['f1_score']:.3f}")
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add AUC if binary classification and we have probabilities
        if y_proba.shape[1] == 2 and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics['auc'] = 0.5
        
        return {k: round(v, 4) for k, v in metrics.items()}
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_probabilities: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions
        
        Args:
            X: Feature matrix
            return_probabilities: Include prediction probabilities
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure we have the right features
        if isinstance(X, pd.DataFrame):
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in X.columns:
                    X[col] = 0
            X_features = X[self.feature_columns].fillna(0)
        else:
            X_features = X
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        result = {'predictions': predictions}
        
        if return_probabilities:
            try:
                probabilities = self.model.predict_proba(X_scaled)
                result['probabilities'] = probabilities
                # Add confidence (max probability)
                result['confidence'] = np.max(probabilities, axis=1)
            except:
                result['confidence'] = np.ones(len(predictions)) * 0.5
        
        return result
    
    def save_model(self, filepath: str) -> bool:
        """Save trained model"""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns,
                'metadata': self.model_metadata,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data.get('label_encoder')
            self.feature_columns = model_data['feature_columns']
            self.model_metadata = model_data.get('metadata', {})
            self.is_trained = model_data.get('is_trained', True)
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False