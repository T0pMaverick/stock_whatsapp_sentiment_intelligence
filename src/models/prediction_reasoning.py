import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime
import logging
from config import Config

logger = logging.getLogger(__name__)

class PredictionReasoning:
    """Generates human-readable reasoning for stock predictions"""
    
    def __init__(self):
        self.bearish_patterns = [
            'avoid', 'sell', 'dump', 'overvalued', 'expensive', 'risk', 'fall', 'drop', 
            'decline', 'bearish', 'negative', 'loss', 'crash', 'correction', 'resistance'
        ]
        
        self.bullish_patterns = [
            'buy', 'accumulate', 'undervalued', 'cheap', 'opportunity', 'rise', 'surge',
            'rally', 'bullish', 'positive', 'profit', 'breakout', 'support', 'target'
        ]
        
        self.volume_keywords = [
            'volume', 'activity', 'interest', 'discussion', 'attention', 'buzz'
        ]
        
        self.valuation_keywords = [
            'nav', 'pe', 'pb', 'valuation', 'price', 'target', 'fair value', 'worth'
        ]

    def generate_reasoning(
        self, 
        company_messages: pd.DataFrame,
        sentiment_score: float,
        confidence: float,
        prediction: str,
        stock_data: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Generate detailed reasoning for prediction
        
        Args:
            company_messages: Messages mentioning the company
            sentiment_score: Overall sentiment score
            confidence: Prediction confidence
            prediction: UP/DOWN prediction
            stock_data: Stock price data
            
        Returns:
            Dictionary with reasoning analysis
        """
        reasoning_elements = []
        hype_score = 0.0
        
        # 1. Analyze discussion volume
        volume_analysis = self._analyze_discussion_volume(company_messages)
        if volume_analysis['reasoning']:
            reasoning_elements.append(volume_analysis['reasoning'])
        hype_score += volume_analysis['hype_contribution']
        
        # 2. Analyze sentiment language
        sentiment_analysis = self._analyze_sentiment_language(company_messages, sentiment_score)
        if sentiment_analysis['reasoning']:
            reasoning_elements.extend(sentiment_analysis['reasoning'])
        
        # 3. Analyze valuation mentions
        valuation_analysis = self._analyze_valuation_mentions(company_messages)
        if valuation_analysis['reasoning']:
            reasoning_elements.extend(valuation_analysis['reasoning'])
        
        # 4. Analyze price targets and technical levels
        technical_analysis = self._analyze_technical_mentions(company_messages, stock_data)
        if technical_analysis['reasoning']:
            reasoning_elements.extend(technical_analysis['reasoning'])
        
        # 5. Analyze user behavior patterns
        behavior_analysis = self._analyze_user_behavior(company_messages)
        if behavior_analysis['reasoning']:
            reasoning_elements.extend(behavior_analysis['reasoning'])
        hype_score += behavior_analysis['hype_contribution']
        
        # Normalize hype score
        hype_score = min(1.0, hype_score)
        
        return {
            'reasoning': reasoning_elements[:4],  # Top 4 reasons
            'hype_score': round(hype_score, 2),
            'analysis_details': {
                'volume_analysis': volume_analysis,
                'sentiment_analysis': sentiment_analysis, 
                'valuation_analysis': valuation_analysis,
                'technical_analysis': technical_analysis,
                'behavior_analysis': behavior_analysis
            }
        }
    
    def _analyze_discussion_volume(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """Analyze discussion volume and activity"""
        if messages.empty:
            return {'reasoning': None, 'hype_contribution': 0.0}
        
        message_count = len(messages)
        unique_users = messages['from_name'].nunique()
        
        # Calculate activity metrics
        if message_count >= 10:
            if unique_users >= 5:
                reasoning = "High discussion volume with diverse participant engagement"
                hype_contribution = 0.4
            else:
                reasoning = "High discussion volume but limited participant diversity"
                hype_contribution = 0.2
        elif message_count >= 5:
            reasoning = "Moderate discussion activity"
            hype_contribution = 0.1
        else:
            return {'reasoning': None, 'hype_contribution': 0.0}
        
        return {
            'reasoning': reasoning,
            'hype_contribution': hype_contribution,
            'message_count': message_count,
            'unique_users': unique_users
        }
    
    def _analyze_sentiment_language(self, messages: pd.DataFrame, sentiment_score: float) -> Dict[str, Any]:
        """Analyze sentiment patterns in language"""
        if messages.empty:
            return {'reasoning': []}
        
        reasoning = []
        all_text = ' '.join(messages['cleaned_text'].fillna('').str.lower())
        
        # Count bearish vs bullish language
        bearish_count = sum(1 for pattern in self.bearish_patterns if pattern in all_text)
        bullish_count = sum(1 for pattern in self.bullish_patterns if pattern in all_text)
        
        total_sentiment_words = bearish_count + bullish_count
        
        if total_sentiment_words >= 3:
            if sentiment_score <= -0.3:
                if bearish_count > bullish_count * 1.5:
                    reasoning.append("Predominantly bearish sentiment language")
                reasoning.append("Strong negative sentiment indicators")
            elif sentiment_score >= 0.3:
                if bullish_count > bearish_count * 1.5:
                    reasoning.append("Predominantly bullish sentiment language")
                reasoning.append("Strong positive sentiment indicators")
            else:
                reasoning.append("Mixed sentiment signals with neutral bias")
        
        # Check for specific warning language
        warning_phrases = ['avoid', 'risk', 'overvalued', 'expensive']
        warning_count = sum(1 for phrase in warning_phrases if phrase in all_text)
        if warning_count >= 2:
            reasoning.append("Multiple risk warnings and cautionary language")
        
        # Check for opportunity language
        opportunity_phrases = ['opportunity', 'undervalued', 'cheap', 'buy the dip']
        opportunity_count = sum(1 for phrase in opportunity_phrases if phrase in all_text)
        if opportunity_count >= 2:
            reasoning.append("Investment opportunity language detected")
        
        return {
            'reasoning': reasoning,
            'bearish_count': bearish_count,
            'bullish_count': bullish_count,
            'sentiment_score': sentiment_score
        }
    
    def _analyze_valuation_mentions(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """Analyze valuation-related discussions"""
        if messages.empty:
            return {'reasoning': []}
        
        reasoning = []
        all_text = ' '.join(messages['cleaned_text'].fillna('').str.lower())
        
        # Look for NAV discussions
        nav_patterns = ['nav', 'net asset value', 'book value']
        nav_mentions = sum(1 for pattern in nav_patterns if pattern in all_text)
        
        if nav_mentions > 0:
            # Check if negative NAV comparison
            negative_nav_patterns = ['below nav', 'discount to nav', 'nav is higher']
            positive_nav_patterns = ['above nav', 'premium to nav', 'nav is lower']
            
            negative_nav = sum(1 for pattern in negative_nav_patterns if pattern in all_text)
            positive_nav = sum(1 for pattern in positive_nav_patterns if pattern in all_text)
            
            if negative_nav > positive_nav:
                reasoning.append("Negative NAV comparison discussions")
            elif positive_nav > negative_nav:
                reasoning.append("Trading at premium to NAV concerns")
        
        # Look for PE/valuation discussions
        if any(word in all_text for word in ['pe', 'pe ratio', 'expensive', 'cheap']):
            if any(word in all_text for word in ['expensive', 'overvalued', 'high pe']):
                reasoning.append("Valuation concerns about high multiples")
            elif any(word in all_text for word in ['cheap', 'undervalued', 'low pe']):
                reasoning.append("Attractive valuation metrics mentioned")
        
        return {
            'reasoning': reasoning,
            'nav_mentions': nav_mentions
        }
    
    def _analyze_technical_mentions(self, messages: pd.DataFrame, stock_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Analyze technical analysis and price target mentions"""
        if messages.empty:
            return {'reasoning': []}
        
        reasoning = []
        all_text = ' '.join(messages['cleaned_text'].fillna('').str.lower())
        
        # Look for price targets
        price_target_patterns = ['target', 'tp', 'price target', 'fair value']
        target_mentions = sum(1 for pattern in price_target_patterns if pattern in all_text)
        
        if target_mentions > 0:
            # Extract numbers near price targets
            price_numbers = re.findall(r'(?:target|tp|worth).*?(\d+\.?\d*)', all_text)
            if price_numbers:
                reasoning.append("Specific downside price targets mentioned")
        
        # Look for technical levels
        technical_patterns = ['resistance', 'support', 'breakout', 'breakdown']
        technical_count = sum(1 for pattern in technical_patterns if pattern in all_text)
        
        if 'resistance' in all_text:
            reasoning.append("Technical resistance levels identified")
        elif 'support' in all_text:
            reasoning.append("Technical support levels discussed")
        
        if 'breakdown' in all_text or 'break down' in all_text:
            reasoning.append("Technical breakdown patterns mentioned")
        elif 'breakout' in all_text or 'break out' in all_text:
            reasoning.append("Potential breakout scenarios discussed")
        
        return {
            'reasoning': reasoning,
            'technical_mentions': technical_count,
            'price_targets': price_numbers if 'price_numbers' in locals() else []
        }
    
    def _analyze_user_behavior(self, messages: pd.DataFrame) -> Dict[str, Any]:
        """Analyze user behavior patterns"""
        if messages.empty:
            return {'reasoning': [], 'hype_contribution': 0.0}
        
        reasoning = []
        hype_contribution = 0.0
        
        # Analyze message timing (clustering)
        if 'timestamp_dt' in messages.columns:
            messages_sorted = messages.sort_values('timestamp_dt')
            time_diffs = messages_sorted['timestamp_dt'].diff().dt.total_seconds() / 60  # minutes
            
            # Check for rapid-fire discussions (multiple messages within 5 minutes)
            rapid_messages = (time_diffs < 5).sum()
            if rapid_messages >= 3:
                reasoning.append("Intense rapid-fire discussion patterns")
                hype_contribution += 0.3
        
        # Check for repeated mentions by same users
        user_message_counts = messages['from_name'].value_counts()
        if len(user_message_counts) > 0:
            max_user_messages = user_message_counts.iloc[0]
            if max_user_messages >= 5:
                reasoning.append("Persistent user engagement and repeated mentions")
                hype_contribution += 0.2
        
        # Check for urgency language
        urgency_words = ['urgent', 'asap', 'immediately', 'quickly', 'now', 'today']
        all_text = ' '.join(messages['cleaned_text'].fillna('').str.lower())
        urgency_count = sum(1 for word in urgency_words if word in all_text)
        
        if urgency_count >= 2:
            reasoning.append("Urgency and time-sensitive language detected")
            hype_contribution += 0.1
        
        return {
            'reasoning': reasoning,
            'hype_contribution': hype_contribution
        }

    def create_prediction_output(
        self,
        company: str,
        prediction: str,
        confidence: float,
        sentiment_score: float,
        reasoning_analysis: Dict[str, Any],
        prediction_window: str = "next_3_days"
    ) -> Dict[str, Any]:
        """
        Create final prediction output with reasoning
        
        Args:
            company: Company ticker
            prediction: UP/DOWN prediction
            confidence: Prediction confidence
            sentiment_score: Overall sentiment
            reasoning_analysis: Analysis from generate_reasoning
            prediction_window: Prediction time window
            
        Returns:
            Formatted prediction output
        """
        return {
            "company": company,
            "prediction_window": prediction_window,
            "direction": prediction,
            "confidence": round(confidence, 2),
            "sentiment_score": round(sentiment_score, 2),
            "hype_score": reasoning_analysis['hype_score'],
            "reasoning": reasoning_analysis['reasoning']
        }