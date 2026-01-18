import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import re
from config import Config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Multi-model sentiment analysis using FinBERT, VADER, and TextBlob"""
    
    def __init__(self):
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.finbert_pipeline = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.financial_keywords = Config.FINANCIAL_KEYWORDS
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.load_finbert()
        self.setup_custom_keywords()
    
    def load_finbert(self):
        """Load FinBERT model for financial sentiment analysis"""
        try:
            model_name = Config.FINBERT_MODEL
            logger.info(f"Loading FinBERT model: {model_name}")
            
            self.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Create pipeline
            self.finbert_pipeline = pipeline(
                "sentiment-analysis",
                model=self.finbert_model,
                tokenizer=self.finbert_tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            logger.warning("Continuing without FinBERT - will use other models only")
            self.finbert_pipeline = None
    
    def setup_custom_keywords(self):
        """Setup custom financial keywords with weights"""
        self.keyword_weights = {

            # ================================
            # VERY STRONG POSITIVE (Conviction)
            # ================================
            'strong buy': 0.9,
            'heavy buy': 0.9,
            'accumulate aggressively': 0.85,
            'multibagger': 0.9,
            'breakout confirmed': 0.8,
            'price discovery': 0.7,
            'undervalued': 0.7,
            'deep value': 0.75,
            'cheap': 0.6,
            'bottomed out': 0.7,
            'reversal confirmed': 0.8,

            # ================================
            # STRONG POSITIVE (Momentum)
            # ================================
            'bullish': 0.8,
            'bull': 0.8,
            'rally': 0.7,
            'surge': 0.8,
            'moon': 0.9,
            'run up': 0.7,
            'upside': 0.6,
            'higher highs': 0.6,
            'trend reversal': 0.7,
            'break above': 0.6,
            'support held': 0.6,
            'volume breakout': 0.7,

            # ================================
            # MODERATE POSITIVE (Constructive)
            # ================================
            'buy': 0.6,
            'accumulate': 0.6,
            'add more': 0.5,
            'hold for long term': 0.5,
            'safe zone': 0.4,
            'fundamentally strong': 0.6,
            'good results': 0.6,
            'earnings growth': 0.6,
            'profit booking done': 0.4,

            # ================================
            # NEUTRAL / WAIT SIGNALS
            # ================================
            'hold': 0.1,
            'neutral': 0.0,
            'sideways': 0.0,
            'range bound': 0.0,
            'consolidation': 0.0,
            'wait and see': 0.0,
            'no clarity': -0.1,
            'uncertain': -0.1,

            # ================================
            # MODERATE NEGATIVE (Caution)
            # ================================
            'overvalued': -0.6,
            'expensive': -0.6,
            'resistance': -0.5,
            'profit taking': -0.4,
            'distribution': -0.5,
            'weak hands': -0.4,
            'fake breakout': -0.6,
            'trap': -0.6,

            # ================================
            # STRONG NEGATIVE (Risk / Exit)
            # ================================
            'bearish': -0.8,
            'bear': -0.8,
            'sell': -0.6,
            'strong sell': -0.9,
            'dump': -0.8,
            'crash': -0.9,
            'breakdown': -0.7,
            'downtrend': -0.7,
            'stop loss hit': -0.8,
            'avoid': -0.7,
            'panic selling': -0.9,
            'fear': -0.6,

            # ================================
            # MANIPULATION / OPERATOR LANGUAGE
            # ================================
            'operator driven': -0.7,
            'pump and dump': -0.9,
            'pump': 0.6,
            'dumping': -0.8,
            'retail trapped': -0.8,
            'fake volume': -0.7,

            # ================================
            # SINHALA (Unicode)
            # ================================
            'ගන්න': 0.6,                # buy
            'ගත්තොත් හොඳයි': 0.7,       # good to buy
            'තවත් ගන්න': 0.6,           # accumulate
            'ඉහළට යයි': 0.7,            # will go up
            'වැඩිවෙනවා': 0.6,           # increasing
            'ගොඩ යයි': 0.8,             # rally strongly

            'විකුණන්න': -0.6,           # sell
            'ඉහළ මිල': -0.5,            # overpriced
            'බිඳ වැටෙයි': -0.8,          # crash
            'පහළට යයි': -0.7,            # will go down
            'අවදානම්': -0.6,            # risky
            'නවත්තන්න': -0.6,           # stop

            'බලාගෙන ඉන්න': 0.0,         # wait
            'තත්ත්වය පැහැදිලි නැහැ': -0.1,

            # ================================
            # SINGLISH (Very Important)
            # ================================
            'ganna': 0.6,
            'gattoth hondai': 0.7,
            'thawath ganna': 0.6,
            'up wenawa': 0.6,
            'hari hondai': 0.6,
            'godai': 0.8,

            'sell karanna': -0.6,
            'vikunanna': -0.6,
            'down wenawa': -0.7,
            'risk ekak': -0.6,
            'bad news': -0.6,
            'wasi ne': -0.5,

            'balagena innawa': 0.0,
            'side eke yanawa': 0.0,

            # ================================
            # CSE / LOCAL SLANG
            # ================================
            'nav play': 0.6,
            'rights issue': -0.4,
            'ri risk': -0.6,
            'illiquid': -0.5,
            'thin volume': -0.5,
            'operator stock': -0.7,
            'retail favourite': 0.4,
            'smart money': 0.7,
        }

    
    def analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        if not self.finbert_pipeline:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}
        
        try:
            # FinBERT expects shorter text segments
            if len(text) > 512:
                text = text[:512]
            
            results = self.finbert_pipeline(text)
            
            # Convert to standard format
            scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            
            for result in results[0]:  # results is a list of lists
                label = result['label'].lower()
                score = result['score']
                
                if 'positive' in label:
                    scores['positive'] = score
                elif 'negative' in label:
                    scores['negative'] = score
                elif 'neutral' in label:
                    scores['neutral'] = score
            
            # Calculate compound score
            compound = scores['positive'] - scores['negative']
            scores['compound'] = compound
            
            return scores
            
        except Exception as e:
            logger.error(f"Error in FinBERT analysis: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0}
    
    def analyze_with_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                "positive": scores['pos'],
                "negative": scores['neg'],
                "neutral": scores['neu'],
                "compound": scores['compound']
            }
        except Exception as e:
            logger.error(f"Error in VADER analysis: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    def analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert polarity to positive/negative/neutral
            if polarity > 0.1:
                positive = polarity
                negative = 0.0
                neutral = 1 - abs(polarity)
            elif polarity < -0.1:
                positive = 0.0
                negative = abs(polarity)
                neutral = 1 - abs(polarity)
            else:
                positive = 0.0
                negative = 0.0
                neutral = 1.0
            
            return {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "compound": polarity
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob analysis: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    def analyze_with_keywords(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using custom financial keywords
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            text_lower = text.lower()
            total_score = 0.0
            keyword_count = 0
            
            # Check for weighted keywords
            for keyword, weight in self.keyword_weights.items():
                if keyword in text_lower:
                    total_score += weight
                    keyword_count += 1
            
            # Normalize score
            if keyword_count > 0:
                avg_score = total_score / keyword_count
            else:
                avg_score = 0.0
            
            # Convert to probability distribution
            if avg_score > 0.1:
                positive = min(1.0, avg_score)
                negative = 0.0
                neutral = 1 - positive
            elif avg_score < -0.1:
                positive = 0.0
                negative = min(1.0, abs(avg_score))
                neutral = 1 - negative
            else:
                positive = 0.0
                negative = 0.0
                neutral = 1.0
            
            return {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "compound": avg_score
            }
            
        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0}
    
    def ensemble_sentiment(
        self, 
        finbert_scores: Dict[str, float],
        vader_scores: Dict[str, float],
        textblob_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Combine sentiment scores from multiple models
        
        Args:
            finbert_scores: FinBERT sentiment scores
            vader_scores: VADER sentiment scores
            textblob_scores: TextBlob sentiment scores
            keyword_scores: Keyword-based scores
            weights: Optional custom weights for models
            
        Returns:
            Combined sentiment scores
        """
        # Default weights based on model performance for financial text
        if weights is None:
            weights = {
                "finbert": 0.40,
                "vader": 0.30,
                "textblob": 0.15,
                "keywords": 0.15
            }
        
        # Ensure weights sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Combine scores
        combined = {
            "positive": (
                finbert_scores["positive"] * weights["finbert"] +
                vader_scores["positive"] * weights["vader"] +
                textblob_scores["positive"] * weights["textblob"] +
                keyword_scores["positive"] * weights["keywords"]
            ),
            "negative": (
                finbert_scores["negative"] * weights["finbert"] +
                vader_scores["negative"] * weights["vader"] +
                textblob_scores["negative"] * weights["textblob"] +
                keyword_scores["negative"] * weights["keywords"]
            ),
            "neutral": (
                finbert_scores["neutral"] * weights["finbert"] +
                vader_scores["neutral"] * weights["vader"] +
                textblob_scores["neutral"] * weights["textblob"] +
                keyword_scores["neutral"] * weights["keywords"]
            ),
            "compound": (
                finbert_scores["compound"] * weights["finbert"] +
                vader_scores["compound"] * weights["vader"] +
                textblob_scores["compound"] * weights["textblob"] +
                keyword_scores["compound"] * weights["keywords"]
            )
        }
        
        return combined
    
    def calculate_confidence(
        self,
        finbert_scores: Dict[str, float],
        vader_scores: Dict[str, float],
        textblob_scores: Dict[str, float],
        keyword_scores: Dict[str, float],
        ensemble_scores: Dict[str, float]
    ) -> float:
        """
        Calculate confidence score based on model agreement
        
        Args:
            All sentiment scores from different models
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Get predicted sentiment class for each model
            models_predictions = []
            
            for scores in [finbert_scores, vader_scores, textblob_scores, keyword_scores]:
                max_score = max(scores["positive"], scores["negative"], scores["neutral"])
                
                if scores["positive"] == max_score:
                    models_predictions.append("positive")
                elif scores["negative"] == max_score:
                    models_predictions.append("negative")
                else:
                    models_predictions.append("neutral")
            
            # Calculate agreement
            most_common = max(set(models_predictions), key=models_predictions.count)
            agreement_count = models_predictions.count(most_common)
            agreement_ratio = agreement_count / len(models_predictions)
            
            # Base confidence on agreement
            base_confidence = agreement_ratio
            
            # Boost confidence if ensemble prediction is strong
            max_ensemble_score = max(
                ensemble_scores["positive"], 
                ensemble_scores["negative"], 
                ensemble_scores["neutral"]
            )
            
            strength_bonus = (max_ensemble_score - 0.33) * 0.5  # Bonus for strong predictions
            
            final_confidence = min(1.0, base_confidence + strength_bonus)
            
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5  # Default moderate confidence
    
    def analyze_sentiment(
        self, 
        text: str, 
        include_individual: bool = False,
        custom_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Perform complete sentiment analysis
        
        Args:
            text: Input text
            include_individual: Include individual model scores
            custom_weights: Custom model weights
            
        Returns:
            Complete sentiment analysis results
        """
        if not text or len(text.strip()) == 0:
            return {
                "sentiment": "neutral",
                "scores": {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "compound": 0.0},
                "confidence": 0.0
            }
        
        # Get scores from all models
        finbert_scores = self.analyze_with_finbert(text)
        vader_scores = self.analyze_with_vader(text)
        textblob_scores = self.analyze_with_textblob(text)
        keyword_scores = self.analyze_with_keywords(text)
        
        # Combine scores
        ensemble_scores = self.ensemble_sentiment(
            finbert_scores, vader_scores, textblob_scores, keyword_scores, custom_weights
        )
        
        # Calculate confidence
        confidence = self.calculate_confidence(
            finbert_scores, vader_scores, textblob_scores, keyword_scores, ensemble_scores
        )
        
        # Determine overall sentiment
        max_score = max(
            ensemble_scores["positive"],
            ensemble_scores["negative"], 
            ensemble_scores["neutral"]
        )
        
        if ensemble_scores["positive"] == max_score:
            sentiment = "positive"
        elif ensemble_scores["negative"] == max_score:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        result = {
            "sentiment": sentiment,
            "scores": ensemble_scores,
            "confidence": confidence,
            "text_length": len(text),
            "analysis_time": datetime.now().isoformat()
        }
        
        if include_individual:
            result["individual_scores"] = {
                "finbert": finbert_scores,
                "vader": vader_scores,
                "textblob": textblob_scores,
                "keywords": keyword_scores
            }
        
        return result