import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple, Optional, Any
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from datetime import datetime
import json
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)

class MessagePreprocessor:
    """Preprocesses WhatsApp messages for sentiment analysis and NER"""
    
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.financial_keywords = Config.FINANCIAL_KEYWORDS
        
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        self.stop_words = set(stopwords.words('english'))
        
    def setup_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load(Config.SPACY_MODEL)
        except OSError:
            logger.error(f"spaCy model '{Config.SPACY_MODEL}' not found. Please install it with: python -m spacy download {Config.SPACY_MODEL}")
            self.nlp = None
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove WhatsApp specific patterns
        text = re.sub(r'@\d+', '', text)  # Remove @mentions
        text = re.sub(r'\+\d+', '', text)  # Remove phone numbers
        
        # Handle emojis - convert common ones to text
        emoji_patterns = {
            r'😀|😁|😂|🤣|😃|😄|😅|😆|😊|☺️|🙂|😉': ' happy ',
            r'😢|😭|😞|😔|😟|😕|😤|😠|😡|🤬': ' sad ',
            r'📈|📊|💰|💵|💴|💶|💷|💸': ' money ',
            r'👍|👌|✅|✔️': ' good ',
            r'👎|❌|❎': ' bad ',
            r'🔥|🚀|📈': ' bullish ',
            r'📉|💥|⬇️': ' bearish '
        }
        
        for pattern, replacement in emoji_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Remove remaining emojis
        text = re.sub(r'[^\w\s\.,!?;:]', ' ', text)
        
        # Handle contractions
        contractions = {
            r"won't": "will not",
            r"can't": "cannot",
            r"n't": " not",
            r"'re": " are",
            r"'ve": " have",
            r"'ll": " will",
            r"'d": " would",
            r"'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = re.sub(contraction, expansion, text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def extract_financial_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract financial entities like prices, percentages, etc.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with extracted entities
        """
        entities = {
            'prices': [],
            'percentages': [],
            'numbers': [],
            'currency_symbols': [],
            'financial_keywords': []
        }
        
        # Extract prices (e.g., 100.50, Rs 100, $50.25)
        price_patterns = [
            r'rs\.?\s*\d+\.?\d*',  # Rs 100, Rs. 100.50
            r'\$\d+\.?\d*',        # $100, $100.50
            r'\d+\.?\d*\s*rs',     # 100 rs, 100.50rs
            r'\d+\.?\d*\s*rupees', # 100 rupees
            r'\d+\.?\d*\s*lkr',    # 100 LKR
        ]
        
        for pattern in price_patterns:
            matches = re.findall(pattern, text.lower())
            entities['prices'].extend(matches)
        
        # Extract percentages
        percentage_pattern = r'\d+\.?\d*\s*%'
        entities['percentages'] = re.findall(percentage_pattern, text)
        
        # Extract standalone numbers
        number_pattern = r'\b\d+\.?\d*\b'
        entities['numbers'] = re.findall(number_pattern, text)
        
        # Extract financial keywords
        all_financial_words = []
        for category in self.financial_keywords.values():
            all_financial_words.extend(category)
        
        words = word_tokenize(text.lower())
        entities['financial_keywords'] = [
            word for word in words if word in all_financial_words
        ]
        
        return entities
    
    def extract_company_mentions(self, text: str, company_data: List[Dict]) -> List[Dict]:
        """
        Extract company mentions from text
        
        Args:
            text: Input text
            company_data: List of company information dictionaries
            
        Returns:
            List of detected company mentions with confidence
        """
        mentions = []
        text_lower = text.lower()
        
        for company in company_data:
            company_name = company['company_name'].lower()
            ticker = company['ticker'].lower()
            aliases = [alias.lower() for alias in company.get('aliases', [])]
            
            # Check for exact matches
            all_names = [company_name, ticker] + aliases
            
            for name in all_names:
                if name in text_lower:
                    # Calculate confidence based on match type and context
                    confidence = self._calculate_mention_confidence(text_lower, name, company)
                    
                    mentions.append({
                        'company_name': company['company_name'],
                        'ticker': company['ticker'],
                        'matched_term': name,
                        'confidence': confidence,
                        'context': self._extract_context(text, name)
                    })
        
        # Remove duplicates and sort by confidence
        unique_mentions = {}
        for mention in mentions:
            key = mention['ticker']
            if key not in unique_mentions or mention['confidence'] > unique_mentions[key]['confidence']:
                unique_mentions[key] = mention
        
        return sorted(unique_mentions.values(), key=lambda x: x['confidence'], reverse=True)
    
    def _calculate_mention_confidence(self, text: str, matched_term: str, company: Dict) -> float:
        """Calculate confidence score for company mention"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for ticker symbols (usually more specific)
        if matched_term == company['ticker'].lower():
            confidence += 0.3
        
        # Higher confidence if surrounded by financial context
        financial_context_words = ['stock', 'price', 'share', 'trading', 'buy', 'sell', 'invest']
        context_words = text.split()
        
        for word in context_words:
            if word in financial_context_words:
                confidence += 0.1
                break
        
        # Higher confidence for exact word boundaries
        if re.search(rf'\b{re.escape(matched_term)}\b', text):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_context(self, text: str, matched_term: str, window: int = 5) -> str:
        """Extract context around matched term"""
        words = text.split()
        
        for i, word in enumerate(words):
            if matched_term.lower() in word.lower():
                start = max(0, i - window)
                end = min(len(words), i + window + 1)
                return ' '.join(words[start:end])
        
        return text[:100]  # Fallback to first 100 characters
    
    def process_messages(self, df: pd.DataFrame, company_data: List[Dict] = None) -> pd.DataFrame:
        """
        Process all messages in DataFrame
        
        Args:
            df: DataFrame with message data
            company_data: Company information for entity extraction
            
        Returns:
            DataFrame with processed columns added
        """
        if df.empty:
            return df
        
        logger.info(f"Processing {len(df)} messages...")
        
        # Load company data if not provided
        if company_data is None:
            try:
                with open(Config.KEYWORDS_FILE, 'r') as f:
                    company_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading company data: {e}")
                company_data = []
        
        # Create processed columns
        df_processed = df.copy()
        
        # Basic text cleaning
        df_processed['cleaned_text'] = df_processed['message_body'].apply(self.clean_text)
        
        # Remove empty messages after cleaning
        df_processed = df_processed[df_processed['cleaned_text'].str.len() > 0]
        
        # Extract financial entities
        df_processed['financial_entities'] = df_processed['cleaned_text'].apply(
            self.extract_financial_entities
        )
        
        # Count sentiment keywords
        df_processed['sentiment_keywords'] = df_processed['cleaned_text'].apply(
            self.detect_sentiment_keywords
        )
        
        # Extract company mentions
        if company_data:
            df_processed['company_mentions'] = df_processed['cleaned_text'].apply(
                lambda x: self.extract_company_mentions(x, company_data)
            )
        else:
            df_processed['company_mentions'] = [[] for _ in range(len(df_processed))]
        
        # Text statistics
        df_processed['word_count'] = df_processed['cleaned_text'].apply(
            lambda x: len(word_tokenize(x))
        )
        df_processed['sentence_count'] = df_processed['cleaned_text'].apply(
            lambda x: len(sent_tokenize(x))
        )
        df_processed['char_count'] = df_processed['cleaned_text'].apply(len)
        
        # Financial relevance score
        df_processed['financial_relevance'] = df_processed.apply(
            self._calculate_financial_relevance, axis=1
        )
        
        # Filter out non-financial messages if desired
        df_processed['is_financial'] = df_processed['financial_relevance'] > 0.3
        
        logger.info(f"Processing completed. {len(df_processed)} messages processed.")
        logger.info(f"Financial messages: {df_processed['is_financial'].sum()}")
        
        return df_processed
    
    def detect_sentiment_keywords(self, text: str) -> Dict[str, int]:
        """
        Count sentiment-related keywords
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with keyword counts
        """
        text_lower = text.lower()
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        for sentiment, keywords in self.financial_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            sentiment_counts[sentiment] = count
        
        return sentiment_counts
    
    def _calculate_financial_relevance(self, row: pd.Series) -> float:
        """Calculate financial relevance score for a message"""
        score = 0.0
        
        # Company mentions boost score significantly
        if row['company_mentions']:
            score += 0.5
            # Higher score for high-confidence mentions
            max_confidence = max([mention['confidence'] for mention in row['company_mentions']])
            score += max_confidence * 0.3
        
        # Financial entities boost score
        entities = row['financial_entities']
        if entities['prices']:
            score += 0.3
        if entities['percentages']:
            score += 0.2
        if entities['financial_keywords']:
            score += min(0.3, len(entities['financial_keywords']) * 0.1)
        
        # Sentiment keywords boost score
        sentiment_counts = row['sentiment_keywords']
        total_sentiment_keywords = sum(sentiment_counts.values())
        score += min(0.2, total_sentiment_keywords * 0.05)
        
        return min(1.0, score)