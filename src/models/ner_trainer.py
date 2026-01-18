import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import json
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import pickle
from config import Config

logger = logging.getLogger(__name__)

class CompanyNERTrainer:
    """Trains custom NER model for company recognition"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.SPACY_MODEL
        self.nlp = None
        self.ner = None
        self.training_data = []
        
    def create_blank_model(self, lang: str = "en") -> None:
        """Create a blank spaCy model"""
        try:
            self.nlp = spacy.blank(lang)
            self.ner = self.nlp.add_pipe("ner", last=True)
            logger.info("Created blank spaCy model")
        except Exception as e:
            logger.error(f"Error creating blank model: {e}")
            raise
    
    def load_existing_model(self) -> bool:
        """Load existing spaCy model"""
        try:
            self.nlp = spacy.load(self.model_name)
            
            # Get the NER component or add if doesn't exist
            if "ner" in self.nlp.pipe_names:
                self.ner = self.nlp.get_pipe("ner")
                logger.info(f"Loaded existing model: {self.model_name}")
            else:
                self.ner = self.nlp.add_pipe("ner", last=True)
                logger.info(f"Added NER to existing model: {self.model_name}")
                
            return True
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            return False
    
    def add_labels(self, labels: List[str]) -> None:
        """Add labels to NER component"""
        if not self.ner:
            raise ValueError("NER component not initialized")
            
        for label in labels:
            self.ner.add_label(label)
        logger.info(f"Added labels: {labels}")
    
    def prepare_training_data_from_companies(
        self, 
        company_data: List[Dict],
        sample_texts: List[str] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Prepare training data from company information
        
        Args:
            company_data: List of company dictionaries
            sample_texts: Optional list of sample texts to augment
            
        Returns:
            List of training examples
        """
        training_data = []
        
        # Generate synthetic training examples
        for company in company_data:
            ticker = company['ticker']
            company_name = company['company_name']
            aliases = company.get('aliases', [])
            
            # Create examples with different patterns
            patterns = [
                f"{ticker} is performing well today",
                f"I'm buying {ticker} shares",
                f"What do you think about {ticker}?",
                f"{company_name} stock is rising",
                f"Should I invest in {company_name}?",
                f"{company_name} announced good results",
                f"Price target for {ticker} is 50",
                f"{ticker} hit resistance at 100",
            ]
            
            # Add alias examples
            for alias in aliases:
                patterns.extend([
                    f"{alias} is a good buy",
                    f"Selling {alias} tomorrow",
                    f"{alias} price went up"
                ])
            
            # Create training examples
            for pattern in patterns:
                entities = []
                
                # Find all company mentions in the pattern
                text_lower = pattern.lower()
                
                # Check for ticker
                ticker_start = text_lower.find(ticker.lower())
                if ticker_start != -1:
                    entities.append((ticker_start, ticker_start + len(ticker), "COMPANY"))
                
                # Check for company name
                name_start = text_lower.find(company_name.lower())
                if name_start != -1:
                    entities.append((name_start, name_start + len(company_name), "COMPANY"))
                
                # Check for aliases
                for alias in aliases:
                    alias_start = text_lower.find(alias.lower())
                    if alias_start != -1:
                        entities.append((alias_start, alias_start + len(alias), "COMPANY"))
                
                if entities:
                    training_data.append((pattern, {"entities": entities}))
        
        # Add negative examples (texts without company mentions)
        negative_examples = [
            "The market is volatile today",
            "Economic indicators show growth",
            "Trading volume is high",
            "Bull market continues",
            "Bear market conditions",
            "Portfolio diversification is important",
            "Technical analysis suggests uptrend",
            "Fundamental analysis is crucial"
        ]
        
        for text in negative_examples:
            training_data.append((text, {"entities": []}))
        
        # Add real text samples if provided
        if sample_texts:
            for text in sample_texts:
                entities = self._find_entities_in_text(text, company_data)
                training_data.append((text, {"entities": entities}))
        
        logger.info(f"Generated {len(training_data)} training examples")
        self.training_data = training_data
        return training_data
    
    def _find_entities_in_text(self, text: str, company_data: List[Dict]) -> List[Tuple[int, int, str]]:
        """Find company entities in text - FIXED VERSION"""
        entities = []
        text_lower = text.lower()
        
        # Collect all potential matches first
        potential_matches = []
        
        for company in company_data:
            all_names = [company['ticker'], company['company_name']] + company.get('aliases', [])
            
            for name in all_names:
                name_lower = name.lower().strip()
                if len(name_lower) < 2:  # Skip very short names
                    continue
                    
                # Find all occurrences using word boundaries
                import re
                pattern = r'\b' + re.escape(name_lower) + r'\b'
                
                for match in re.finditer(pattern, text_lower):
                    start = match.start()
                    end = match.end()
                    # Store match with its length (for prioritizing longer matches)
                    potential_matches.append((start, end, 'COMPANY', len(name_lower)))
        
        # Remove overlaps - keep longer matches
        entities = self._remove_overlapping_entities_fixed(potential_matches)
        return entities

    def _remove_overlapping_entities_fixed(self, potential_matches: List[Tuple[int, int, str, int]]) -> List[Tuple[int, int, str]]:
        """Remove overlapping entities, keeping longer ones - FIXED VERSION"""
        if not potential_matches:
            return []
        
        # Sort by start position, then by length (descending)
        potential_matches = sorted(potential_matches, key=lambda x: (x[0], -x[3]))
        
        filtered = []
        
        for start, end, label, length in potential_matches:
            # Check if this overlaps with any existing entity
            overlaps = False
            for existing_start, existing_end, _ in filtered:
                # Check for any overlap
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append((start, end, label))
        
        return filtered

    def prepare_training_data_from_companies(
        self, 
        company_data: List[Dict],
        sample_texts: List[str] = None
    ) -> List[Tuple[str, Dict]]:
        """
        Prepare training data from company information - FIXED VERSION
        """
        training_data = []
        
        # Generate cleaner synthetic training examples
        for company in company_data:
            ticker = company['ticker'].replace('.N0000', '')  # Clean ticker
            company_name = company['company_name']
            aliases = company.get('aliases', [])
            
            # Create simpler patterns that won't cause overlaps
            base_patterns = [
                f"{ticker} is performing well",
                f"Buying {ticker} shares",
                f"What about {ticker}?",
                f"{company_name} stock is rising",
                f"Invest in {company_name}",
                f"{company_name} announced results"
            ]
            
            # Add each pattern as a separate training example
            for pattern in base_patterns:
                # Find entities in this specific pattern
                entities = self._find_entities_in_text(pattern, [company])
                
                if entities:
                    training_data.append((pattern, {"entities": entities}))
            
            # Add alias examples separately to avoid conflicts
            for alias in aliases[:2]:  # Limit to first 2 aliases
                alias_patterns = [
                    f"{alias} looks good",
                    f"Selling {alias} tomorrow"
                ]
                
                for pattern in alias_patterns:
                    entities = self._find_entities_in_text(pattern, [company])
                    if entities:
                        training_data.append((pattern, {"entities": entities}))
        
        # Add negative examples (no entities)
        negative_examples = [
            "Market is volatile today",
            "Economic indicators show growth", 
            "Trading volume is high",
            "Technical analysis suggests uptrend"
        ]
        
        for text in negative_examples:
            training_data.append((text, {"entities": []}))
        
        # Add real text samples if provided
        if sample_texts:
            for text in sample_texts[:10]:  # Limit to prevent overloading
                entities = self._find_entities_in_text(text, company_data)
                if entities:  # Only add if entities found
                    training_data.append((text, {"entities": entities}))
        
        # Final cleanup - remove any training examples with overlapping entities
        cleaned_training_data = []
        for text, annotations in training_data:
            entities = annotations.get("entities", [])
            
            # Double-check for overlaps
            if len(entities) <= 1:
                cleaned_training_data.append((text, annotations))
            else:
                # Check all pairs for overlaps
                has_overlap = False
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        start1, end1, _ = entities[i]
                        start2, end2, _ = entities[j]
                        
                        # Check if they overlap
                        if not (end1 <= start2 or start1 >= end2):
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                
                if not has_overlap:
                    cleaned_training_data.append((text, annotations))
        
        logger.info(f"Generated {len(cleaned_training_data)} clean training examples (removed {len(training_data) - len(cleaned_training_data)} with overlaps)")
        self.training_data = cleaned_training_data
        return cleaned_training_data
    
    def train_model(
        self,
        training_data: List[Tuple[str, Dict]] = None,
        iterations: int = None,
        dropout: float = None,
        batch_size: int = None,
        learning_rate: float = None
    ) -> Dict[str, Any]:
        """
        Train the NER model
        
        Args:
            training_data: Training data (uses self.training_data if None)
            iterations: Number of training iterations
            dropout: Dropout rate
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            Training statistics
        """
        if not self.nlp or not self.ner:
            raise ValueError("Model not initialized. Call create_blank_model() or load_existing_model() first")
        
        # Use provided data or instance data
        training_data = training_data or self.training_data
        if not training_data:
            raise ValueError("No training data available")
        
        # Use config defaults if not provided
        config = Config.NER_CONFIG
        iterations = iterations or config.get('iterations', 30)
        dropout = dropout or config.get('dropout', 0.5)
        batch_size = batch_size or config.get('batch_size', 16)
        learning_rate = learning_rate or config.get('learning_rate', 0.0001)
        
        # Add COMPANY label
        self.add_labels(["COMPANY"])
        
        # Prepare training examples
        train_examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            train_examples.append(example)
        
        logger.info(f"Starting training with {len(train_examples)} examples")
        logger.info(f"Parameters: iterations={iterations}, dropout={dropout}, batch_size={batch_size}")
        
        # Training statistics
        losses = {}
        training_stats = {
            "iterations": iterations,
            "examples_count": len(train_examples),
            "losses": [],
            "start_time": datetime.now().isoformat()
        }
        
        # Get other pipes to disable during training
        pipe_exceptions = ["ner"]
        unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]
        
        # Training loop
        # Training loop
        with self.nlp.disable_pipes(*unaffected_pipes):
            optimizer = self.nlp.begin_training()
            
            for iteration in range(iterations):
                random.shuffle(train_examples)
                
                batches = minibatch(train_examples, size=batch_size)
                
                for batch in batches:
                    self.nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)
                
                # Log progress
                if iteration % 5 == 0:
                    logger.info(f"Iteration {iteration}, Losses: {losses}")
                    training_stats["losses"].append({
                        "iteration": iteration,
                        "loss": losses.get("ner", 0.0)
                    })
        
        training_stats["end_time"] = datetime.now().isoformat()
        training_stats["final_loss"] = losses.get("ner", 0.0)
        
        logger.info(f"Training completed. Final loss: {losses.get('ner', 0.0):.4f}")
        return training_stats
    
    def save_model(self, output_dir: str) -> bool:
        """Save the trained model"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.nlp.to_disk(output_path)
            
            # Save training metadata
            metadata = {
                "model_name": self.model_name,
                "training_examples": len(self.training_data),
                "save_time": datetime.now().isoformat(),
                "labels": list(self.ner.labels)
            }
            
            with open(output_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Model saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
