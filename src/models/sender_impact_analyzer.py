import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import json
import uuid
import psycopg2
from sqlalchemy import create_engine, text
from config import Config

logger = logging.getLogger(__name__)

class SenderImpactAnalyzer:
    """Analyzes sender impact on stock price movements with PostgreSQL storage - FIXED VERSION"""
    
    def __init__(self):
        self.sender_profiles = {}
        self.impact_thresholds = {
            'mover': 0.15,      # 15%+ positive correlation = price mover
            'seller': -0.15,    # 15%+ negative correlation = price seller  
            'neutral': 0.05     # Within 5% = neutral
        }
        self.engine = self._create_db_engine()
        
    def _create_db_engine(self):
        """Create database engine"""
        try:
            db_url = f"postgresql://{Config.DB_USER}:{Config.DB_PASSWORD}@{Config.DB_HOST}:{Config.DB_PORT}/{Config.DB_NAME}"
            return create_engine(db_url)
        except Exception as e:
            logger.error(f"Error creating database engine: {e}")
            return None
    
    def analyze_sender_impact(
        self,
        messages_df: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        analysis_window_days: int = 7,
        save_to_db: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze impact of each sender on stock price movements
        """
        logger.info("Starting sender impact analysis...")
        
        # Generate unique session ID
        session_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Get messages with company mentions and sentiment
        financial_messages = messages_df[
            (messages_df.get('is_financial', False)) & 
            (messages_df['company_mentions'].apply(lambda x: len(x) > 0 if x else False))
        ].copy()
        
        if financial_messages.empty:
            return {"error": "No financial messages found"}
        
        # Calculate sender impacts
        sender_impacts = self._calculate_sender_impacts(
            financial_messages, stock_data, analysis_window_days
        )
        
        # Classify senders into groups
        sender_groups = self._classify_senders(sender_impacts)
        
        # Calculate group statistics
        group_stats = self._calculate_group_statistics(sender_groups)
        
        # Rank senders by impact
        sender_rankings = self._rank_senders(sender_impacts)
        
        analysis_result = {
            "session_id": session_id,
            "sender_impacts": sender_impacts,
            "sender_groups": sender_groups,
            "group_statistics": group_stats,
            "sender_rankings": sender_rankings,
            "analysis_summary": self._create_analysis_summary(sender_groups, group_stats)
        }
        
        # Save to database if requested
        if save_to_db and self.engine:
            try:
                self._save_to_database(session_id, analysis_result, analysis_window_days)
                logger.info(f"Analysis saved to database with session ID: {session_id}")
            except Exception as e:
                logger.error(f"Error saving to database: {e}")
        
        return analysis_result
    
    def _save_to_database(self, session_id: str, analysis: Dict, window_days: int):
        """Save analysis results to PostgreSQL database - FIXED VERSION"""
        
        with self.engine.connect() as conn:
            # Save analysis session
            session_data = analysis['analysis_summary']
            
            conn.execute(text("""
                INSERT INTO sender_analysis_sessions 
                (session_id, analysis_window_days, total_participants, classified_participants, 
                 classification_rate, dominant_group, insights, metadata)
                VALUES (:session_id, :window_days, :total_participants, :classified_participants,
                        :classification_rate, :dominant_group, :insights, :metadata)
            """), {
                'session_id': session_id,
                'window_days': window_days,
                'total_participants': session_data['total_participants'],
                'classified_participants': session_data['classified_participants'],
                'classification_rate': float(session_data['classification_rate']),
                'dominant_group': session_data['dominant_group'],
                'insights': session_data['insights'],
                'metadata': json.dumps(session_data.get('group_distribution', {}))
            })
            
            # Save sender impacts
            for sender, data in analysis['sender_impacts'].items():
                conn.execute(text("""
                    INSERT INTO sender_impacts 
                    (session_id, sender_name, correlation, p_value, total_messages, 
                     companies_count, companies, accuracy_rate, avg_impact, impact_std,
                     significant_correlation, confidence_score, classification, successful_impacts)
                    VALUES (:session_id, :sender_name, :correlation, :p_value, :total_messages,
                            :companies_count, :companies, :accuracy_rate, :avg_impact, :impact_std,
                            :significant_correlation, :confidence_score, :classification, :successful_impacts)
                """), {
                    'session_id': session_id,
                    'sender_name': sender,
                    'correlation': float(data['correlation']),
                    'p_value': float(data['p_value']),
                    'total_messages': int(data['total_messages']),
                    'successful_impacts': int(data.get('successful_impacts', 0)),  # NEW FIELD
                    'companies_count': int(data['companies_count']),
                    'companies': data['companies'],
                    'accuracy_rate': float(data['accuracy_rate']),
                    'avg_impact': float(data['avg_impact']),
                    'impact_std': float(data['impact_std']),
                    'significant_correlation': bool(data['significant_correlation']),
                    'confidence_score': float(data['confidence_score']),
                    'classification': self._get_sender_classification(data['correlation'], data['confidence_score'])
                })
            
            # Save group statistics
            for group_name, stats in analysis['group_statistics'].items():
                conn.execute(text("""
                    INSERT INTO sender_groups 
                    (session_id, group_name, sender_count, percentage, sender_names)
                    VALUES (:session_id, :group_name, :sender_count, :percentage, :sender_names)
                """), {
                    'session_id': session_id,
                    'group_name': group_name,
                    'sender_count': int(stats['count']),
                    'percentage': float(stats['percentage']),
                    'sender_names': stats['senders']
                })
            
            conn.commit()
    
    def _calculate_sender_impacts(
        self,
        messages_df: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        window_days: int
    ) -> Dict[str, Dict]:
        """Calculate impact correlation for each sender - FIXED VERSION"""
        
        sender_impacts = defaultdict(lambda: {
            'total_messages': 0,
            'companies_mentioned': set(),
            'price_impacts': [],
            'sentiment_accuracy': [],
            'avg_sentiment': 0,
            'significant_calls': 0
        })
        
        print(f"\n🔍 PROCESSING {len(messages_df)} FINANCIAL MESSAGES...")
        processed_count = 0
        successful_impacts = 0
        failed_impacts = 0
        
        # Process each message
        for idx, row in messages_df.iterrows():
            sender = row['from_name']
            message_date = pd.to_datetime(row['timestamp_formatted']).date()
            sentiment_score = row.get('sentiment_score', 0)
            
            print(f"📝 Processing message {idx}: {sender} on {message_date}, sentiment: {sentiment_score:.3f}")
            
            # Process each company mentioned in the message
            company_mentions = row.get('company_mentions', [])
            if not company_mentions:
                print(f"   ❌ No company mentions")
                continue
                
            for mention in company_mentions:
                company_ticker = mention['ticker']
                print(f"   🏢 Company: {company_ticker}")
                
                if company_ticker not in stock_data:
                    print(f"   ❌ No stock data for {company_ticker}")
                    sender_impacts[sender]['total_messages'] += 1  # Count failed attempt
                    sender_impacts[sender]['companies_mentioned'].add(company_ticker)
                    failed_impacts += 1
                    continue
                    
                print(f"   ✅ Stock data available for {company_ticker}")
                
                # Get stock data for this company
                stock_df = stock_data[company_ticker].copy()
                
                # Find price impact after the message
                impact = self._calculate_price_impact(
                    stock_df, message_date, window_days, sentiment_score
                )
                
                if impact is not None:
                    print(f"   ✅ SUCCESSFUL IMPACT: {impact['price_change']:.4f}")
                    sender_impacts[sender]['total_messages'] += 1
                    sender_impacts[sender]['companies_mentioned'].add(company_ticker)
                    sender_impacts[sender]['price_impacts'].append(impact)
                    
                    # Track sentiment accuracy (did sentiment match price direction?)
                    sentiment_correct = (sentiment_score * impact['price_change']) > 0
                    sender_impacts[sender]['sentiment_accuracy'].append(sentiment_correct)
                    successful_impacts += 1
                else:
                    print(f"   ❌ Price impact calculation failed")
                    sender_impacts[sender]['total_messages'] += 1  # Count failed attempt
                    sender_impacts[sender]['companies_mentioned'].add(company_ticker)
                    failed_impacts += 1
            
            processed_count += 1
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   Messages processed: {processed_count}")
        print(f"   Successful impacts: {successful_impacts}")
        print(f"   Failed impacts: {failed_impacts}")
        print(f"   Senders with data: {len(sender_impacts)}")
        
        # Calculate final metrics for each sender
        final_impacts = {}
        for sender, data in sender_impacts.items():
            print(f"\n👤 SENDER: {sender}")
            print(f"   Total messages: {data['total_messages']}")
            print(f"   Companies: {list(data['companies_mentioned'])}")
            print(f"   Successful price impacts: {len(data['price_impacts'])}")
            
            impacts = data['price_impacts']
            accuracies = data['sentiment_accuracy']
            successful_impact_count = len(impacts)
            
            # FIXED: Only include senders with at least 1 successful impact AND proper correlation handling
            if successful_impact_count >= 2:
                # Sufficient data for correlation calculation
                try:
                    sentiments = [impact['sentiment_prediction'] for impact in impacts]
                    price_changes = [impact['price_change'] for impact in impacts]
                    
                    correlation, p_value = self._safe_correlation(sentiments, price_changes)
                    print(f"   ✅ CORRELATION CALCULATED: {correlation:.4f} (p={p_value:.4f})")
                    
                    final_impacts[sender] = {
                        'correlation': float(correlation),
                        'p_value': float(p_value),
                        'total_messages': int(data['total_messages']),
                        'successful_impacts': int(successful_impact_count),
                        'companies_count': int(len(data['companies_mentioned'])),
                        'companies': list(data['companies_mentioned']),
                        'accuracy_rate': float(np.mean(accuracies)) if accuracies else 0.0,
                        'avg_impact': float(np.mean(price_changes)) if price_changes else 0.0,
                        'impact_std': float(np.std(price_changes)) if price_changes else 0.0,
                        'significant_correlation': bool(p_value < 0.05),
                        'confidence_score': float(self._calculate_confidence(correlation, p_value, successful_impact_count))
                    }
                except Exception as e:
                    print(f"   ❌ Error calculating correlation: {e}")
                    continue
                    
            elif successful_impact_count == 1:
                # FIXED: Handle single impact case with default values instead of NaN
                print(f"   ⚠️  Only 1 successful impact - using default correlation values")
                price_change = impacts[0]['price_change']
                
                final_impacts[sender] = {
                    'correlation': 0.0,  # FIXED: Default instead of NaN
                    'p_value': 1.0,      # FIXED: Default instead of NaN  
                    'total_messages': int(data['total_messages']),
                    'successful_impacts': 1,
                    'companies_count': int(len(data['companies_mentioned'])),
                    'companies': list(data['companies_mentioned']),
                    'accuracy_rate': float(accuracies[0]) if accuracies else 0.0,
                    'avg_impact': float(price_change),
                    'impact_std': 0.0,  # No variation with 1 sample
                    'significant_correlation': False,
                    'confidence_score': 0.1  # Very low confidence
                }
            else:
                # FIXED: Don't include senders with 0 successful impacts
                print(f"   ❌ No successful impacts ({successful_impact_count}) - excluding from results")
                continue
        
        print(f"\n🎯 FINAL RESULT: {len(final_impacts)} senders with valid impact data")
        return final_impacts
    
    def _calculate_price_impact(
        self,
        stock_df: pd.DataFrame,
        message_date: datetime.date,
        window_days: int,
        sentiment_score: float
    ) -> Optional[Dict]:
        """Calculate price impact after a message - FIXED VERSION"""
        
        try:
            # Ensure we have a date column or create one from index
            if 'date' not in stock_df.columns:
                # Stock data likely has datetime in index, reset it
                stock_df_fixed = stock_df.reset_index()
                if 'datetime' in stock_df_fixed.columns:
                    stock_df_fixed['date'] = pd.to_datetime(stock_df_fixed['datetime']).dt.date
                elif stock_df_fixed.index.name == 'datetime':
                    stock_df_fixed['date'] = pd.to_datetime(stock_df_fixed.index).date
                else:
                    # If no datetime info, skip this entry
                    print(f"   ❌ No datetime info in stock data")
                    return None
            else:
                stock_df_fixed = stock_df.copy()
                stock_df_fixed['date'] = pd.to_datetime(stock_df_fixed['date']).dt.date
            
            # Convert message_date to date object if it isn't already
            if isinstance(message_date, str):
                message_date = pd.to_datetime(message_date).date()
            elif hasattr(message_date, 'date'):
                message_date = message_date.date()
            
            # Find exact or closest date match
            stock_dates = stock_df_fixed['date'].tolist()
            
            print(f"   📊 Looking for {message_date} in {len(stock_dates)} stock dates")
            print(f"   📊 Stock date range: {min(stock_dates)} to {max(stock_dates)}")
            
            # Try exact match first
            if message_date in stock_dates:
                start_idx = stock_dates.index(message_date)
                print(f"   ✅ Exact date match at index {start_idx}")
            else:
                # Find closest date within reasonable range (7 days)
                date_diffs = [(abs((d - message_date).days), i) for i, d in enumerate(stock_dates)]
                min_diff, start_idx = min(date_diffs, key=lambda x: x[0])
                
                print(f"   📊 Closest date: {stock_dates[start_idx]}, diff: {min_diff} days")
                
                if min_diff > 7:  # Too far apart
                    print(f"   ❌ Date too far apart ({min_diff} days)")
                    return None
            
            # Check if we have enough future data
            if start_idx + window_days >= len(stock_df_fixed):
                print(f"   ❌ Not enough future data (need {window_days} days, have {len(stock_df_fixed) - start_idx - 1})")
                return None
            
            # Calculate price impact
            start_price = float(stock_df_fixed.iloc[start_idx]['close'])
            end_price = float(stock_df_fixed.iloc[start_idx + window_days]['close'])
            price_change = float((end_price - start_price) / start_price)
            
            # Determine sentiment prediction direction
            sentiment_prediction = 1 if sentiment_score > 0.1 else (-1 if sentiment_score < -0.1 else 0)
            
            print(f"   💰 Price: {start_price:.2f} → {end_price:.2f} = {price_change:.4f} ({price_change*100:.2f}%)")
            
            return {
                'start_price': start_price,
                'end_price': end_price,
                'price_change': price_change,
                'sentiment_prediction': int(sentiment_prediction),
                'days_elapsed': int(window_days)
            }
            
        except Exception as e:
            print(f"   ❌ Error calculating impact for {message_date}: {e}")
            return None
    
    def _calculate_confidence(self, correlation: float, p_value: float, sample_size: int) -> float:
        """Calculate confidence score for sender classification"""
        
        # Base confidence on correlation strength
        base_confidence = abs(float(correlation))
        
        # Adjust for statistical significance
        if float(p_value) < 0.01:
            significance_bonus = 0.3
        elif float(p_value) < 0.05:
            significance_bonus = 0.2
        else:
            significance_bonus = 0.0
        
        # Adjust for sample size
        if int(sample_size) >= 20:
            sample_bonus = 0.2
        elif int(sample_size) >= 10:
            sample_bonus = 0.1
        else:
            sample_bonus = 0.0
        
        return min(1.0, base_confidence + significance_bonus + sample_bonus)
    
    def _classify_senders(self, sender_impacts: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Classify senders into impact groups"""
        
        groups = {
            'movers': [],      # Positive price impact
            'sellers': [],     # Negative price impact  
            'neutrals': [],    # No significant impact
            'unclassified': [] # Insufficient data
        }
        
        for sender, data in sender_impacts.items():
            correlation = float(data['correlation'])
            confidence = float(data['confidence_score'])
            
            # Only classify if confidence is reasonable
            if confidence < 0.3:
                groups['unclassified'].append(sender)
            elif correlation >= self.impact_thresholds['mover']:
                groups['movers'].append(sender)
            elif correlation <= self.impact_thresholds['seller']:
                groups['sellers'].append(sender)
            else:
                groups['neutrals'].append(sender)
        
        return groups
    
    def _calculate_group_statistics(self, sender_groups: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate statistics for each sender group"""
        
        total_senders = sum(len(senders) for senders in sender_groups.values())
        
        stats = {}
        for group_name, senders in sender_groups.items():
            count = len(senders)
            percentage = float((count / total_senders * 100)) if total_senders > 0 else 0.0
            
            stats[group_name] = {
                'count': int(count),
                'percentage': round(percentage, 1),
                'senders': senders
            }
        
        return stats
    
    def _safe_correlation(self, sentiments, price_changes):
        """Safely calculate correlation, handling edge cases that cause NaN"""
        
        # Convert to numpy arrays for easier handling
        sentiments = np.array(sentiments, dtype=float)
        price_changes = np.array(price_changes, dtype=float)
        
        # Check for any NaN values in input
        if np.any(np.isnan(sentiments)) or np.any(np.isnan(price_changes)):
            return 0.0, 1.0
        
        # Check for zero variance (all values the same)
        if np.var(sentiments) == 0 or np.var(price_changes) == 0:
            return 0.0, 1.0
        
        # Check for insufficient unique values
        if len(np.unique(sentiments)) < 2 or len(np.unique(price_changes)) < 2:
            return 0.0, 1.0
        
        try:
            correlation, p_value = stats.pearsonr(sentiments, price_changes)
            
            # Final NaN check
            if np.isnan(correlation) or np.isnan(p_value):
                return 0.0, 1.0
                
            return float(correlation), float(p_value)
            
        except Exception as e:
            print(f"   ❌ Correlation calculation failed: {e}")
            return 0.0, 1.0
        
    
    def _rank_senders(self, sender_impacts: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """Rank senders by their prediction impact"""
        
        rankings = []
        
        for sender, data in sender_impacts.items():
            rankings.append({
                'sender': sender,
                'correlation': round(float(data['correlation']), 3),
                'accuracy_rate': round(float(data['accuracy_rate']), 3),
                'confidence': round(float(data['confidence_score']), 3),
                'total_messages': int(data['total_messages']),
                'successful_impacts': int(data.get('successful_impacts', 0)),  # NEW FIELD
                'avg_impact': round(float(data['avg_impact']), 4),
                'companies_count': int(data['companies_count']),
                'significant': bool(data['significant_correlation']),
                'classification': self._get_sender_classification(data['correlation'], data['confidence_score'])
            })
        
        # Sort by confidence score and correlation
        rankings.sort(key=lambda x: (x['confidence'], abs(x['correlation'])), reverse=True)
        
        return rankings
    
    def _get_sender_classification(self, correlation: float, confidence: float) -> str:
        """Get classification label for a sender"""
        correlation = float(correlation)
        confidence = float(confidence)
        
        if confidence < 0.3:
            return 'unclassified'
        elif correlation >= self.impact_thresholds['mover']:
            return 'mover'
        elif correlation <= self.impact_thresholds['seller']:
            return 'seller'
        else:
            return 'neutral'
    
    def _create_analysis_summary(self, sender_groups: Dict, group_stats: Dict) -> Dict[str, Any]:
        """Create summary of the analysis"""
        
        total_classified = sum(group_stats[group]['count'] for group in ['movers', 'sellers', 'neutrals'])
        total_participants = sum(group_stats[group]['count'] for group in group_stats.keys())
        
        return {
            'total_participants': int(total_participants),
            'classified_participants': int(total_classified),
            'classification_rate': round(float((total_classified / total_participants * 100)) if total_participants > 0 else 0, 1),
            'dominant_group': max(group_stats.keys(), key=lambda k: group_stats[k]['count']) if group_stats else None,
            'group_distribution': {
                group: f"{stats['count']} ({stats['percentage']}%)" 
                for group, stats in group_stats.items()
            },
            'insights': self._generate_insights(group_stats)
        }
    
    def _generate_insights(self, group_stats: Dict) -> List[str]:
        """Generate insights from group analysis"""
        insights = []
        
        total = sum(stats['count'] for stats in group_stats.values())
        
        if total == 0:
            return ["Insufficient data for analysis"]
        
        # Analyze group distribution
        movers_pct = float(group_stats.get('movers', {}).get('percentage', 0))
        sellers_pct = float(group_stats.get('sellers', {}).get('percentage', 0))
        neutrals_pct = float(group_stats.get('neutrals', {}).get('percentage', 0))
        
        if movers_pct > 30:
            insights.append(f"High proportion of price movers ({movers_pct}%) suggests strong bullish influence")
        elif sellers_pct > 30:
            insights.append(f"High proportion of sellers ({sellers_pct}%) suggests bearish sentiment dominance")
        elif neutrals_pct > 50:
            insights.append(f"Majority are neutral participants ({neutrals_pct}%) indicating balanced discussion")
        
        # Most influential group
        max_group = max(group_stats.keys(), key=lambda k: group_stats[k]['count'])
        if group_stats[max_group]['count'] > 0:
            insights.append(f"'{max_group.title()}' group dominates with {group_stats[max_group]['percentage']}% of participants")
        
        return insights
    
    def get_sender_influence_weight(self, sender: str, sender_impacts: Dict) -> float:
        """Get influence weight for a sender (for prediction weighting)"""
        if sender not in sender_impacts:
            return 1.0  # Default weight
        
        data = sender_impacts[sender]
        confidence = float(data['confidence_score'])
        accuracy = float(data['accuracy_rate'])
        
        # Calculate influence weight (0.5 to 2.0)
        base_weight = 1.0
        confidence_bonus = confidence * 0.5  # Up to +0.5
        accuracy_bonus = (accuracy - 0.5) * 0.5  # -0.25 to +0.25
        
        weight = base_weight + confidence_bonus + accuracy_bonus
        return max(0.5, min(2.0, weight))  # Clamp between 0.5 and 2.0

    # Additional methods for database operations remain the same...
    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """Get the most recent analysis from database"""
        if not self.engine:
            return None
            
        try:
            with self.engine.connect() as conn:
                # Get latest session
                result = conn.execute(text("""
                    SELECT * FROM sender_analysis_sessions 
                    ORDER BY created_at DESC LIMIT 1
                """))
                session = result.fetchone()
                
                if not session:
                    return None
                
                session_id = session.session_id
                
                # Get sender impacts for this session
                impacts_result = conn.execute(text("""
                    SELECT * FROM sender_impacts 
                    WHERE session_id = :session_id
                    ORDER BY confidence_score DESC
                """), {'session_id': session_id})
                
                impacts = []
                for row in impacts_result.fetchall():
                    impacts.append({
                        'sender': row.sender_name,
                        'correlation': float(row.correlation),
                        'accuracy_rate': float(row.accuracy_rate),
                        'confidence': float(row.confidence_score),
                        'total_messages': row.total_messages,
                        'classification': row.classification,
                        'companies_count': row.companies_count
                    })
                
                # Get group statistics
                groups_result = conn.execute(text("""
                    SELECT * FROM sender_groups 
                    WHERE session_id = :session_id
                """), {'session_id': session_id})
                
                group_stats = {}
                for row in groups_result.fetchall():
                    group_stats[row.group_name] = {
                        'count': row.sender_count,
                        'percentage': float(row.percentage),
                        'senders': row.sender_names
                    }
                
                return {
                    'session_id': session_id,
                    'created_at': session.created_at,
                    'analysis_summary': {
                        'total_participants': session.total_participants,
                        'classified_participants': session.classified_participants,
                        'classification_rate': float(session.classification_rate),
                        'dominant_group': session.dominant_group,
                        'insights': session.insights if isinstance(session.insights, list) else (session.insights.split('\n') if session.insights else []),
                        'group_distribution': session.metadata if isinstance(session.metadata, dict) else (json.loads(session.metadata) if session.metadata else {})
                    },
                    'sender_rankings': impacts,
                    'group_statistics': group_stats
                }
                
        except Exception as e:
            logger.error(f"Error getting latest analysis: {e}")
            return None
        
    def get_sender_history(self, sender_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get historical performance for a specific sender"""
        if not self.engine:
            return []
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT si.*, s.created_at as analysis_date
                    FROM sender_impacts si
                    JOIN sender_analysis_sessions s ON si.session_id = s.session_id
                    WHERE si.sender_name = :sender_name
                    ORDER BY s.created_at DESC
                    LIMIT :limit
                """), {'sender_name': sender_name, 'limit': limit})
                
                history = []
                for row in result.fetchall():
                    history.append({
                        'analysis_date': row.analysis_date,
                        'correlation': float(row.correlation),
                        'accuracy_rate': float(row.accuracy_rate),
                        'confidence_score': float(row.confidence_score),
                        'classification': row.classification,
                        'total_messages': row.total_messages,
                        'avg_impact': float(row.avg_impact)
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Error getting sender history: {e}")
            return []