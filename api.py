from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import text,create_engine

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import Config
from data.whatsapp_parser import WhatsAppDataParser
from data.stock_processor import StockDataProcessor
from processing.text_preprocessor import MessagePreprocessor
from models.sentiment_analyzer import SentimentAnalyzer
from models.stock_predictor import StockPredictor
from models.prediction_reasoning import PredictionReasoning

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for loaded models
sentiment_analyzer = None
stock_predictor = None
reasoning_engine = None
preprocessor = None
stock_processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    global sentiment_analyzer, stock_predictor, reasoning_engine, preprocessor, stock_processor
    
    logger.info("Loading models...")
    
    try:
        # Load sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        logger.info("✅ Sentiment analyzer loaded")
        
        # Load stock predictor
        stock_predictor = StockPredictor()
        predictor_path = Config.MODELS_DIR / "trained" / "stock_predictor.joblib"
        
        if predictor_path.exists():
            stock_predictor.load_model(str(predictor_path))
            logger.info("✅ Stock predictor loaded")
        else:
            logger.warning("⚠️ Stock predictor model not found - predictions will be sentiment-based only")
        
        # Load other components
        reasoning_engine = PredictionReasoning()
        preprocessor = MessagePreprocessor()
        stock_processor = StockDataProcessor()
        
        logger.info("✅ All models loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Sentiment Stock Prediction API",
    description="AI-powered stock prediction using WhatsApp sentiment analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class MessageInput(BaseModel):
    text: str = Field(..., description="Message text to analyze")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")
    author: Optional[str] = Field(default=None, description="Message author")

class BatchMessageInput(BaseModel):
    messages: List[MessageInput] = Field(..., description="List of messages to analyze")
    company_filter: Optional[str] = Field(default=None, description="Filter predictions for specific company")

class PredictionResponse(BaseModel):
    company: str
    prediction_window: str = "next_3_days"
    direction: str
    confidence: float
    sentiment_score: float
    hype_score: float
    reasoning: List[str]
    metadata: Dict[str, Any] = {}

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    models_loaded: Dict[str, bool]
    version: str = "1.0.0"

class SentimentResponse(BaseModel):
    sentiment: str
    scores: Dict[str, float]
    confidence: float
    individual_models: Optional[Dict[str, Dict[str, float]]] = None

@app.get("/predict/company/{symbol}", response_model=PredictionResponse)
async def predict_company_movement(symbol: str, days_back: int = 7):
    """
    Get prediction for a specific company based on recent sentiment data
    
    Args:
        symbol: Company symbol (e.g., "SAMP.X0000" or "SAMP")
        days_back: Number of days to look back for sentiment data (default: 7)
    """
    try:
        start_time = datetime.now()
        
        # Normalize symbol format
        if not symbol.endswith('.N0000') and not symbol.endswith('.X0000'):
            # Try both formats
            normalized_symbol = symbol + '.N0000'
            alt_symbol = symbol + '.X0000'
        else:
            normalized_symbol = symbol
            alt_symbol = None
        
        # Get recent messages for this company from database
        company_messages = await _get_recent_company_messages(
            normalized_symbol, alt_symbol, days_back
        )
        
        if company_messages.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No recent sentiment data found for {symbol}"
            )
        
        # Calculate aggregated sentiment
        sentiment_scores = company_messages['sentiment_score'].tolist()
        confidences = company_messages['confidence'].tolist()
        
        avg_sentiment = np.mean(sentiment_scores)
        avg_confidence = np.mean(confidences)
        sentiment_std = np.std(sentiment_scores)
        
        # Make prediction based on sentiment
        if avg_sentiment > 0.2:
            prediction = "UP"
        elif avg_sentiment < -0.2:
            prediction = "DOWN"
        else:
            prediction = "HOLD"
        
        # Generate reasoning using the reasoning engine
        reasoning_analysis = reasoning_engine.generate_reasoning(
            company_messages=company_messages,
            sentiment_score=avg_sentiment,
            confidence=avg_confidence,
            prediction=prediction
        )
        
        # Get clean company name (remove suffix)
        clean_company = normalized_symbol.replace('.N0000', '').replace('.X0000', '')
        
        # Create response
        response = PredictionResponse(
            company=clean_company,
            prediction_window="next_3_days",
            direction=prediction,
            confidence=round(avg_confidence, 2),
            sentiment_score=round(avg_sentiment, 2),
            hype_score=reasoning_analysis['hype_score'],
            reasoning=reasoning_analysis['reasoning'],
            metadata={
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "message_count": len(company_messages),
                "sentiment_std": round(sentiment_std, 3),
                "days_analyzed": days_back,
                "unique_senders": company_messages['from_name'].nunique()
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in company prediction for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _get_recent_company_messages(symbol: str, alt_symbol: str = None, days_back: int = 7) -> pd.DataFrame:
    """
    Get recent messages mentioning the specified company from database
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Create database engine
        engine = create_engine(Config.get_database_url())
        
        # Build the query with JSONB search
        # Use company_mentions::text to convert JSONB to text for LIKE search
        # OR use @> operator to check if JSONB contains the value
        
        query = text("""
            SELECT
                message_body,
                sentiment_score,
                confidence,
                from_name,
                timestamp_formatted,
                cleaned_text,
                company_mentions
            FROM processed_messages
            WHERE
                (
                    company_mentions::text LIKE :symbol1 
                    OR company_mentions::text LIKE :symbol2
                )
                AND timestamp_formatted >= :start_date
                AND is_financial = true
            ORDER BY timestamp_formatted DESC
            LIMIT 100
        """)
        
        params = {
            'symbol1': f'%{symbol}%',
            'symbol2': f'%{alt_symbol}%' if alt_symbol else f'%{symbol}%',
            'start_date': start_date
        }
        
        with engine.connect() as conn:
            result = conn.execute(query, params)
            rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(rows, columns=result.keys())
            
            logger.info(f"Found {len(df)} messages for {symbol}")
            return df
            
    except Exception as e:
        logger.error(f"Error getting recent messages for {symbol}: {e}")
        raise
# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        models_loaded={
            "sentiment_analyzer": sentiment_analyzer is not None,
            "stock_predictor": stock_predictor is not None and stock_predictor.is_trained,
            "reasoning_engine": reasoning_engine is not None,
            "preprocessor": preprocessor is not None
        }
    )

@app.post("/predict/single", response_model=PredictionResponse)
async def predict_single_message(message: MessageInput):
    """Predict stock movement from a single message"""
    try:
        start_time = datetime.now()
        
        # Preprocess message
        df = pd.DataFrame([{
            'message_body': message.text,
            'timestamp_formatted': message.timestamp or datetime.now(),
            'from_name': message.author or 'unknown',
            'group_name': 'API'
        }])
        
        processed_messages = preprocessor.process_messages(df)
        
        if processed_messages.empty or not processed_messages.iloc[0].get('is_financial'):
            raise HTTPException(
                status_code=400, 
                detail="Message does not appear to be financial in nature"
            )
        
        # Get company mentions
        row = processed_messages.iloc[0]
        company_mentions = row.get('company_mentions', [])
        
        if not company_mentions:
            raise HTTPException(
                status_code=400,
                detail="No company mentions detected in message"
            )
        
        # Use first mentioned company
        company = company_mentions[0]
        company_ticker = company['ticker']
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze_sentiment(row['cleaned_text'])
        sentiment_score = sentiment_result['scores']['compound']
        confidence = sentiment_result['confidence']
        
        # Make prediction
        if sentiment_score > 0.2:
            prediction = "UP"
        elif sentiment_score < -0.2:
            prediction = "DOWN"
        else:
            prediction = "HOLD"
        
        # Generate reasoning
        company_messages = processed_messages
        reasoning_analysis = reasoning_engine.generate_reasoning(
            company_messages=company_messages,
            sentiment_score=sentiment_score,
            confidence=confidence,
            prediction=prediction
        )
        
        # Create response
        response = PredictionResponse(
            company=company_ticker.replace('.N0000', ''),
            direction=prediction,
            confidence=round(confidence, 2),
            sentiment_score=round(sentiment_score, 2),
            hype_score=reasoning_analysis['hype_score'],
            reasoning=reasoning_analysis['reasoning'],
            metadata={
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "company_confidence": company['confidence'],
                "message_length": len(message.text)
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add new response models
class SenderAnalysisResponse(BaseModel):
    session_id: str
    analysis_date: datetime
    total_participants: int
    group_distribution: Dict[str, str]
    top_movers: List[Dict[str, Any]]
    top_sellers: List[Dict[str, Any]]
    insights: List[str]

class SenderHistoryResponse(BaseModel):
    sender_name: str
    history: List[Dict[str, Any]]
    performance_trend: str

# Update sender analysis endpoint
@app.get("/analysis/senders", response_model=SenderAnalysisResponse)
async def get_latest_sender_analysis():
    """Get latest sender impact analysis from database"""
    try:
        from models.sender_impact_analyzer import SenderImpactAnalyzer
        print(1)
        analyzer = SenderImpactAnalyzer()
        analysis = analyzer.get_latest_analysis()
        print("Analysis : ",analysis)
        if not analysis:
            raise HTTPException(status_code=404, detail="No sender analysis found")
        
        # Extract top performers
        rankings = analysis.get('sender_rankings', [])
        top_movers = [r for r in rankings if r['classification'] == 'mover'][:5]
        top_sellers = [r for r in rankings if r['classification'] == 'seller'][:5]
        
        return SenderAnalysisResponse(
            session_id=analysis['session_id'],
            analysis_date=analysis['created_at'],
            total_participants=analysis['analysis_summary']['total_participants'],
            group_distribution=analysis['analysis_summary']['group_distribution'],
            top_movers=top_movers,
            top_sellers=top_sellers,
            insights=analysis['analysis_summary']['insights']
        )
        
    except Exception as e:
        logger.error(f"Error in sender analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint for sender history
@app.get("/analysis/sender/{sender_name}", response_model=SenderHistoryResponse)
async def get_sender_history(sender_name: str):
    """Get historical performance for a specific sender"""
    try:
        from models.sender_impact_analyzer import SenderImpactAnalyzer
        
        analyzer = SenderImpactAnalyzer()
        history = analyzer.get_sender_history(sender_name)
        
        if not history:
            raise HTTPException(status_code=404, detail=f"No history found for sender: {sender_name}")
        
        # Calculate performance trend
        if len(history) >= 2:
            recent_confidence = history[0]['confidence_score']
            old_confidence = history[-1]['confidence_score']
            
            if recent_confidence > old_confidence + 0.1:
                trend = "improving"
            elif recent_confidence < old_confidence - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return SenderHistoryResponse(
            sender_name=sender_name,
            history=history,
            performance_trend=trend
        )
        
    except Exception as e:
        logger.error(f"Error getting sender history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# New endpoint to trigger analysis
@app.post("/analysis/run-sender-analysis")
async def run_sender_analysis():
    """Trigger a new sender impact analysis"""
    try:
        # This would integrate with your main pipeline
        # For now, return a placeholder response
        return {
            "status": "analysis_triggered",
            "message": "Sender impact analysis has been queued for processing",
            "estimated_completion": "5-10 minutes"
        }
        
    except Exception as e:
        logger.error(f"Error triggering analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_messages(batch_input: BatchMessageInput):
    """Predict stock movements from multiple messages"""
    try:
        start_time = datetime.now()
        
        # Convert to DataFrame
        messages_data = []
        for i, msg in enumerate(batch_input.messages):
            messages_data.append({
                'id': i,
                'message_body': msg.text,
                'timestamp_formatted': msg.timestamp or datetime.now(),
                'from_name': msg.author or f'user_{i}',
                'group_name': 'API_BATCH'
            })
        
        df = pd.DataFrame(messages_data)
        
        # Preprocess messages
        processed_messages = preprocessor.process_messages(df)
        financial_messages = processed_messages[processed_messages.get('is_financial', False)]
        
        if financial_messages.empty:
            raise HTTPException(
                status_code=400,
                detail="No financial messages found in batch"
            )
        
        # Analyze sentiment for all messages
        sentiment_results = []
        for _, row in financial_messages.iterrows():
            result = sentiment_analyzer.analyze_sentiment(row['cleaned_text'])
            sentiment_results.append(result)
        
        financial_messages = financial_messages.copy()
        financial_messages['sentiment_analysis'] = sentiment_results
        financial_messages['sentiment_score'] = financial_messages['sentiment_analysis'].apply(
            lambda x: x['scores']['compound']
        )
        financial_messages['confidence'] = financial_messages['sentiment_analysis'].apply(
            lambda x: x['confidence']
        )
        
        # Get unique companies
        all_companies = set()
        for _, row in financial_messages.iterrows():
            if row.get('company_mentions'):
                for mention in row['company_mentions']:
                    all_companies.add(mention['ticker'])
        
        # Apply company filter if specified
        if batch_input.company_filter:
            all_companies = {c for c in all_companies if batch_input.company_filter.upper() in c}
        
        predictions = []
        
        for company_ticker in all_companies:
            # Filter messages for this company
            company_messages = financial_messages[
                financial_messages['company_mentions'].apply(
                    lambda mentions: any(mention['ticker'] == company_ticker for mention in mentions)
                )
            ].copy()
            
            if len(company_messages) < 1:
                continue
            
            # Calculate sentiment
            sentiment_scores = [r['scores']['compound'] for r in company_messages['sentiment_analysis']]
            avg_sentiment = np.mean(sentiment_scores)
            avg_confidence = np.mean([r['confidence'] for r in company_messages['sentiment_analysis']])
            
            # Make prediction
            if avg_sentiment > 0.2:
                prediction = "UP"
            elif avg_sentiment < -0.2:
                prediction = "DOWN"
            else:
                prediction = "HOLD"
            
            # Generate reasoning
            reasoning_analysis = reasoning_engine.generate_reasoning(
                company_messages=company_messages,
                sentiment_score=avg_sentiment,
                confidence=avg_confidence,
                prediction=prediction
            )
            
            # Create prediction
            pred = PredictionResponse(
                company=company_ticker.replace('.N0000', ''),
                direction=prediction,
                confidence=round(avg_confidence, 2),
                sentiment_score=round(avg_sentiment, 2),
                hype_score=reasoning_analysis['hype_score'],
                reasoning=reasoning_analysis['reasoning'],
                metadata={
                    "message_count": len(company_messages),
                    "avg_sentiment": round(avg_sentiment, 3),
                    "sentiment_std": round(np.std(sentiment_scores), 3)
                }
            )
            
            predictions.append(pred)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create summary
        summary = {
            "total_messages": len(batch_input.messages),
            "financial_messages": len(financial_messages),
            "companies_analyzed": len(predictions),
            "predictions_by_direction": {
                "UP": len([p for p in predictions if p.direction == "UP"]),
                "DOWN": len([p for p in predictions if p.direction == "DOWN"]),
                "HOLD": len([p for p in predictions if p.direction == "HOLD"])
            },
            "average_confidence": round(np.mean([p.confidence for p in predictions]), 3) if predictions else 0,
            "average_sentiment": round(np.mean([p.sentiment_score for p in predictions]), 3) if predictions else 0
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
            processing_time=round(processing_time, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment/analyze", response_model=SentimentResponse)
async def analyze_sentiment(message: MessageInput):
    """Analyze sentiment of a single message"""
    try:
        if not message.text.strip():
            raise HTTPException(status_code=400, detail="Message text cannot be empty")
        
        # Clean text
        cleaned_text = preprocessor.clean_text(message.text)
        
        # Analyze sentiment
        result = sentiment_analyzer.analyze_sentiment(cleaned_text, include_individual=True)
        
        return SentimentResponse(
            sentiment=result['sentiment'],
            scores=result['scores'],
            confidence=result['confidence'],
            individual_models=result.get('individual_scores')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/companies", response_model=List[Dict[str, str]])
async def get_supported_companies():
    """Get list of supported companies"""
    try:
        companies = stock_processor.load_symbols()
        return [
            {
                "ticker": company['ticker'],
                "name": company['company_name'],
                "aliases": ', '.join(company.get('aliases', []))
            }
            for company in companies
        ]
    except Exception as e:
        logger.error(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return await root()

# Run configuration
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=Config.API_RELOAD,
        log_level="info"
    )