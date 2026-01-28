#!/usr/bin/env python3
"""
Train the LightGBM classifier using local stock data.

Usage:
    python train_classifier.py

This script uses the pre-downloaded stock data in the 'data' directory.
No internet connection required.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("data")
PRICES_DIR = DATA_DIR / "prices"
INFO_DIR = DATA_DIR / "stock_info"
MODELS_DIR = Path("models")


def get_available_stocks():
    """Get list of stocks that have both price and info data."""
    price_files = set(f.stem for f in PRICES_DIR.glob("*.parquet"))
    info_files = set(f.stem for f in INFO_DIR.glob("*.json"))
    return sorted(price_files & info_files)


def load_stock_data(symbol: str):
    """Load price and info data for a stock."""
    try:
        # Load prices
        price_file = PRICES_DIR / f"{symbol}.parquet"
        prices_df = pd.read_parquet(price_file)
        prices_df['date'] = pd.to_datetime(prices_df['date'])
        
        # Load info
        info_file = INFO_DIR / f"{symbol}.json"
        with open(info_file) as f:
            raw_info = json.load(f)
        
        # Normalize info
        info = {
            'symbol': symbol,
            'name': raw_info.get('longName', raw_info.get('shortName', symbol)),
            'sector': raw_info.get('sector', 'Unknown'),
            'industry': raw_info.get('industry', 'Unknown'),
            'market_cap': raw_info.get('marketCap', 0),
            'current_price': raw_info.get('currentPrice', raw_info.get('regularMarketPrice', 0)),
            'pe_ratio': raw_info.get('trailingPE'),
            'pb_ratio': raw_info.get('priceToBook'),
            'dividend_yield': (raw_info.get('dividendYield', 0) or 0) * 100,
            'eps': raw_info.get('trailingEps'),
            'book_value': raw_info.get('bookValue'),
            'roe': (raw_info.get('returnOnEquity') or 0) * 100 if raw_info.get('returnOnEquity') else None,
            'debt_to_equity': raw_info.get('debtToEquity'),
            'fifty_two_week_high': raw_info.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': raw_info.get('fiftyTwoWeekLow'),
        }
        
        return prices_df, info
    except Exception as e:
        return None, None


def analyze_stock_simple(symbol: str, prices_df: pd.DataFrame, info: dict):
    """
    Simple analysis to generate features and labels.
    Uses rule-based logic similar to the main app.
    """
    if prices_df is None or prices_df.empty or len(prices_df) < 60:
        return None, None
    
    try:
        # Get price series
        if 'close' in prices_df.columns:
            prices = prices_df['close'].values
        else:
            prices = prices_df.iloc[:, 3].values  # Assume 4th column is close
        
        if len(prices) < 60:
            return None, None
        
        # Calculate features
        features = {}
        
        # Price-based features
        features['return_1d'] = (prices[-1] / prices[-2] - 1) * 100 if len(prices) >= 2 else 0
        features['return_5d'] = (prices[-1] / prices[-5] - 1) * 100 if len(prices) >= 5 else 0
        features['return_20d'] = (prices[-1] / prices[-20] - 1) * 100 if len(prices) >= 20 else 0
        features['return_60d'] = (prices[-1] / prices[-60] - 1) * 100 if len(prices) >= 60 else 0
        features['return_252d'] = (prices[-1] / prices[-252] - 1) * 100 if len(prices) >= 252 else 0
        
        # Volatility
        returns = np.diff(prices) / prices[:-1]
        features['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else 0
        features['volatility_60d'] = np.std(returns[-60:]) * np.sqrt(252) * 100 if len(returns) >= 60 else 0
        
        # Moving average ratios
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else prices[-1]
        features['sma_ratio_20'] = prices[-1] / sma_20 if sma_20 > 0 else 1
        features['sma_ratio_50'] = prices[-1] / sma_50 if sma_50 > 0 else 1
        features['sma_ratio_200'] = prices[-1] / sma_200 if sma_200 > 0 else 1
        
        # 52-week high/low position
        high_52w = info.get('fifty_two_week_high') or np.max(prices[-252:]) if len(prices) >= 252 else np.max(prices)
        low_52w = info.get('fifty_two_week_low') or np.min(prices[-252:]) if len(prices) >= 252 else np.min(prices)
        features['high_52w_pct'] = (prices[-1] / high_52w) if high_52w > 0 else 1
        features['low_52w_pct'] = (prices[-1] / low_52w) if low_52w > 0 else 1
        
        # RSI
        gains = np.maximum(np.diff(prices[-15:]), 0)
        losses = np.abs(np.minimum(np.diff(prices[-15:]), 0))
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Fundamental features from info
        features['pe_ratio'] = info.get('pe_ratio') or 0
        features['pb_ratio'] = info.get('pb_ratio') or 0
        features['dividend_yield'] = info.get('dividend_yield') or 0
        features['roe'] = info.get('roe') or 0
        features['debt_to_equity'] = info.get('debt_to_equity') or 0
        features['market_cap'] = np.log10(info.get('market_cap', 1e9) + 1)  # Log scale
        
        # Generate label based on simple rules
        score = 50  # Base score
        
        # Price momentum (30 points max)
        if features['return_252d'] > 20:
            score += 15
        elif features['return_252d'] > 10:
            score += 10
        elif features['return_252d'] < -20:
            score -= 15
        elif features['return_252d'] < -10:
            score -= 10
        
        if features['sma_ratio_200'] > 1.1:
            score += 10
        elif features['sma_ratio_200'] < 0.9:
            score -= 10
        
        # Valuation (20 points max)
        pe = features['pe_ratio']
        if 0 < pe < 15:
            score += 10
        elif 15 <= pe < 25:
            score += 5
        elif pe > 40:
            score -= 10
        
        # Quality (20 points max)
        roe = features['roe']
        if roe > 20:
            score += 10
        elif roe > 15:
            score += 5
        elif roe < 5:
            score -= 5
        
        de = features['debt_to_equity']
        if de is not None and de < 0.5:
            score += 5
        elif de is not None and de > 2:
            score -= 10
        
        # Risk (volatility penalty)
        if features['volatility_60d'] > 40:
            score -= 10
        elif features['volatility_60d'] > 30:
            score -= 5
        
        # Determine signal (adjusted thresholds for balanced distribution)
        if score >= 65:
            label = "BUY"
        elif score >= 45:
            label = "HOLD"
        elif score >= 30:
            label = "AVOID"
        else:
            label = "SELL"
        
        return features, label
        
    except Exception as e:
        logger.debug(f"Analysis failed for {symbol}: {e}")
        return None, None


def main():
    print("\n" + "=" * 60)
    print("  LightGBM Classifier Training (Local Data)")
    print("=" * 60)
    
    # Check data directory
    if not PRICES_DIR.exists() or not INFO_DIR.exists():
        print("\n❌ Error: Data directory not found!")
        print("   Make sure you have stock data in the 'data' directory.")
        return False
    
    # Get available stocks
    stocks = get_available_stocks()
    print(f"\n📊 Found {len(stocks)} stocks with local data")
    
    if len(stocks) < 50:
        print("❌ Not enough stocks for training (need at least 50)")
        return False
    
    # Sample stocks for training (use up to 500 for speed)
    sample_size = min(500, len(stocks))
    
    # Prioritize well-known stocks
    priority_stocks = [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR",
        "ITC", "KOTAKBANK", "SBIN", "BAJFINANCE", "BHARTIARTL", "ASIANPAINT",
        "MARUTI", "TITAN", "NESTLEIND", "SUNPHARMA", "HCLTECH", "WIPRO",
        "TECHM", "AXISBANK", "ULTRACEMCO", "DRREDDY", "POWERGRID", "NTPC",
        "TATASTEEL", "JSWSTEEL", "ONGC", "COALINDIA", "PERSISTENT", "LTIM",
    ]
    
    # Build training set: priority stocks first, then random sample
    training_stocks = [s for s in priority_stocks if s in stocks]
    remaining = [s for s in stocks if s not in training_stocks]
    np.random.seed(42)
    np.random.shuffle(remaining)
    training_stocks.extend(remaining[:sample_size - len(training_stocks)])
    
    print(f"📈 Training on {len(training_stocks)} stocks...")
    
    # Collect training data
    training_data = []
    labels = []
    successful = 0
    
    for i, symbol in enumerate(training_stocks):
        if (i + 1) % 50 == 0:
            print(f"   Processing... {i + 1}/{len(training_stocks)}")
        
        prices_df, info = load_stock_data(symbol)
        if prices_df is None:
            continue
        
        features, label = analyze_stock_simple(symbol, prices_df, info)
        if features and label:
            training_data.append(features)
            labels.append(label)
            successful += 1
    
    print(f"\n✓ Successfully processed {successful}/{len(training_stocks)} stocks")
    
    if successful < 50:
        print("❌ Not enough valid training samples")
        return False
    
    # Create DataFrame
    X = pd.DataFrame(training_data)
    y = pd.Series(labels)
    
    # Fill NaN
    X = X.fillna(0)
    
    print(f"\n📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Label distribution:")
    for label, count in y.value_counts().items():
        print(f"     {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train classifier
    print("\n🔧 Training classifier...")
    
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score
    
    # Encode labels
    label_map = {'BUY': 0, 'HOLD': 1, 'AVOID': 2, 'SELL': 3}
    y_encoded = y.map(label_map)
    
    # Walk-Forward Validation: Use time-series split to avoid look-ahead bias
    print("   Using Walk-Forward Validation (time-series split)")
    
    X = X.sort_index()
    y_encoded = y_encoded.reindex(X.index)
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    for train_idx, test_idx in tscv.split(X):
        pass  # Get last split
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded.iloc[train_idx], y_encoded.iloc[test_idx]
    
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    model = None
    model_type = "lightgbm"
    
    # Try LightGBM first
    try:
        import lightgbm as lgb
        print("   Using LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=10,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        
    except Exception as e:
        print(f"   LightGBM failed: {str(e)[:50]}...")
        print("   Falling back to sklearn GradientBoosting...")
        
        # Fallback to sklearn
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            model_type = "sklearn_gb"
            
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
            )
            model.fit(X_train, y_train)
            
        except Exception as e2:
            print(f"\n❌ Training failed: {e2}")
            print("\n💡 To fix LightGBM on Mac, run:")
            print("   brew install libomp")
            print("   pip install --force-reinstall lightgbm")
            return False
    
    if model is None:
        print("\n❌ No classifier model available")
        return False
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Training complete! (using {model_type})")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   F1 Score: {f1:.2%}")
    
    # Feature importance
    try:
        importance = dict(zip(X.columns, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\n📊 Top 10 Important Features:")
        for i, (feat, imp) in enumerate(sorted_imp, 1):
            print(f"   {i}. {feat}: {imp:.3f}")
    except:
        pass
    
    # Save model
    model_dir = MODELS_DIR / "classifier" / "lightgbm"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    import pickle
    model_path = model_dir / "model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': list(X.columns),
            'label_map': label_map,
            'metrics': {'accuracy': accuracy, 'f1': f1},
            'model_type': model_type,
        }, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    print("\n🎉 Classifier training complete! You can now run the app.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
