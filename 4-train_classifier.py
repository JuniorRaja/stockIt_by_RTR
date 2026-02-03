#!/usr/bin/env python3
"""
Train the classifier using local stock data.

Usage:
    python train_classifier.py

This script uses the pre-downloaded stock data in the 'data' directory.
Classifier type (LightGBM or CatBoost) is read from config/settings.yaml.
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
import yaml
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.ml_models.features import FeatureEngineer
from src.data.macro import get_macro_provider
from scripts.download_delisted_stocks import DelistedStockProvider

# Directories
DATA_DIR = Path("data")
PRICES_DIR = DATA_DIR / "prices"
INFO_DIR = DATA_DIR / "stock_info"
MODELS_DIR = Path("models")
DELISTED_DIR = DATA_DIR / "delisted"
CONFIG_PATH = Path("config/settings.yaml")


def get_classifier_type_from_config() -> str:
    """Read classifier type from config/settings.yaml."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                settings = yaml.safe_load(f)
            classifier_model = settings.get('ml_config', {}).get('classifier', {}).get('model', 'lightgbm')
            if classifier_model in ('lightgbm', 'catboost'):
                return classifier_model
        except Exception as e:
            logger.warning(f"Could not read config, defaulting to lightgbm: {e}")
    return 'lightgbm'


def _find_delisted_price_files() -> Dict[str, Path]:
    if not DELISTED_DIR.exists():
        return {}
    price_files = {}
    for pattern in ["*.parquet", "*.csv"]:
        for f in DELISTED_DIR.rglob(pattern):
            price_files[f.stem] = f
    return price_files


def _find_delisted_info_files() -> Dict[str, Path]:
    if not DELISTED_DIR.exists():
        return {}
    info_files = {}
    for f in DELISTED_DIR.rglob("*.json"):
        info_files[f.stem] = f
    return info_files


def get_available_stocks():
    """Get list of active + delisted stocks with price data."""
    active_price_files = set(f.stem for f in PRICES_DIR.glob("*.parquet"))
    active_info_files = set(f.stem for f in INFO_DIR.glob("*.json"))
    active = active_price_files & active_info_files

    delisted_prices = set(_find_delisted_price_files().keys())

    return sorted(active | delisted_prices)


def load_stock_data(symbol: str):
    """Load price and info data for a stock (active or delisted)."""
    try:
        prices_df = None
        info = None

        # Load active prices
        price_file = PRICES_DIR / f"{symbol}.parquet"
        if price_file.exists():
            prices_df = pd.read_parquet(price_file)
            prices_df['date'] = pd.to_datetime(prices_df['date'])

        # Load active info
        info_file = INFO_DIR / f"{symbol}.json"
        if info_file.exists():
            with open(info_file) as f:
                raw_info = json.load(f)
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

        # Load delisted prices if active not found
        if prices_df is None:
            delisted_prices = _find_delisted_price_files()
            delisted_file = delisted_prices.get(symbol)
            if delisted_file:
                if delisted_file.suffix == ".parquet":
                    prices_df = pd.read_parquet(delisted_file)
                else:
                    prices_df = pd.read_csv(delisted_file)
                if 'date' in prices_df.columns:
                    prices_df['date'] = pd.to_datetime(prices_df['date'])
                else:
                    prices_df['date'] = pd.to_datetime(prices_df.iloc[:, 0])

        # Load delisted info if active not found
        if info is None:
            delisted_info = _find_delisted_info_files().get(symbol)
            if delisted_info:
                with open(delisted_info) as f:
                    raw_info = json.load(f)
                info = {
                    'symbol': symbol,
                    'name': raw_info.get('name', symbol),
                    'sector': raw_info.get('sector', 'Unknown'),
                    'industry': raw_info.get('industry', 'Unknown'),
                    'market_cap': raw_info.get('market_cap', 0),
                    'current_price': raw_info.get('last_known_price', 0),
                    'pe_ratio': raw_info.get('pe_ratio'),
                    'pb_ratio': raw_info.get('pb_ratio'),
                    'dividend_yield': raw_info.get('dividend_yield'),
                    'eps': raw_info.get('eps'),
                    'book_value': raw_info.get('book_value'),
                    'roe': raw_info.get('roe'),
                    'debt_to_equity': raw_info.get('debt_to_equity'),
                    'fifty_two_week_high': raw_info.get('peak_price'),
                    'fifty_two_week_low': raw_info.get('low_price'),
                }

        if prices_df is None:
            return None, None

        if info is None:
            info = {'symbol': symbol}
        return prices_df, info
    except Exception as e:
        return None, None


def calculate_forward_sharpe(prices: np.ndarray, forward_days: int = 90, 
                             risk_free: float = 0.06) -> float:
    """
    Calculate forward-looking Sharpe ratio as ML target.
    
    This shifts the ML target from "price prediction" to 
    "risk-adjusted return prediction" which is more meaningful.
    """
    if len(prices) < forward_days + 1:
        return 0.0
    
    forward_returns = np.diff(prices[:forward_days + 1]) / prices[:forward_days]
    
    if len(forward_returns) < 20:
        return 0.0
    
    annualized_return = np.mean(forward_returns) * 252
    annualized_vol = np.std(forward_returns) * np.sqrt(252)
    
    if annualized_vol == 0 or np.isnan(annualized_vol):
        return 0.0
    
    sharpe = (annualized_return - risk_free) / annualized_vol
    return float(sharpe)


def calculate_forward_sortino(prices: np.ndarray, forward_days: int = 90,
                              risk_free: float = 0.06) -> float:
    """
    Calculate forward-looking Sortino ratio.
    Sortino ratio only penalizes downside volatility.
    """
    if len(prices) < forward_days + 1:
        return 0.0
    
    forward_returns = np.diff(prices[:forward_days + 1]) / prices[:forward_days]
    
    if len(forward_returns) < 20:
        return 0.0
    
    annualized_return = np.mean(forward_returns) * 252
    
    downside_returns = forward_returns[forward_returns < 0]
    if len(downside_returns) < 5:
        return annualized_return
    
    downside_vol = np.std(downside_returns) * np.sqrt(252)
    
    if downside_vol == 0 or np.isnan(downside_vol):
        return annualized_return
    
    sortino = (annualized_return - risk_free) / downside_vol
    return float(sortino)


def generate_risk_adjusted_label(sharpe: float, sortino: float = None) -> str:
    """
    Convert Sharpe/Sortino ratio to signal labels.
    
    This creates labels based on risk-adjusted returns rather than
    just price movement or arbitrary rules.
    """
    effective_ratio = sharpe
    if sortino is not None:
        effective_ratio = (sharpe + sortino) / 2
    
    if effective_ratio >= 1.5:
        return 'BUY'
    elif effective_ratio >= 0.5:
        return 'HOLD'
    elif effective_ratio >= -0.5:
        return 'AVOID'
    else:
        return 'SELL'


@dataclass
class ProxyFinancialResult:
    revenue_cagr_3y: float = 0.0
    revenue_cagr_5y: float = 0.0
    pat_cagr_3y: float = 0.0
    pat_cagr_5y: float = 0.0
    roce_current: float = 0.0
    fcf_yield: float = 0.0
    debt_to_equity: float = 0.0
    earnings_quality: float = 0.0
    overall_score: float = 50.0
    details: dict = None


@dataclass
class ProxyValuationResult:
    current_pe: float = 0.0
    pe_percentile_own: float = 50.0
    current_pb: float = 0.0
    pb_percentile_own: float = 50.0
    ev_ebitda: float = 0.0
    peg_ratio: float = 0.0
    overall_score: float = 50.0


def _normalize_price_df(prices_df: pd.DataFrame) -> pd.DataFrame:
    df = prices_df.copy()
    if 'close' not in df.columns:
        if 'Close' in df.columns:
            df['close'] = df['Close']
        else:
            df['close'] = df.iloc[:, 3]
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date')


def _filter_macro_data(macro_data: Optional[Dict[str, Any]], cutoff_date: pd.Timestamp) -> Optional[Dict[str, Any]]:
    if macro_data is None:
        return None
    filtered = {}
    for key, value in macro_data.items():
        if isinstance(value, pd.DataFrame):
            df = value.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[df['date'] <= cutoff_date].set_index('date')
            else:
                df.index = pd.to_datetime(df.index)
                df = df[df.index <= cutoff_date]
            filtered[key] = df
        elif isinstance(value, pd.Series):
            series = value.copy()
            series.index = pd.to_datetime(series.index)
            filtered[key] = series[series.index <= cutoff_date]
        else:
            filtered[key] = value
    return filtered


def generate_time_series_samples(
    symbol: str,
    prices_df: pd.DataFrame,
    info: dict,
    feature_engineer: FeatureEngineer,
    macro_data: Optional[Dict[str, Any]] = None,
    forward_days: int = 90,
    min_history: int = 252,
    stride: int = 21
) -> Tuple[List[Dict[str, float]], List[str], List[pd.Timestamp]]:
    if prices_df is None or prices_df.empty:
        return [], [], []
    info = info or {}

    df = _normalize_price_df(prices_df)
    prices = df['close'].values
    dates = df['date'].values

    if len(prices) < min_history + forward_days + 1:
        return [], [], []

    samples, labels, sample_dates = [], [], []

    proxy_fin = ProxyFinancialResult(
        roce_current=float(info.get('roe') or 0),
        debt_to_equity=float(info.get('debt_to_equity') or 0),
        details={}
    )
    proxy_val = ProxyValuationResult(
        current_pe=float(info.get('pe_ratio') or 0),
        current_pb=float(info.get('pb_ratio') or 0),
    )

    for idx in range(min_history, len(prices) - forward_days, stride):
        cutoff_date = pd.to_datetime(dates[idx])
        price_slice = df.iloc[:idx + 1].set_index('date')['close']
        forward_prices = prices[idx:idx + forward_days + 1]

        if len(forward_prices) < forward_days + 1:
            continue

        sharpe = calculate_forward_sharpe(forward_prices, forward_days=forward_days)
        sortino = calculate_forward_sortino(forward_prices, forward_days=forward_days)
        label = generate_risk_adjusted_label(sharpe, sortino)

        macro_for_cutoff = _filter_macro_data(macro_data, cutoff_date)
        feature_set = feature_engineer.extract_features(
            symbol=symbol,
            prices=price_slice,
            financial_result=proxy_fin,
            valuation_result=proxy_val,
            macro_data=macro_for_cutoff,
        )

        samples.append(feature_set.features)
        labels.append(label)
        sample_dates.append(cutoff_date)

    return samples, labels, sample_dates


def main():
    # Get classifier type from config
    classifier_type = get_classifier_type_from_config()
    classifier_name = classifier_type.upper()
    
    print("\n" + "=" * 60)
    print(f"  {classifier_name} Classifier Training (Local Data)")
    print("=" * 60)
    
    # Check data directory
    if not PRICES_DIR.exists() or not INFO_DIR.exists():
        print("\n❌ Error: Data directory not found!")
        print("   Make sure you have stock data in the 'data' directory.")
        return False
    
    # Get available stocks
    stocks = get_available_stocks()
    print(f"\n📊 Found {len(stocks)} stocks with local data")
    print(f"🔧 Using classifier: {classifier_name}")
    
    if len(stocks) < 50:
        print("❌ Not enough stocks for training (need at least 50)")
        return False
    
    max_stocks = 200
    np.random.seed(42)
    if len(stocks) > max_stocks:
        training_stocks = sorted(list(np.random.choice(stocks, size=max_stocks, replace=False)))
    else:
        training_stocks = stocks
    print(f"📈 Training on {len(training_stocks)} random stocks...")
    
    # Collect training data
    feature_engineer = FeatureEngineer(include_technical=True, include_forecast=False, include_macro=True)
    delisted_provider = DelistedStockProvider()
    macro_provider = get_macro_provider()
    macro_data = None
    try:
        macro_data = macro_provider.get_all_macro_data(years=30)
    except Exception as e:
        logger.info(f"Macro data unavailable, continuing without: {e}")
    
    training_data = []
    labels = []
    sample_dates = []
    sample_weights = []
    successful = 0
    
    for i, symbol in enumerate(training_stocks):
        if (i + 1) % 50 == 0:
            print(f"   Processing... {i + 1}/{len(training_stocks)}")
        
        prices_df, info = load_stock_data(symbol)
        if prices_df is None:
            continue
        
        samples, sample_labels, dates = generate_time_series_samples(
            symbol=symbol,
            prices_df=prices_df,
            info=info,
            feature_engineer=feature_engineer,
            macro_data=macro_data,
            forward_days=90,
            min_history=252,
            stride=63
        )
        
        if samples:
            weight = delisted_provider.get_training_weight(symbol)
            training_data.extend(samples)
            labels.extend(sample_labels)
            sample_dates.extend(dates)
            sample_weights.extend([weight] * len(samples))
            successful += 1
    
    print(f"\n✓ Successfully processed {successful}/{len(training_stocks)} stocks")
    
    if len(training_data) < 200:
        print("❌ Not enough valid training samples")
        return False
    
    # Create DataFrame
    X = pd.DataFrame(training_data)
    y = pd.Series(labels)
    sample_dates = pd.to_datetime(pd.Series(sample_dates))
    sample_weights = pd.Series(sample_weights)
    
    # Fill NaN
    X = X.fillna(0)
    
    print(f"\n📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Label distribution:")
    for label, count in y.value_counts().items():
        print(f"     {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train classifier
    print("\n🔧 Training classifier...")
    
    from sklearn.metrics import accuracy_score, f1_score
    
    # Encode labels
    label_map = {'BUY': 0, 'HOLD': 1, 'AVOID': 2, 'SELL': 3}
    y_encoded = y.map(label_map)
    
    # Walk-Forward Validation: Use time-based split to avoid look-ahead bias
    print("   Using Walk-Forward Validation (time-based split)")
    
    order = sample_dates.sort_values().index
    X = X.iloc[order].reset_index(drop=True)
    y_encoded = y_encoded.iloc[order].reset_index(drop=True)
    sample_weights = sample_weights.iloc[order].reset_index(drop=True)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_encoded.iloc[:split_idx], y_encoded.iloc[split_idx:]
    w_train, w_test = sample_weights.iloc[:split_idx], sample_weights.iloc[split_idx:]
    
    print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    model = None
    model_type = classifier_type
    
    # Train based on configured classifier type
    if classifier_type == "catboost":
        try:
            from catboost import CatBoostClassifier
            print("   Using CatBoost...")
            
            model = CatBoostClassifier(
                iterations=200,
                learning_rate=0.05,
                depth=6,
                loss_function='MultiClass',
                classes_count=4,
                auto_class_weights='Balanced',
                random_seed=42,
                verbose=0,
            )
            model.fit(X_train, y_train, sample_weight=w_train)
            
        except ImportError:
            print("   CatBoost not installed, falling back to LightGBM...")
            classifier_type = "lightgbm"
        except Exception as e:
            print(f"   CatBoost failed: {str(e)[:50]}...")
            print("   Falling back to LightGBM...")
            classifier_type = "lightgbm"
    
    if classifier_type == "lightgbm" and model is None:
        try:
            import lightgbm as lgb
            print("   Using LightGBM...")
            model_type = "lightgbm"
            
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
            model.fit(X_train, y_train, sample_weight=w_train)
            
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
                model.fit(X_train, y_train, sample_weight=w_train)
                
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
    
    # Retrain on full dataset for final model
    try:
        model.fit(X, y_encoded, sample_weight=sample_weights)
    except Exception:
        pass
    
    # Save model to the correct directory based on classifier type
    model_dir = MODELS_DIR / "classifier" / model_type
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
    print(f"\n🎉 {model_type.upper()} classifier training complete! You can now run the app.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
