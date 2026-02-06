"""
Time-Series Forecasting Module using Random Forest.

Apply to: [Capacity planning, resource forecasting, demand prediction, 
infrastructure scaling, cost estimation, traffic prediction, load forecasting,
disk usage prediction, memory consumption trends, database growth projection,
API rate limit planning, autoscaling decisions, budget planning]

Features:
- Advanced feature engineering (lag features, rolling statistics)
- Random Forest for non-linear time-series patterns
- Train/predict pipeline with data validation
- Confidence intervals using prediction variance
- Model persistence (save/load)
- Capacity planning: "When will metric X hit threshold Y?"
- Production-ready error handling
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class ForecastResult:
    """Result of time-series forecast."""
    timestamp: datetime
    predicted_value: float
    lower_bound: float
    upper_bound: float
    confidence: float
    metadata: Optional[Dict] = None


@dataclass
class CapacityPlanResult:
    """Result of capacity planning analysis."""
    current_value: float
    threshold: float
    predicted_date: Optional[datetime]
    days_until_threshold: Optional[int]
    growth_rate: float
    forecast_values: List[ForecastResult]
    warning_level: str


class TimeSeriesForecaster:
    """
    Random Forest-based time-series forecaster with feature engineering.
    
    Uses lag features and rolling statistics to capture temporal patterns.
    Suitable for non-linear, non-stationary time series.
    
    Example:
        >>> forecaster = TimeSeriesForecaster()
        >>> forecaster.train(df, target_column='cpu_usage', date_column='timestamp')
        >>> forecast = forecaster.predict(periods=30)
        >>> capacity = forecaster.capacity_planning(threshold=90, periods=90)
    """
    
    def __init__(
        self,
        lag_features: List[int] = None,
        rolling_windows: List[int] = None,
        n_estimators: int = 100,
        max_depth: int = 20,
        random_state: int = 42,
        confidence_level: float = 0.95
    ):
        """
        Initialize time-series forecaster.
        
        Args:
            lag_features: List of lag periods (default: [1, 2, 3, 7])
            rolling_windows: List of rolling window sizes (default: [7, 30])
            n_estimators: Number of trees in Random Forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
            confidence_level: Confidence level for intervals (0.0-1.0)
        """
        self.lag_features = lag_features or [1, 2, 3, 7]
        self.rolling_windows = rolling_windows or [7, 30]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.confidence_level = confidence_level
        
        # Model
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Training metadata
        self.is_trained = False
        self.feature_columns: List[str] = []
        self.target_column: Optional[str] = None
        self.last_known_date: Optional[datetime] = None
        self.last_known_values: Optional[pd.Series] = None
        self.training_metrics: Optional[Dict] = None
        
    def _create_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        is_training: bool = True
    ) -> pd.DataFrame:
        """
        Create lag features and rolling statistics.
        
        Args:
            df: DataFrame with date index and target column
            target_col: Name of target column
            is_training: Whether creating features for training
            
        Returns:
            DataFrame with engineered features
        """
        # Keep only target column
        features_df = df[[target_col]].copy() if target_col in df.columns else df.copy()
        
        # Lag features (previous days)
        for lag in self.lag_features:
            features_df[f'lag_{lag}'] = features_df[target_col].shift(lag)
        
        # Rolling statistics (moving averages and std)
        for window in self.rolling_windows:
            features_df[f'rolling_mean_{window}'] = (
                features_df[target_col].rolling(window=window, min_periods=1).mean()
            )
            features_df[f'rolling_std_{window}'] = (
                features_df[target_col].rolling(window=window, min_periods=1).std()
            )
            features_df[f'rolling_min_{window}'] = (
                features_df[target_col].rolling(window=window, min_periods=1).min()
            )
            features_df[f'rolling_max_{window}'] = (
                features_df[target_col].rolling(window=window, min_periods=1).max()
            )
        
        # Time-based features
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['day_of_month'] = features_df.index.day
        features_df['month'] = features_df.index.month
        features_df['day_of_year'] = features_df.index.dayofyear
        
        # Trend feature (days since start)
        features_df['days_since_start'] = (
            (features_df.index - features_df.index[0]).days
        )
        
        if is_training:
            # Store feature columns (exclude target)
            self.feature_columns = [
                col for col in features_df.columns if col != target_col
            ]
        
        return features_df
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str,
        date_column: Optional[str] = None,
        test_size: float = 0.2
    ) -> Dict:
        """
        Train the forecasting model.
        
        Args:
            df: DataFrame with time-series data
            target_column: Name of target column to forecast
            date_column: Name of date column (if not index)
            test_size: Fraction of data for testing (for metrics)
            
        Returns:
            Dictionary with training metrics
        """
        # Prepare data
        data = df.copy()
        
        # Set date as index if needed
        if date_column and date_column in data.columns:
            data[date_column] = pd.to_datetime(data[date_column])
            data = data.set_index(date_column)
        
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Sort by date
        data = data.sort_index()
        
        # Store metadata
        self.target_column = target_column
        self.last_known_date = data.index[-1]
        self.last_known_values = data[target_column].tail(max(self.lag_features) + 1)
        
        # Create features
        features_df = self._create_features(data, target_column, is_training=True)
        
        # Remove rows with NaN (from lag/rolling calculations)
        features_df = features_df.dropna()
        
        if len(features_df) < 50:
            raise ValueError(
                f"Insufficient data after feature engineering. "
                f"Need at least 50 samples, got {len(features_df)}"
            )
        
        # Split features and target
        X = features_df[self.feature_columns]
        y = features_df[target_column]
        
        # Train/test split (time-based)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        self.training_metrics = {
            "train_mae": mean_absolute_error(y_train, train_pred),
            "train_rmse": np.sqrt(mean_squared_error(y_train, train_pred)),
            "train_r2": r2_score(y_train, train_pred),
            "test_mae": mean_absolute_error(y_test, test_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, test_pred)),
            "test_r2": r2_score(y_test, test_pred),
            "n_samples": len(features_df),
            "n_features": len(self.feature_columns),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        return self.training_metrics
    
    def predict(
        self,
        periods: int = 30,
        start_date: Optional[datetime] = None
    ) -> List[ForecastResult]:
        """
        Predict future values.
        
        Args:
            periods: Number of periods to forecast
            start_date: Start date for forecast (default: day after last training date)
            
        Returns:
            List of ForecastResults
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction. Call train() first.")
        
        # Determine start date
        if start_date is None:
            start_date = self.last_known_date + timedelta(days=1)
        
        # Initialize with historical values
        forecast_data = pd.DataFrame({
            self.target_column: self.last_known_values
        }, index=pd.date_range(
            start=self.last_known_date - timedelta(days=len(self.last_known_values) - 1),
            periods=len(self.last_known_values),
            freq='D'
        ))
        
        predictions = []
        
        for i in range(periods):
            current_date = start_date + timedelta(days=i)
            
            # Add placeholder for current date
            forecast_data.loc[current_date] = np.nan
            
            # Create features for current date
            features_df = self._create_features(
                forecast_data,
                self.target_column,
                is_training=False
            )
            
            # Get features for current date
            current_features = features_df.loc[current_date, self.feature_columns]
            
            # Handle any remaining NaN values
            current_features = current_features.ffill().bfill()
            
            # Predict
            X_pred = pd.DataFrame(
                [current_features.values],
                columns=self.feature_columns
            )
            predicted_value = self.model.predict(X_pred)[0]
            
            # Calculate confidence interval using tree predictions
            tree_predictions = np.array([
                tree.predict(X_pred)[0] for tree in self.model.estimators_
            ])
            std = np.std(tree_predictions)
            
            # Z-score for confidence level
            z_score = 1.96 if self.confidence_level == 0.95 else 2.576
            margin = z_score * std
            
            lower_bound = predicted_value - margin
            upper_bound = predicted_value + margin
            
            # Update forecast data with prediction
            forecast_data.loc[current_date, self.target_column] = predicted_value
            
            # Create result
            result = ForecastResult(
                timestamp=current_date,
                predicted_value=predicted_value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                confidence=self.confidence_level,
                metadata={"std": std, "margin": margin}
            )
            
            predictions.append(result)
        
        return predictions
    
    def capacity_planning(
        self,
        threshold: float,
        periods: int = 90,
        start_date: Optional[datetime] = None,
        metric_name: str = "metric"
    ) -> CapacityPlanResult:
        """
        Capacity planning: When will metric hit threshold?
        
        Args:
            threshold: Threshold value to check against
            periods: Maximum periods to forecast
            start_date: Start date for forecast
            metric_name: Name of metric for reporting
            
        Returns:
            CapacityPlanResult with date threshold will be hit
        """
        # Get forecast
        forecast = self.predict(periods=periods, start_date=start_date)
        
        # Find current value
        current_value = self.last_known_values.iloc[-1]
        
        # Find when threshold is crossed
        predicted_date = None
        days_until_threshold = None
        
        for result in forecast:
            if result.predicted_value >= threshold:
                predicted_date = result.timestamp
                days_until_threshold = (predicted_date - self.last_known_date).days
                break
        
        # Calculate growth rate
        if len(forecast) > 0:
            final_value = forecast[-1].predicted_value
            days = len(forecast)
            growth_rate = ((final_value - current_value) / current_value) / days * 100
        else:
            growth_rate = 0.0
        
        # Determine warning level
        if predicted_date is None:
            warning_level = "NORMAL"
        elif days_until_threshold <= 7:
            warning_level = "CRITICAL"
        elif days_until_threshold <= 30:
            warning_level = "WARNING"
        else:
            warning_level = "INFO"
        
        return CapacityPlanResult(
            current_value=current_value,
            threshold=threshold,
            predicted_date=predicted_date,
            days_until_threshold=days_until_threshold,
            growth_rate=growth_rate,
            forecast_values=forecast,
            warning_level=warning_level
        )
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "last_known_date": self.last_known_date,
            "last_known_values": self.last_known_values,
            "training_metrics": self.training_metrics,
            "lag_features": self.lag_features,
            "rolling_windows": self.rolling_windows,
            "confidence_level": self.confidence_level
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: Union[str, Path]) -> 'TimeSeriesForecaster':
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.feature_columns = model_data["feature_columns"]
        self.target_column = model_data["target_column"]
        self.last_known_date = model_data["last_known_date"]
        self.last_known_values = model_data["last_known_values"]
        self.training_metrics = model_data["training_metrics"]
        self.lag_features = model_data["lag_features"]
        self.rolling_windows = model_data["rolling_windows"]
        self.confidence_level = model_data["confidence_level"]
        
        self.is_trained = True
        
        return self


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Time-Series Forecasting Module - Example Usage")
    print("=" * 80)
    
    # Generate synthetic time-series data
    np.random.seed(42)
    
    # 365 days of data with trend and seasonality
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    # CPU usage with weekly seasonality and upward trend
    trend = np.linspace(40, 70, 365)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 7)
    noise = np.random.normal(0, 3, 365)
    cpu_usage = trend + seasonality + noise
    cpu_usage = np.clip(cpu_usage, 0, 100)
    
    # Memory usage with monthly seasonality and upward trend
    trend_mem = np.linspace(50, 85, 365)
    seasonality_mem = 8 * np.sin(2 * np.pi * np.arange(365) / 30)
    noise_mem = np.random.normal(0, 2, 365)
    memory_usage = trend_mem + seasonality_mem + noise_mem
    memory_usage = np.clip(memory_usage, 0, 100)
    
    # Disk usage with linear growth
    disk_usage = np.linspace(30, 80, 365) + np.random.normal(0, 2, 365)
    disk_usage = np.clip(disk_usage, 0, 100)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'disk_usage': disk_usage
    })
    
    print("\n1. TRAINING MODELS")
    print("-" * 80)
    
    # Train CPU forecaster
    cpu_forecaster = TimeSeriesForecaster(
        lag_features=[1, 2, 3, 7],
        rolling_windows=[7, 30],
        n_estimators=100
    )
    
    cpu_metrics = cpu_forecaster.train(
        df,
        target_column='cpu_usage',
        date_column='date',
        test_size=0.2
    )
    
    print("✓ CPU Usage Forecaster Trained:")
    print(f"  Training RMSE: {cpu_metrics['train_rmse']:.2f}")
    print(f"  Test RMSE: {cpu_metrics['test_rmse']:.2f}")
    print(f"  Test R²: {cpu_metrics['test_r2']:.3f}")
    print(f"  Features: {cpu_metrics['n_features']}")
    
    # Train memory forecaster
    mem_forecaster = TimeSeriesForecaster()
    mem_metrics = mem_forecaster.train(df, target_column='memory_usage', date_column='date')
    
    print("\n✓ Memory Usage Forecaster Trained:")
    print(f"  Test RMSE: {mem_metrics['test_rmse']:.2f}")
    print(f"  Test R²: {mem_metrics['test_r2']:.3f}")
    
    # Train disk forecaster
    disk_forecaster = TimeSeriesForecaster()
    disk_metrics = disk_forecaster.train(df, target_column='disk_usage', date_column='date')
    
    print("\n✓ Disk Usage Forecaster Trained:")
    print(f"  Test RMSE: {disk_metrics['test_rmse']:.2f}")
    print(f"  Test R²: {disk_metrics['test_r2']:.3f}")
    
    print("\n2. FORECASTING")
    print("-" * 80)
    
    # Forecast next 30 days
    cpu_forecast = cpu_forecaster.predict(periods=30)
    
    print("✓ CPU Usage Forecast (next 30 days):")
    print(f"  {'Date':<12} {'Predicted':<12} {'95% CI':<25} {'Confidence'}")
    print(f"  {'-'*12} {'-'*12} {'-'*25} {'-'*10}")
    
    for i, result in enumerate(cpu_forecast[:5]):  # Show first 5 days
        ci_str = f"[{result.lower_bound:.1f}, {result.upper_bound:.1f}]"
        print(f"  {result.timestamp.strftime('%Y-%m-%d'):<12} "
              f"{result.predicted_value:<12.2f} {ci_str:<25} {result.confidence:.0%}")
    print(f"  ... (showing 5 of 30 days)")
    
    print("\n3. CAPACITY PLANNING")
    print("-" * 80)
    
    # CPU capacity planning (threshold: 85%)
    cpu_capacity = cpu_forecaster.capacity_planning(
        threshold=85.0,
        periods=90,
        metric_name="CPU Usage"
    )
    
    print("✓ CPU Usage Capacity Planning:")
    print(f"  Current value: {cpu_capacity.current_value:.2f}%")
    print(f"  Threshold: {cpu_capacity.threshold:.2f}%")
    print(f"  Daily growth rate: {cpu_capacity.growth_rate:.3f}%")
    
    if cpu_capacity.predicted_date:
        print(f"  🚨 Threshold will be reached on: {cpu_capacity.predicted_date.strftime('%Y-%m-%d')}")
        print(f"  Days until threshold: {cpu_capacity.days_until_threshold}")
        print(f"  Warning level: {cpu_capacity.warning_level}")
    else:
        print(f"  ✓ Threshold not expected to be reached in next 90 days")
    
    # Memory capacity planning (threshold: 90%)
    mem_capacity = mem_forecaster.capacity_planning(
        threshold=90.0,
        periods=90,
        metric_name="Memory Usage"
    )
    
    print("\n✓ Memory Usage Capacity Planning:")
    print(f"  Current value: {mem_capacity.current_value:.2f}%")
    print(f"  Threshold: {mem_capacity.threshold:.2f}%")
    print(f"  Daily growth rate: {mem_capacity.growth_rate:.3f}%")
    
    if mem_capacity.predicted_date:
        print(f"  🚨 Threshold will be reached on: {mem_capacity.predicted_date.strftime('%Y-%m-%d')}")
        print(f"  Days until threshold: {mem_capacity.days_until_threshold}")
        print(f"  Warning level: {mem_capacity.warning_level}")
    else:
        print(f"  ✓ Threshold not expected to be reached in next 90 days")
    
    # Disk capacity planning (threshold: 90%)
    disk_capacity = disk_forecaster.capacity_planning(
        threshold=90.0,
        periods=90,
        metric_name="Disk Usage"
    )
    
    print("\n✓ Disk Usage Capacity Planning:")
    print(f"  Current value: {disk_capacity.current_value:.2f}%")
    print(f"  Threshold: {disk_capacity.threshold:.2f}%")
    print(f"  Daily growth rate: {disk_capacity.growth_rate:.3f}%")
    
    if disk_capacity.predicted_date:
        print(f"  🚨 Threshold will be reached on: {disk_capacity.predicted_date.strftime('%Y-%m-%d')}")
        print(f"  Days until threshold: {disk_capacity.days_until_threshold}")
        print(f"  Warning level: {disk_capacity.warning_level}")
    else:
        print(f"  ✓ Threshold not expected to be reached in next 90 days")
    
    print("\n4. MODEL PERSISTENCE")
    print("-" * 80)
    
    # Save model
    model_path = "/tmp/cpu_forecaster.joblib"
    cpu_forecaster.save_model(model_path)
    print(f"✓ Model saved to: {model_path}")
    
    # Load model
    loaded_forecaster = TimeSeriesForecaster()
    loaded_forecaster.load_model(model_path)
    print(f"✓ Model loaded successfully")
    
    # Verify loaded model works
    loaded_forecast = loaded_forecaster.predict(periods=7)
    print(f"✓ Loaded model prediction for next 7 days: "
          f"{loaded_forecast[0].predicted_value:.2f}% → {loaded_forecast[-1].predicted_value:.2f}%")
    
    print("\n5. MULTI-RESOURCE CAPACITY SUMMARY")
    print("-" * 80)
    
    resources = [
        ("CPU", cpu_capacity),
        ("Memory", mem_capacity),
        ("Disk", disk_capacity)
    ]
    
    print("✓ Infrastructure Capacity Summary:")
    print(f"  {'Resource':<10} {'Current':<10} {'Threshold':<10} {'Days Left':<12} {'Status'}")
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
    
    for name, capacity in resources:
        current = f"{capacity.current_value:.1f}%"
        threshold = f"{capacity.threshold:.1f}%"
        days = capacity.days_until_threshold if capacity.days_until_threshold else "90+"
        status = capacity.warning_level
        
        print(f"  {name:<10} {current:<10} {threshold:<10} {str(days):<12} {status}")
    
    # Recommendations
    critical_resources = [
        name for name, cap in resources 
        if cap.warning_level in ["CRITICAL", "WARNING"]
    ]
    
    if critical_resources:
        print(f"\n  ⚠️  Immediate attention required for: {', '.join(critical_resources)}")
        print(f"  💡 Recommendation: Plan capacity increase or optimization")
    else:
        print(f"\n  ✓ All resources within acceptable capacity limits")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
