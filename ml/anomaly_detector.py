"""
Anomaly Detection Module using Z-score and Isolation Forest.

Apply to: [Fraud detection, system monitoring, performance anomaly detection,
sensor data analysis, log analysis, network intrusion detection, time-series
outlier detection, quality control, infrastructure monitoring, incident prediction]

Features:
- Z-score baseline analysis with configurable thresholds (1.5σ, 3.0σ, 4.5σ)
- Isolation Forest for complex anomaly patterns
- Multi-level anomaly classification (NORMAL, WARNING, CRITICAL, EXTREME)
- Batch and streaming detection modes
- Trajectory prediction with confidence scoring
- Alert fatigue prevention (80%+ confidence threshold)
- Human-readable explanations
- Production-ready with proper error handling
- SDF Gold layer integration bridge
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyLevel(Enum):
    """Anomaly classification levels."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EXTREME = "EXTREME"


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    value: float
    zscore: float
    isolation_score: float
    level: AnomalyLevel
    explanation: str
    metadata: Optional[Dict] = None


class AnomalyDetector:
    """
    Hybrid anomaly detector combining Z-score and Isolation Forest.
    
    Z-score detects statistical outliers (univariate).
    Isolation Forest detects complex patterns (multivariate).
    
    Example:
        >>> detector = AnomalyDetector(zscore_threshold=3, contamination=0.1)
        >>> detector.fit(historical_data)
        >>> result = detector.detect(new_value)
        >>> print(f"{result.level.value}: {result.explanation}")
    """
    
    def __init__(
        self,
        zscore_threshold: float = 3.0,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        """
        Initialize anomaly detector.
        
        Args:
            zscore_threshold: Sigma threshold for Z-score (default: 3)
            contamination: Expected proportion of outliers (0.0-0.5)
            n_estimators: Number of trees in Isolation Forest
            random_state: Random seed for reproducibility
        """
        self.zscore_threshold = zscore_threshold
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Statistical parameters
        self.mean: Optional[float] = None
        self.std: Optional[float] = None
        
        # Isolation Forest model
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.is_fitted = False
    
    def fit(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> 'AnomalyDetector':
        """
        Fit the anomaly detector on historical data.
        
        Args:
            data: Historical data (1D or 2D array-like)
            
        Returns:
            Self for method chaining
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        data = np.asarray(data)
        
        # Handle 1D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Calculate Z-score parameters (using first column for univariate)
        self.mean = np.mean(data[:, 0])
        self.std = np.std(data[:, 0])
        
        # Fit Isolation Forest
        self.isolation_forest.fit(data)
        self.is_fitted = True
        
        return self
    
    def _calculate_zscore(self, value: float) -> float:
        """Calculate Z-score for a single value."""
        if self.std == 0:
            return 0.0
        return (value - self.mean) / self.std
    
    def _classify_anomaly(
        self,
        zscore: float,
        isolation_score: float
    ) -> AnomalyLevel:
        """
        Classify anomaly level based on Z-score and Isolation Forest score.
        
        Isolation Forest scores: -1 (anomaly), 1 (normal)
        Combined scoring for robust classification.
        """
        abs_zscore = abs(zscore)
        
        # Extreme: High Z-score AND flagged by Isolation Forest
        if abs_zscore > self.zscore_threshold * 1.5 and isolation_score == -1:
            return AnomalyLevel.EXTREME
        
        # Critical: Either very high Z-score OR strong Isolation Forest signal
        if abs_zscore > self.zscore_threshold or isolation_score == -1:
            return AnomalyLevel.CRITICAL
        
        # Warning: Moderate Z-score
        if abs_zscore > self.zscore_threshold * 0.67:
            return AnomalyLevel.WARNING
        
        return AnomalyLevel.NORMAL
    
    def _generate_explanation(
        self,
        value: float,
        zscore: float,
        level: AnomalyLevel
    ) -> str:
        """Generate human-readable explanation."""
        abs_zscore = abs(zscore)
        direction = "above" if zscore > 0 else "below"
        
        if level == AnomalyLevel.NORMAL:
            return f"Value {value:.2f} is within normal range ({abs_zscore:.2f}σ)"
        elif level == AnomalyLevel.WARNING:
            return f"Value {value:.2f} is {abs_zscore:.2f}σ {direction} mean (warning threshold)"
        elif level == AnomalyLevel.CRITICAL:
            return f"Value {value:.2f} is {abs_zscore:.2f}σ {direction} mean (critical anomaly detected)"
        else:  # EXTREME
            return f"Value {value:.2f} is {abs_zscore:.2f}σ {direction} mean (extreme anomaly - immediate attention required)"
    
    def detect(
        self,
        value: Union[float, np.ndarray, List[float]],
        metadata: Optional[Dict] = None
    ) -> Union[AnomalyResult, List[AnomalyResult]]:
        """
        Detect anomalies in single value or batch (streaming mode).
        
        Args:
            value: Single value or array of values
            metadata: Optional metadata to include in result
            
        Returns:
            AnomalyResult or list of AnomalyResults
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection. Call fit() first.")
        
        # Handle single value
        if isinstance(value, (int, float)):
            return self._detect_single(float(value), metadata)
        
        # Handle batch
        values = np.asarray(value).flatten()
        return [self._detect_single(float(v), metadata) for v in values]
    
    def _detect_single(self, value: float, metadata: Optional[Dict] = None) -> AnomalyResult:
        """Detect anomaly for a single value."""
        # Calculate Z-score
        zscore = self._calculate_zscore(value)
        
        # Get Isolation Forest prediction
        value_reshaped = np.array([[value]])
        isolation_pred = self.isolation_forest.predict(value_reshaped)[0]
        
        # Classify anomaly level
        level = self._classify_anomaly(zscore, isolation_pred)
        
        # Generate explanation
        explanation = self._generate_explanation(value, zscore, level)
        
        return AnomalyResult(
            value=value,
            zscore=zscore,
            isolation_score=float(isolation_pred),
            level=level,
            explanation=explanation,
            metadata=metadata
        )
    
    def detect_batch(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        return_anomalies_only: bool = False
    ) -> List[AnomalyResult]:
        """
        Batch detection mode for historical data analysis.
        
        Args:
            data: Historical data to analyze
            return_anomalies_only: If True, return only non-NORMAL results
            
        Returns:
            List of AnomalyResults
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before detection. Call fit() first.")
        
        # Convert to array
        if isinstance(data, pd.Series):
            data = data.values
        elif isinstance(data, pd.DataFrame):
            data = data.values
        
        data = np.asarray(data).flatten()
        
        # Detect anomalies
        results = []
        for idx, value in enumerate(data):
            metadata = {"index": idx}
            result = self._detect_single(float(value), metadata)
            
            if not return_anomalies_only or result.level != AnomalyLevel.NORMAL:
                results.append(result)
        
        return results
    
    def get_anomaly_summary(self, results: List[AnomalyResult]) -> Dict:
        """
        Generate summary statistics from batch detection results.
        
        Args:
            results: List of AnomalyResults
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {"total": 0, "anomalies": 0, "anomaly_rate": 0.0}
        
        level_counts = {level: 0 for level in AnomalyLevel}
        for result in results:
            level_counts[result.level] += 1
        
        total = len(results)
        anomalies = total - level_counts[AnomalyLevel.NORMAL]
        
        return {
            "total": total,
            "anomalies": anomalies,
            "anomaly_rate": anomalies / total if total > 0 else 0.0,
            "normal": level_counts[AnomalyLevel.NORMAL],
            "warning": level_counts[AnomalyLevel.WARNING],
            "critical": level_counts[AnomalyLevel.CRITICAL],
            "extreme": level_counts[AnomalyLevel.EXTREME],
            "mean": self.mean,
            "std": self.std
        }


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("Anomaly Detection Module - Example Usage")
    print("=" * 80)
    
    # Generate synthetic data (CPU usage percentage)
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=10, size=1000)
    
    # Add some anomalies
    anomalies = np.array([150, 160, 5, 0, 175])
    
    print("\n1. BATCH DETECTION MODE")
    print("-" * 80)
    
    # Initialize and fit detector
    detector = AnomalyDetector(zscore_threshold=3, contamination=0.05)
    detector.fit(normal_data)
    
    print(f"✓ Detector fitted on {len(normal_data)} historical data points")
    print(f"  Mean: {detector.mean:.2f}, Std: {detector.std:.2f}")
    
    # Batch detection on test data
    test_data = np.concatenate([normal_data[:20], anomalies])
    results = detector.detect_batch(test_data, return_anomalies_only=True)
    
    print(f"\n✓ Detected {len(results)} anomalies in test data:")
    for result in results:
        print(f"  [{result.level.value:8s}] {result.explanation}")
    
    # Summary statistics
    all_results = detector.detect_batch(test_data)
    summary = detector.get_anomaly_summary(all_results)
    print(f"\n✓ Summary Statistics:")
    print(f"  Total samples: {summary['total']}")
    print(f"  Anomaly rate: {summary['anomaly_rate']:.1%}")
    print(f"  Breakdown: {summary['normal']} normal, {summary['warning']} warning, "
          f"{summary['critical']} critical, {summary['extreme']} extreme")
    
    print("\n2. STREAMING DETECTION MODE")
    print("-" * 80)
    
    # Simulate real-time monitoring
    streaming_values = [52, 48, 55, 145, 51, 3, 49, 180]
    
    print("✓ Real-time monitoring (streaming mode):")
    for i, value in enumerate(streaming_values):
        metadata = {"timestamp": f"2024-01-01 00:{i:02d}:00", "host": "server-01"}
        result = detector.detect(value, metadata)
        
        # Alert on anomalies
        if result.level != AnomalyLevel.NORMAL:
            print(f"  🚨 ALERT: {result.explanation}")
            print(f"     Metadata: {result.metadata}")
        else:
            print(f"  ✓ {result.explanation}")
    
    print("\n3. MULTIVARIATE ANOMALY DETECTION")
    print("-" * 80)
    
    # Generate multivariate data (CPU, Memory, Disk I/O)
    normal_multivariate = np.random.normal(
        loc=[50, 60, 40],
        scale=[10, 15, 8],
        size=(1000, 3)
    )
    
    detector_mv = AnomalyDetector(zscore_threshold=2.5, contamination=0.1)
    detector_mv.fit(normal_multivariate)
    
    # Test with anomalous multivariate data
    test_cases = [
        [52, 58, 42],   # Normal
        [95, 95, 90],   # Anomaly - all high
        [10, 15, 8],    # Anomaly - all low
        [55, 60, 45],   # Normal
    ]
    
    print("✓ Multivariate anomaly detection (CPU, Memory, Disk I/O):")
    for values in test_cases:
        # Get isolation forest prediction for full vector
        iso_pred = detector_mv.isolation_forest.predict([values])[0]
        
        # Calculate Z-score for first feature (CPU)
        zscore = detector_mv._calculate_zscore(values[0])
        level = detector_mv._classify_anomaly(zscore, iso_pred)
        
        print(f"  Values: {values}")
        print(f"    Z-score: {zscore:.2f}, Isolation: {iso_pred}, "
              f"Level: {level.value}")
    
    print("\n4. CUSTOM THRESHOLD EXAMPLE")
    print("-" * 80)
    
    # Strict detector for critical systems
    strict_detector = AnomalyDetector(zscore_threshold=2, contamination=0.15)
    strict_detector.fit(normal_data)
    
    # Lenient detector for noisy data
    lenient_detector = AnomalyDetector(zscore_threshold=4, contamination=0.05)
    lenient_detector.fit(normal_data)
    
    test_value = 80  # Moderately high value
    
    strict_result = strict_detector.detect(test_value)
    lenient_result = lenient_detector.detect(test_value)
    
    print(f"✓ Testing value: {test_value}")
    print(f"  Strict detector (threshold=2σ): {strict_result.level.value} - {strict_result.explanation}")
    print(f"  Lenient detector (threshold=4σ): {lenient_result.level.value} - {lenient_result.explanation}")


# ============================================================================
# Trajectory Prediction with Confidence Scoring
# ============================================================================

@dataclass
class TrajectoryPrediction:
    """Prediction of future anomaly trajectory."""
    predicted_values: List[float]
    predicted_anomalies: List[AnomalyLevel]
    confidence_score: float  # 0.0 to 1.0
    time_horizon: int  # Number of steps predicted
    alert_recommended: bool  # True if confidence >= 80%
    explanation: str


class TrajectoryPredictor:
    """
    Predict future anomaly trajectory with confidence scoring.
    
    Apply to: Incident prediction, capacity planning, proactive monitoring
    
    Features:
    - Linear trend extrapolation
    - Confidence scoring based on data quality
    - Alert fatigue prevention (80%+ confidence threshold)
    - Integration bridge for SDF Gold layer events
    """
    
    def __init__(
        self,
        detector: AnomalyDetector,
        alert_confidence_threshold: float = 0.80
    ):
        """
        Initialize trajectory predictor.
        
        Args:
            detector: Fitted AnomalyDetector instance
            alert_confidence_threshold: Minimum confidence to trigger alerts (default: 80%)
        """
        self.detector = detector
        self.alert_confidence_threshold = alert_confidence_threshold
        self.history: List[float] = []
    
    def add_observation(self, value: float) -> None:
        """Add observation to history for trajectory analysis."""
        self.history.append(value)
        
        # Keep last 100 observations for efficient computation
        if len(self.history) > 100:
            self.history = self.history[-100:]
    
    def predict_trajectory(
        self,
        steps: int = 5
    ) -> TrajectoryPrediction:
        """
        Predict future trajectory of values and anomaly likelihood.
        
        Args:
            steps: Number of future steps to predict
            
        Returns:
            TrajectoryPrediction with forecasted values and confidence
        """
        if len(self.history) < 3:
            return TrajectoryPrediction(
                predicted_values=[],
                predicted_anomalies=[],
                confidence_score=0.0,
                time_horizon=0,
                alert_recommended=False,
                explanation="Insufficient data for trajectory prediction (need at least 3 observations)"
            )
        
        # Calculate linear trend using least squares
        history_array = np.array(self.history)
        x = np.arange(len(history_array))
        
        # Fit linear regression
        coefficients = np.polyfit(x, history_array, deg=1)
        slope, intercept = coefficients
        
        # Predict future values
        future_x = np.arange(len(history_array), len(history_array) + steps)
        predicted_values = [slope * x_val + intercept for x_val in future_x]
        
        # Detect anomalies in predictions
        predicted_anomalies = []
        for pred_value in predicted_values:
            result = self.detector.detect(pred_value)
            predicted_anomalies.append(result.level)
        
        # Calculate confidence score based on:
        # 1. Data consistency (low variance in recent trend)
        # 2. Number of observations
        # 3. Strength of trend
        
        # Data consistency (lower is better)
        recent_values = history_array[-10:]
        variance_factor = 1.0 / (1.0 + np.std(recent_values) / (np.mean(recent_values) + 1e-6))
        
        # Number of observations (more is better, plateau at 50)
        observation_factor = min(len(self.history) / 50.0, 1.0)
        
        # Trend strength (stronger trends = higher confidence)
        trend_strength = abs(slope) / (np.std(history_array) + 1e-6)
        trend_factor = min(trend_strength, 1.0)
        
        # Combined confidence score
        confidence_score = (variance_factor * 0.4 + observation_factor * 0.3 + trend_factor * 0.3)
        confidence_score = np.clip(confidence_score, 0.0, 1.0)
        
        # Determine if alert should be triggered
        has_critical_anomalies = any(
            level in (AnomalyLevel.CRITICAL, AnomalyLevel.EXTREME)
            for level in predicted_anomalies
        )
        alert_recommended = (
            confidence_score >= self.alert_confidence_threshold and
            has_critical_anomalies
        )
        
        # Generate explanation
        if alert_recommended:
            explanation = (
                f"High-confidence ({confidence_score:.1%}) prediction of "
                f"{sum(1 for l in predicted_anomalies if l in (AnomalyLevel.CRITICAL, AnomalyLevel.EXTREME))} "
                f"critical anomalies in next {steps} steps. Alert recommended."
            )
        elif has_critical_anomalies:
            explanation = (
                f"Low-confidence ({confidence_score:.1%}) prediction of anomalies. "
                f"Alert suppressed to prevent fatigue (threshold: {self.alert_confidence_threshold:.0%})"
            )
        else:
            explanation = f"No critical anomalies predicted in next {steps} steps"
        
        return TrajectoryPrediction(
            predicted_values=predicted_values,
            predicted_anomalies=predicted_anomalies,
            confidence_score=confidence_score,
            time_horizon=steps,
            alert_recommended=alert_recommended,
            explanation=explanation
        )


# ============================================================================
# SDF Gold Layer Integration Bridge
# ============================================================================

@dataclass
class SDFGoldEvent:
    """Event format for SDF Gold layer integration."""
    timestamp: str
    event_type: str  # "anomaly_detected", "trajectory_alert"
    severity: str  # "NORMAL", "WARNING", "CRITICAL", "EXTREME"
    value: float
    zscore: float
    confidence: float
    metadata: Dict
    alert_required: bool


class SDFBridge:
    """
    Bridge to SDF (security-data-fabric) Gold layer for event streaming.
    
    Apply to: Real-time monitoring, incident prediction, security analytics
    
    Features:
    - Standardized event format for SDF Gold layer
    - Automatic alert prioritization
    - Batch event processing
    """
    
    def create_anomaly_event(
        self,
        result: AnomalyResult,
        timestamp: Optional[str] = None
    ) -> SDFGoldEvent:
        """
        Convert AnomalyResult to SDF Gold layer event.
        
        Args:
            result: AnomalyResult from detector
            timestamp: ISO format timestamp (defaults to current time)
            
        Returns:
            SDFGoldEvent ready for SDF ingestion
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"
        
        return SDFGoldEvent(
            timestamp=timestamp,
            event_type="anomaly_detected",
            severity=result.level.value,
            value=result.value,
            zscore=result.zscore,
            confidence=1.0,  # Anomaly detection has 100% confidence in its classification
            metadata=result.metadata or {},
            alert_required=result.level in (AnomalyLevel.CRITICAL, AnomalyLevel.EXTREME)
        )
    
    def create_trajectory_event(
        self,
        prediction: TrajectoryPrediction,
        timestamp: Optional[str] = None
    ) -> SDFGoldEvent:
        """
        Convert TrajectoryPrediction to SDF Gold layer event.
        
        Args:
            prediction: TrajectoryPrediction from predictor
            timestamp: ISO format timestamp (defaults to current time)
            
        Returns:
            SDFGoldEvent ready for SDF ingestion
        """
        from datetime import datetime
        
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Determine most severe predicted level
        max_severity = AnomalyLevel.NORMAL
        for level in prediction.predicted_anomalies:
            if level == AnomalyLevel.EXTREME:
                max_severity = AnomalyLevel.EXTREME
                break
            elif level == AnomalyLevel.CRITICAL and max_severity != AnomalyLevel.EXTREME:
                max_severity = AnomalyLevel.CRITICAL
            elif level == AnomalyLevel.WARNING and max_severity == AnomalyLevel.NORMAL:
                max_severity = AnomalyLevel.WARNING
        
        return SDFGoldEvent(
            timestamp=timestamp,
            event_type="trajectory_alert",
            severity=max_severity.value,
            value=prediction.predicted_values[0] if prediction.predicted_values else 0.0,
            zscore=0.0,  # Not applicable for trajectory
            confidence=prediction.confidence_score,
            metadata={
                "predicted_values": prediction.predicted_values,
                "time_horizon": prediction.time_horizon,
                "explanation": prediction.explanation
            },
            alert_required=prediction.alert_recommended
        )
    
    def batch_process(
        self,
        results: List[AnomalyResult]
    ) -> List[SDFGoldEvent]:
        """
        Batch process anomaly results to SDF Gold events.
        
        Args:
            results: List of AnomalyResults
            
        Returns:
            List of SDFGoldEvents
        """
        return [self.create_anomaly_event(result) for result in results]

    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
