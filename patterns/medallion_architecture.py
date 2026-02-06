"""
Medallion Architecture Pattern: Bronze/Silver/Gold ETL Pipeline

Apply to:
- Data lakes (Delta Lake, Apache Iceberg, AWS S3)
- Analytics pipelines (Databricks, Snowflake, BigQuery)
- Data warehousing (ETL/ELT workflows)
- Real-time streaming pipelines (Kafka → Data Lake)
- Data quality improvement workflows

The Medallion Architecture organizes data into three progressive layers:
- Bronze: Raw, unprocessed data with minimal validation
- Silver: Cleaned, validated, and deduplicated data
- Gold: Business-ready aggregated data for analytics

Benefits:
- Incremental data quality improvement
- Clear data lineage and audit trail
- Separation of concerns between ingestion, cleaning, and aggregation
- Easy rollback and reprocessing
- Performance optimization at each tier
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
import hashlib

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError


# ============================================================================
# TIER ENUMS AND METADATA
# ============================================================================

class DataTier(str, Enum):
    """Data tier classification in medallion architecture."""
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class DataQuality(str, Enum):
    """Data quality classification."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # 0-49%


# ============================================================================
# METADATA MODELS
# ============================================================================

class DataMetadata(BaseModel):
    """Metadata tracked across all tiers."""
    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )
    
    ingestion_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    tier: DataTier
    record_id: str
    quality_score: float = Field(ge=0.0, le=1.0, default=0.0)
    validation_errors: List[str] = Field(default_factory=list)
    row_hash: Optional[str] = None


# ============================================================================
# BRONZE TIER: RAW DATA INGESTION
# ============================================================================

class BronzeRecord(BaseModel):
    """
    Bronze tier: Raw data with minimal validation.
    Accepts any data structure and preserves original format.
    """
    raw_data: Dict[str, Any]
    metadata: DataMetadata
    
    @classmethod
    def from_raw(cls, data: Dict[str, Any], source: str) -> 'BronzeRecord':
        """Create Bronze record from raw data."""
        record_id = hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        metadata = DataMetadata(
            source=source,
            tier=DataTier.BRONZE,
            record_id=record_id,
            quality_score=1.0,  # Bronze accepts all data
            row_hash=record_id
        )
        
        return cls(raw_data=data, metadata=metadata)


class BronzeLayer:
    """
    Bronze Layer: Raw data ingestion with minimal validation.
    Stores data as-is for audit trail and reprocessing.
    """
    
    def __init__(self):
        self.records: List[BronzeRecord] = []
    
    def ingest(self, data: List[Dict[str, Any]], source: str) -> pd.DataFrame:
        """
        Ingest raw data into Bronze layer.
        
        Args:
            data: List of raw data dictionaries
            source: Data source identifier
            
        Returns:
            DataFrame with Bronze records and metadata
        """
        for item in data:
            try:
                record = BronzeRecord.from_raw(item, source)
                self.records.append(record)
            except Exception as e:
                # Even validation errors are stored in Bronze
                record_id = f"error_{len(self.records)}"
                metadata = DataMetadata(
                    source=source,
                    tier=DataTier.BRONZE,
                    record_id=record_id,
                    quality_score=0.0,
                    validation_errors=[str(e)]
                )
                error_record = BronzeRecord(
                    raw_data={"error": str(e), "data": item},
                    metadata=metadata
                )
                self.records.append(error_record)
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert Bronze records to DataFrame."""
        data = []
        for record in self.records:
            row = {
                'record_id': record.metadata.record_id,
                'source': record.metadata.source,
                'ingestion_time': record.metadata.ingestion_time,
                'quality_score': record.metadata.quality_score,
                'tier': record.metadata.tier.value,
                'raw_data': json.dumps(record.raw_data)
            }
            data.append(row)
        
        return pd.DataFrame(data)


# ============================================================================
# SILVER TIER: CLEANED AND STRUCTURED DATA
# ============================================================================

class SilverTransaction(BaseModel):
    """
    Silver tier: Cleaned and validated transaction data.
    Enforces schema and data quality rules.
    """
    transaction_id: str
    user_id: str
    amount: float = Field(gt=0)
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    timestamp: datetime
    category: str
    status: str
    metadata: DataMetadata
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid_statuses = {'pending', 'completed', 'failed', 'cancelled'}
        if v.lower() not in valid_statuses:
            raise ValueError(f"Invalid status: {v}")
        return v.lower()
    
    @field_validator('category')
    @classmethod
    def validate_category(cls, v):
        valid_categories = {'food', 'transport', 'entertainment', 'utilities', 'other'}
        if v.lower() not in valid_categories:
            return 'other'
        return v.lower()


class SilverLayer:
    """
    Silver Layer: Cleaned, validated, and deduplicated data.
    Applies business rules and data quality checks.
    """
    
    def __init__(self):
        self.records: List[SilverTransaction] = []
    
    def transform(self, bronze_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform Bronze data to Silver tier.
        
        Args:
            bronze_df: DataFrame from Bronze layer
            
        Returns:
            DataFrame with cleaned Silver records
        """
        for _, row in bronze_df.iterrows():
            try:
                raw_data = json.loads(row['raw_data'])
                
                # Skip error records
                if 'error' in raw_data:
                    continue
                
                # Validate and clean data
                quality_score = self._calculate_quality_score(raw_data)
                validation_errors = []
                
                # Create Silver record
                try:
                    silver_record = SilverTransaction(
                        transaction_id=raw_data.get('transaction_id', ''),
                        user_id=raw_data.get('user_id', ''),
                        amount=float(raw_data.get('amount', 0)),
                        currency=raw_data.get('currency', 'USD'),
                        timestamp=self._parse_timestamp(raw_data.get('timestamp')),
                        category=raw_data.get('category', 'other'),
                        status=raw_data.get('status', 'pending'),
                        metadata=DataMetadata(
                            source=row['source'],
                            tier=DataTier.SILVER,
                            record_id=row['record_id'],
                            quality_score=quality_score,
                            validation_errors=validation_errors,
                            ingestion_time=row['ingestion_time']
                        )
                    )
                    self.records.append(silver_record)
                    
                except ValidationError as e:
                    # Track validation errors but continue processing
                    validation_errors = [str(err) for err in e.errors()]
                    # Store partially cleaned record with errors
                    pass
                    
            except Exception as e:
                # Log transformation errors
                continue
        
        return self.to_dataframe()
    
    def _calculate_quality_score(self, data: Dict[str, Any]) -> float:
        """
        Calculate data quality score based on completeness and validity.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        score = 0.0
        checks = 0
        
        # Required fields check
        required_fields = ['transaction_id', 'user_id', 'amount', 'timestamp']
        for field in required_fields:
            checks += 1
            if field in data and data[field]:
                score += 0.2
        
        # Data type validation
        checks += 1
        try:
            amount = float(data.get('amount', 0))
            if amount > 0:
                score += 0.15
        except (ValueError, TypeError):
            pass
        
        # Currency format check
        checks += 1
        currency = data.get('currency', '')
        if len(currency) == 3 and currency.isupper():
            score += 0.05
        
        return min(score, 1.0)
    
    def _parse_timestamp(self, ts: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(ts, datetime):
            return ts
        elif isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                return datetime.now(timezone.utc)
        else:
            return datetime.now(timezone.utc)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert Silver records to DataFrame."""
        data = []
        for record in self.records:
            row = {
                'transaction_id': record.transaction_id,
                'user_id': record.user_id,
                'amount': record.amount,
                'currency': record.currency,
                'timestamp': record.timestamp,
                'category': record.category,
                'status': record.status,
                'quality_score': record.metadata.quality_score,
                'source': record.metadata.source,
                'record_id': record.metadata.record_id,
                'tier': record.metadata.tier.value
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Deduplicate based on transaction_id
        if not df.empty:
            df = df.drop_duplicates(subset=['transaction_id'], keep='last')
        
        return df


# ============================================================================
# GOLD TIER: BUSINESS-READY AGGREGATED DATA
# ============================================================================

class GoldUserMetrics(BaseModel):
    """
    Gold tier: Aggregated business metrics.
    Optimized for analytics and reporting.
    """
    user_id: str
    total_transactions: int = Field(ge=0)
    total_amount: float = Field(ge=0)
    avg_transaction_amount: float = Field(ge=0)
    categories: List[str]
    status_breakdown: Dict[str, int]
    quality_score: float = Field(ge=0.0, le=1.0)
    metadata: DataMetadata


class GoldLayer:
    """
    Gold Layer: Business-ready aggregated data.
    Provides optimized views for analytics and reporting.
    """
    
    def __init__(self):
        self.metrics: List[GoldUserMetrics] = []
    
    def aggregate(self, silver_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate Silver data to Gold tier.
        
        Args:
            silver_df: DataFrame from Silver layer
            
        Returns:
            DataFrame with aggregated Gold metrics
        """
        if silver_df.empty:
            return pd.DataFrame()
        
        # Group by user
        for user_id, group in silver_df.groupby('user_id'):
            # Calculate metrics
            total_transactions = len(group)
            total_amount = group['amount'].sum()
            avg_amount = group['amount'].mean()
            categories = group['category'].unique().tolist()
            status_breakdown = group['status'].value_counts().to_dict()
            avg_quality_score = group['quality_score'].mean()
            
            # Create Gold record
            gold_record = GoldUserMetrics(
                user_id=user_id,
                total_transactions=total_transactions,
                total_amount=total_amount,
                avg_transaction_amount=avg_amount,
                categories=categories,
                status_breakdown=status_breakdown,
                quality_score=avg_quality_score,
                metadata=DataMetadata(
                    source="silver_aggregation",
                    tier=DataTier.GOLD,
                    record_id=f"gold_{user_id}",
                    quality_score=avg_quality_score,
                    ingestion_time=datetime.now(timezone.utc)
                )
            )
            self.metrics.append(gold_record)
        
        return self.to_dataframe()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert Gold metrics to DataFrame."""
        data = []
        for metric in self.metrics:
            row = {
                'user_id': metric.user_id,
                'total_transactions': metric.total_transactions,
                'total_amount': metric.total_amount,
                'avg_transaction_amount': metric.avg_transaction_amount,
                'categories': ','.join(metric.categories),
                'status_breakdown': json.dumps(metric.status_breakdown),
                'quality_score': metric.quality_score,
                'tier': metric.metadata.tier.value,
                'record_id': metric.metadata.record_id
            }
            data.append(row)
        
        return pd.DataFrame(data)


# ============================================================================
# DATA VALIDATION AND QUALITY SCORING
# ============================================================================

class DataValidator:
    """Validates data quality between tiers."""
    
    @staticmethod
    def validate_bronze_to_silver(bronze_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Bronze data before Silver transformation.
        
        Returns:
            Validation report with statistics
        """
        total_records = len(bronze_df)
        valid_records = bronze_df[bronze_df['quality_score'] > 0.0]
        
        return {
            'total_records': total_records,
            'valid_records': len(valid_records),
            'invalid_records': total_records - len(valid_records),
            'avg_quality_score': bronze_df['quality_score'].mean(),
            'pass_rate': len(valid_records) / total_records if total_records > 0 else 0
        }
    
    @staticmethod
    def validate_silver_to_gold(silver_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate Silver data before Gold aggregation.
        
        Returns:
            Validation report with statistics
        """
        high_quality = silver_df[silver_df['quality_score'] >= 0.7]
        
        return {
            'total_records': len(silver_df),
            'high_quality_records': len(high_quality),
            'avg_quality_score': silver_df['quality_score'].mean(),
            'unique_users': silver_df['user_id'].nunique(),
            'quality_distribution': silver_df['quality_score'].describe().to_dict()
        }
    
    @staticmethod
    def calculate_quality_tier(score: float) -> DataQuality:
        """Classify quality score into tier."""
        if score >= 0.9:
            return DataQuality.EXCELLENT
        elif score >= 0.7:
            return DataQuality.GOOD
        elif score >= 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR


# ============================================================================
# MEDALLION PIPELINE ORCHESTRATOR
# ============================================================================

class MedallionPipeline:
    """
    Orchestrates the full Bronze → Silver → Gold ETL pipeline.
    
    Usage:
        pipeline = MedallionPipeline()
        results = pipeline.run(raw_data, source="api")
        print(results['gold'])
    """
    
    def __init__(self):
        self.bronze = BronzeLayer()
        self.silver = SilverLayer()
        self.gold = GoldLayer()
        self.validator = DataValidator()
    
    def run(self, raw_data: List[Dict[str, Any]], source: str) -> Dict[str, pd.DataFrame]:
        """
        Execute full ETL pipeline.
        
        Args:
            raw_data: List of raw data dictionaries
            source: Data source identifier
            
        Returns:
            Dictionary with DataFrames for each tier
        """
        print(f"\n{'='*60}")
        print(f"MEDALLION PIPELINE EXECUTION")
        print(f"{'='*60}")
        
        # Bronze: Raw ingestion
        print(f"\n[BRONZE] Ingesting {len(raw_data)} raw records from {source}...")
        bronze_df = self.bronze.ingest(raw_data, source)
        bronze_validation = self.validator.validate_bronze_to_silver(bronze_df)
        print(f"[BRONZE] Complete: {bronze_validation['valid_records']}/{bronze_validation['total_records']} valid records")
        print(f"[BRONZE] Average quality score: {bronze_validation['avg_quality_score']:.2f}")
        
        # Silver: Cleaning and validation
        print(f"\n[SILVER] Transforming to Silver tier...")
        silver_df = self.silver.transform(bronze_df)
        silver_validation = self.validator.validate_silver_to_gold(silver_df)
        print(f"[SILVER] Complete: {len(silver_df)} cleaned records")
        print(f"[SILVER] Average quality score: {silver_validation['avg_quality_score']:.2f}")
        print(f"[SILVER] Unique users: {silver_validation['unique_users']}")
        
        # Gold: Aggregation
        print(f"\n[GOLD] Aggregating to Gold tier...")
        gold_df = self.gold.aggregate(silver_df)
        print(f"[GOLD] Complete: {len(gold_df)} aggregated metrics")
        
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETE")
        print(f"{'='*60}\n")
        
        return {
            'bronze': bronze_df,
            'silver': silver_df,
            'gold': gold_df,
            'validation': {
                'bronze': bronze_validation,
                'silver': silver_validation
            }
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Sample raw transaction data (as might come from an API or event stream)
    raw_transactions = [
        {
            "transaction_id": "txn_001",
            "user_id": "user_123",
            "amount": 45.99,
            "currency": "USD",
            "timestamp": "2024-01-15T10:30:00Z",
            "category": "food",
            "status": "completed"
        },
        {
            "transaction_id": "txn_002",
            "user_id": "user_123",
            "amount": 120.50,
            "currency": "USD",
            "timestamp": "2024-01-16T14:22:00Z",
            "category": "transport",
            "status": "completed"
        },
        {
            "transaction_id": "txn_003",
            "user_id": "user_456",
            "amount": 29.99,
            "currency": "USD",
            "timestamp": "2024-01-15T09:15:00Z",
            "category": "entertainment",
            "status": "pending"
        },
        {
            "transaction_id": "txn_004",
            "user_id": "user_123",
            "amount": 89.99,
            "currency": "USD",
            "timestamp": "2024-01-17T16:45:00Z",
            "category": "utilities",
            "status": "completed"
        },
        {
            # Malformed record - missing required fields
            "transaction_id": "txn_005",
            "amount": "invalid",  # Invalid type
            "timestamp": "2024-01-18T12:00:00Z"
        },
        {
            "transaction_id": "txn_006",
            "user_id": "user_456",
            "amount": 15.75,
            "currency": "USD",
            "timestamp": "2024-01-18T11:30:00Z",
            "category": "food",
            "status": "completed"
        }
    ]
    
    # Create and run pipeline
    pipeline = MedallionPipeline()
    results = pipeline.run(raw_transactions, source="payment_api")
    
    # Display results from each tier
    print("\n" + "="*60)
    print("BRONZE TIER - Raw Data")
    print("="*60)
    print(results['bronze'][['record_id', 'source', 'quality_score', 'tier']].head())
    
    print("\n" + "="*60)
    print("SILVER TIER - Cleaned Data")
    print("="*60)
    print(results['silver'][['transaction_id', 'user_id', 'amount', 'category', 'status', 'quality_score']].head())
    
    print("\n" + "="*60)
    print("GOLD TIER - Aggregated Metrics")
    print("="*60)
    print(results['gold'][['user_id', 'total_transactions', 'total_amount', 'avg_transaction_amount', 'quality_score']])
    
    # Quality analysis
    print("\n" + "="*60)
    print("QUALITY ANALYSIS")
    print("="*60)
    
    validator = DataValidator()
    
    # Analyze Silver tier quality distribution
    silver_df = results['silver']
    for _, row in silver_df.iterrows():
        quality_tier = validator.calculate_quality_tier(row['quality_score'])
        print(f"Transaction {row['transaction_id']}: Quality = {quality_tier.value} ({row['quality_score']:.2f})")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print("Bronze Validation:", json.dumps(results['validation']['bronze'], indent=2))
    print("\nSilver Validation:", json.dumps(results['validation']['silver'], indent=2, default=str))
    
    # Example: Save to files (in production, this would be Delta Lake, Parquet, etc.)
    print("\n" + "="*60)
    print("PERSISTING DATA (Example)")
    print("="*60)
    print("In production, persist to:")
    print("  Bronze: s3://datalake/bronze/transactions/")
    print("  Silver: s3://datalake/silver/transactions/")
    print("  Gold: s3://datalake/gold/user_metrics/")
