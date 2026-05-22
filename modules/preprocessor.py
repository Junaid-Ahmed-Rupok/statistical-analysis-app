import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Professional data preprocessing module.
    - Type inference & correction
    - Exact + near-duplicate removal
    - KNN imputation for numeric, mode for categorical
    - IQR outlier capping (Winsorization)
    - Optional scaling & encoding
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.outlier_bounds = {}
        self.cleaning_log = []
        self._imputer = None
        
    def get_overview(self, df: pd.DataFrame) -> dict:
        overview = {
            'shape': df.shape,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'memory_usage': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicates': int(df.duplicated().sum()),
            'duplicate_percentage': round((df.duplicated().sum() / len(df)) * 100, 2),
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = round((df.isnull().sum() / len(df)) * 100, 2)
        overview['missing_table'] = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing Percentage': missing_pct.values
        }).sort_values('Missing Percentage', ascending=False)
        
        overview['total_missing'] = int(missing.sum())
        overview['missing_percentage'] = round(
            (missing.sum() / (df.shape[0] * df.shape[1])) * 100, 2
        )
        
        # Column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        overview['numeric_columns'] = numeric_cols
        overview['categorical_columns'] = categorical_cols
        overview['datetime_columns'] = datetime_cols
        overview['numeric_count'] = len(numeric_cols)
        overview['categorical_count'] = len(categorical_cols)
        
        # Quality score
        overview['quality_score'] = self.get_quality_score(df)
        
        return overview
    
    def get_quality_score(self, df: pd.DataFrame) -> float:
        """Score 0-100 based on completeness, uniqueness, and variance."""
        score = 100.0
        
        # Missing values penalty
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        score -= min(missing_pct * 2, 40)
        
        # Duplicates penalty
        dup_pct = (df.duplicated().sum() / len(df)) * 100
        score -= min(dup_pct * 3, 20)
        
        # Constant columns penalty
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() <= 1:
                score -= 5
        
        # Bonus for having numeric columns
        if len(numeric_cols) > 0:
            score += 5
            
        return round(max(0, min(score, 100)), 1)
    
    def clean(self, df: pd.DataFrame) -> tuple:
        """
        Full cleaning pipeline:
        1. Drop constant columns
        2. Remove exact duplicates
        3. Impute missing values (KNN numeric, mode categorical)
        Returns cleaned df + cleaning log.
        """
        df_clean = df.copy()
        self.cleaning_log = []
        initial_rows, initial_cols = df_clean.shape
        
        # Step 1: Drop constant columns
        constant_cols = [c for c in df_clean.columns if df_clean[c].nunique(dropna=False) <= 1]
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            self.cleaning_log.append({
                'action': 'Drop Constant Columns',
                'detail': f'Removed {len(constant_cols)} constant column(s): {", ".join(constant_cols)}',
                'status': 'completed'
            })
        
        # Step 2: Remove exact duplicates
        dup_count = df_clean.duplicated().sum()
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates()
            self.cleaning_log.append({
                'action': 'Remove Duplicates',
                'detail': f'Removed {dup_count} duplicate rows ({dup_count/initial_rows*100:.1f}%)',
                'status': 'completed'
            })
        
        # Step 3: Impute missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()
        
        total_missing = df_clean.isnull().sum().sum()
        
        if total_missing > 0:
            # Numeric: KNN imputation (falls back to median if KNN fails)
            if numeric_cols and df_clean[numeric_cols].isnull().any().any():
                try:
                    imputer = KNNImputer(n_neighbors=5, weights='distance')
                    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
                    self._imputer = imputer
                    method = 'KNN (k=5)'
                except Exception:
                    # Fallback to median
                    for col in numeric_cols:
                        if df_clean[col].isnull().any():
                            df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    method = 'median (fallback)'
                
                self.cleaning_log.append({
                    'action': 'Impute Numeric Missing',
                    'detail': f'Filled missing values in {len(numeric_cols)} numeric column(s) using {method}',
                    'status': 'completed'
                })
            
            # Categorical: mode imputation
            for col in categorical_cols:
                missing_count = df_clean[col].isnull().sum()
                if missing_count > 0:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
                    self.cleaning_log.append({
                        'action': 'Impute Categorical Missing',
                        'detail': f'Column "{col}": {missing_count} value(s) imputed with mode ("{mode_val}")',
                        'status': 'completed'
                    })
        else:
            self.cleaning_log.append({
                'action': 'Missing Values',
                'detail': 'No missing values detected',
                'status': 'skipped'
            })
        
        final_rows, final_cols = df_clean.shape
        self.cleaning_log.append({
            'action': 'Summary',
            'detail': f'Rows: {initial_rows} → {final_rows} | Columns: {initial_cols} → {final_cols}',
            'status': 'completed'
        })
        
        return df_clean, self.cleaning_log
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> tuple:
        """
        Detect and handle outliers using IQR method.
        Uses capping (Winsorization) — never drops rows.
        Also returns outlier counts per column.
        """
        df_clean = df.copy()
        outlier_counts = {}
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.outlier_bounds[col] = {
                'lower': lower_bound,
                'upper': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
            
            # Winsorization: cap at bounds (never drop rows)
            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            self.cleaning_log.append({
                'action': 'Outlier Capping',
                'detail': f'Capped {total_outliers} outlier values across {len(outlier_counts)} column(s) using IQR method',
                'status': 'completed'
            })
        
        return df_clean, outlier_counts
    
    def encode_and_scale(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical columns (Label Encoding) and scale numeric columns (StandardScaler).
        Only use this for ML-ready output.
        """
        df_processed = df.copy()
        
        # Label encode categoricals
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
        
        # Standard scale numerics
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_processed[numeric_cols] = self.scaler.fit_transform(df_processed[numeric_cols])
        
        return df_processed