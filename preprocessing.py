from typing import List, Union, Dict, Optional, Any, TypeVar
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, \
    OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import logging


@dataclass
class BuilderConfig:
    """Configuration container for preprocessing operations"""
    operation: str
    config: Dict[str, Any]


class DataPreprocessor:
    def __init__(self, logging_level: int = logging.INFO):
        self._transformers: Dict[str, Any] = {}
        self._pending_operations: Dict[str, Any] = {}
        self._trained: bool = False
        logging.basicConfig(level=logging_level)
        self.logger = logging.getLogger(__name__)

    class NumericMissingChain:
        def __init__(self, builder: 'DataPreprocessor.MissingValuesBuilder'):
            self.builder = builder

        def with_mean(self) -> 'DataPreprocessor.MissingValuesBuilder':
            for col in self.builder._current_columns:
                self.builder.config['numeric'][col] = 'mean'
            return self.builder

        def with_median(self) -> 'DataPreprocessor.MissingValuesBuilder':
            for col in self.builder._current_columns:
                self.builder.config['numeric'][col] = 'median'
            return self.builder

        def with_knn(self, n_neighbors: int = 5) -> 'DataPreprocessor.MissingValuesBuilder':
            self.builder.config['knn'] = {
                'columns': self.builder._current_columns,
                'n_neighbors': n_neighbors
            }
            return self.builder

        def where(self, **values: Any) -> 'DataPreprocessor.MissingValuesBuilder':
            for col, value in values.items():
                self.builder.config['numeric'][col] = value
            return self.builder

    class CategoricalMissingChain:
        def __init__(self, builder: 'DataPreprocessor.MissingValuesBuilder'):
            self.builder = builder

        def with_mode(self) -> 'DataPreprocessor.MissingValuesBuilder':
            for col in self.builder._current_columns:
                self.builder.config['categorical'][col] = 'mode'
            return self.builder

        def where(self, **values: str) -> 'DataPreprocessor.MissingValuesBuilder':
            for col, value in values.items():
                self.builder.config['categorical'][col] = value
            return self.builder

    class MissingValuesBuilder:
        def __init__(self, preprocessor: 'DataPreprocessor'):
            self.preprocessor = preprocessor
            self.config = {'numeric': {}, 'categorical': {}, 'knn': None}
            self._current_columns: Optional[List[str]] = None

        def numeric(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.NumericMissingChain':
            self._current_columns = [columns] if isinstance(columns, str) else columns
            return DataPreprocessor.NumericMissingChain(self)

        def categorical(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.CategoricalMissingChain':
            self._current_columns = [columns] if isinstance(columns, str) else columns
            return DataPreprocessor.CategoricalMissingChain(self)

        def done(self) -> 'DataPreprocessor':
            self.preprocessor._pending_operations['missing_values'] = self.config
            return self.preprocessor

    class OutliersMethodChain:
        def __init__(self, builder: 'DataPreprocessor.OutliersBuilder'):
            self.builder = builder

        def using_iqr(self, k: float = 1.5) -> 'DataPreprocessor.OutliersBuilder':
            self.builder.config['method'] = 'iqr'
            self.builder.config['params'] = {'k': k}
            return self.builder

        def using_zscore(self, threshold: float = 3.0) -> 'DataPreprocessor.OutliersBuilder':
            self.builder.config['method'] = 'zscore'
            self.builder.config['params'] = {'threshold': threshold}
            return self.builder

        def using_percentile(self, lower: float = 1.0, upper: float = 99.0) -> 'DataPreprocessor.OutliersBuilder':
            self.builder.config['method'] = 'percentile'
            self.builder.config['params'] = {'lower': lower, 'upper': upper}
            return self.builder

    class OutliersBuilder:
        def __init__(self, preprocessor: 'DataPreprocessor'):
            self.preprocessor = preprocessor
            self.config: Dict[str, Any] = {}

        def columns(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.OutliersMethodChain':
            self.config['columns'] = [columns] if isinstance(columns, str) else columns
            return DataPreprocessor.OutliersMethodChain(self)

        def done(self) -> 'DataPreprocessor':
            self.preprocessor._pending_operations['outliers'] = self.config
            return self.preprocessor

    class EncodingBuilder:
        def __init__(self, preprocessor: 'DataPreprocessor'):
            self.preprocessor = preprocessor
            self.config: Dict[str, Any] = {}

        def onehot(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.EncodingBuilder':
            cols = [columns] if isinstance(columns, str) else columns
            if 'onehot' not in self.config:
                self.config['onehot'] = []
            self.config['onehot'].extend(cols)
            return self

        def label(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.EncodingBuilder':
            cols = [columns] if isinstance(columns, str) else columns
            if 'label' not in self.config:
                self.config['label'] = []
            self.config['label'].extend(cols)
            return self

        def ordinal(self, column: str, order: List[Any]) -> 'DataPreprocessor.EncodingBuilder':
            if 'ordinal' not in self.config:
                self.config['ordinal'] = {}
            self.config['ordinal'][column] = order
            return self

        def done(self) -> 'DataPreprocessor':
            self.preprocessor._pending_operations['encoding'] = self.config
            return self.preprocessor

    class ScalingBuilder:
        def __init__(self, preprocessor: 'DataPreprocessor'):
            self.preprocessor = preprocessor
            self.config: Dict[str, Any] = {}

        def standard(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.ScalingBuilder':
            cols = [columns] if isinstance(columns, str) else columns
            if 'standard' not in self.config:
                self.config['standard'] = []
            self.config['standard'].extend(cols)
            return self

        def minmax(self, columns: Union[str, List[str]], range: tuple = (0, 1)) -> 'DataPreprocessor.ScalingBuilder':
            cols = [columns] if isinstance(columns, str) else columns
            if 'minmax' not in self.config:
                self.config['minmax'] = {}
            for col in cols:
                self.config['minmax'][col] = range
            return self

        def robust(self, columns: Union[str, List[str]]) -> 'DataPreprocessor.ScalingBuilder':
            cols = [columns] if isinstance(columns, str) else columns
            if 'robust' not in self.config:
                self.config['robust'] = []
            self.config['robust'].extend(cols)
            return self

        def done(self) -> 'DataPreprocessor':
            self.preprocessor._pending_operations['scaling'] = self.config
            return self.preprocessor

    def fill_missing(self) -> MissingValuesBuilder:
        return self.MissingValuesBuilder(self)

    def handle_outliers(self) -> OutliersBuilder:
        return self.OutliersBuilder(self)

    def encode(self) -> EncodingBuilder:
        return self.EncodingBuilder(self)

    def scale(self) -> ScalingBuilder:
        return self.ScalingBuilder(self)

    def _apply_missing_values(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply missing value handling based on configuration."""
        df_copy = df.copy()

        # Handle numeric columns
        for col, strategy in config.get('numeric', {}).items():
            if isinstance(strategy, (int, float)):
                df_copy[col] = df_copy[col].fillna(strategy)
                self._transformers[f'fill_{col}'] = strategy
            else:
                imputer = SimpleImputer(strategy=strategy)
                df_copy[col] = imputer.fit_transform(df_copy[[col]])
                self._transformers[f'fill_{col}'] = imputer

        # Handle categorical columns
        for col, strategy in config.get('categorical', {}).items():
            if strategy == 'mode':
                fill_value = df_copy[col].mode()[0]
            else:
                fill_value = strategy
            df_copy[col] = df_copy[col].fillna(fill_value)
            self._transformers[f'fill_{col}'] = fill_value

        # Handle KNN imputation
        if 'knn' in config:
            cols = config['knn']['columns']
            imputer = KNNImputer(n_neighbors=config['knn']['n_neighbors'])
            df_copy[cols] = imputer.fit_transform(df_copy[cols])
            self._transformers['knn_imputer'] = imputer

        return df_copy

    def _apply_outliers(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply outlier handling based on configuration."""
        df_copy = df.copy()
        method = config['method']
        columns = config['columns']
        params = config['params']

        for col in columns:
            if method == 'iqr':
                q1 = df_copy[col].quantile(0.25)
                q3 = df_copy[col].quantile(0.75)
                iqr = q3 - q1
                k = params.get('k', 1.5)
                lower = q1 - k * iqr
                upper = q3 + k * iqr
                self._transformers[f'outliers_{col}'] = {'method': 'iqr', 'lower': lower, 'upper': upper}

            elif method == 'zscore':
                threshold = params.get('threshold', 3)
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                z_scores = np.abs((df_copy[col] - mean) / std)
                self._transformers[f'outliers_{col}'] = {
                    'method': 'zscore',
                    'mean': mean,
                    'std': std,
                    'threshold': threshold
                }

            elif method == 'percentile':
                lower = np.percentile(df_copy[col], params.get('lower', 1))
                upper = np.percentile(df_copy[col], params.get('upper', 99))
                self._transformers[f'outliers_{col}'] = {
                    'method': 'percentile',
                    'lower': lower,
                    'upper': upper
                }

        return df_copy

    def _apply_encoding(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply categorical encoding based on configuration."""
        df_copy = df.copy()

        # One-hot encoding
        for col in config.get('onehot', []):
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(df_copy[[col]])
            encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
            df_copy = pd.concat([
                df_copy.drop(columns=[col]),
                pd.DataFrame(encoded, columns=encoded_cols, index=df_copy.index)
            ], axis=1)
            self._transformers[f'onehot_{col}'] = encoder

        # Label encoding
        for col in config.get('label', []):
            encoder = LabelEncoder()
            df_copy[col] = encoder.fit_transform(df_copy[col])
            self._transformers[f'label_{col}'] = encoder

        # Ordinal encoding
        for col, categories in config.get('ordinal', {}).items():
            encoder = OrdinalEncoder(categories=[categories])
            df_copy[col] = encoder.fit_transform(df_copy[[col]])
            self._transformers[f'ordinal_{col}'] = encoder

        return df_copy

    def _apply_scaling(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Apply feature scaling based on configuration."""
        df_copy = df.copy()

        # Standard scaling
        for col in config.get('standard', []):
            scaler = StandardScaler()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
            self._transformers[f'scale_{col}'] = scaler

        # Min-max scaling
        for col, range_vals in config.get('minmax', {}).items():
            scaler = MinMaxScaler(feature_range=range_vals)
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
            self._transformers[f'scale_{col}'] = scaler

        # Robust scaling
        for col in config.get('robust', []):
            scaler = RobustScaler()
            df_copy[col] = scaler.fit_transform(df_copy[[col]])
            self._transformers[f'scale_{col}'] = scaler

        return df_copy

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all configured transformations to the input DataFrame."""
        result = df.copy()

        # Apply transformations in order
        for operation in ['missing_values', 'outliers', 'encoding', 'scaling']:
            if operation in self._pending_operations:
                config = self._pending_operations[operation]
                if operation == 'missing_values':
                    result = self._apply_missing_values(result, config)
                elif operation == 'outliers':
                    result = self._apply_outliers(result, config)
                elif operation == 'encoding':
                    result = self._apply_encoding(result, config)
                elif operation == 'scaling':
                    result = self._apply_scaling(result, config)

        self._trained = True
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted transformations."""
        if not self._trained:
            raise ValueError("Must call fit_transform() before transform()")

        result = df.copy()

        # Apply transformations in order
        for name, transformer in self._transformers.items():
            operation, col = name.split('_', 1)

            if operation == 'fill':
                if isinstance(transformer, SimpleImputer):
                    result[col] = transformer.transform(result[[col]])
                else:
                    result[col] = result[col].fillna(transformer)

            elif operation == 'knn' and isinstance(transformer, KNNImputer):
                result[transformer.feature_names_in_] = transformer.transform(
                    result[transformer.feature_names_in_])

            elif operation == 'outliers':
                method = transformer['method']
                if method == 'iqr':
                    outliers = (result[col] < transformer['lower']) | (result[col] > transformer['upper'])
                    if outliers.any():
                        self.logger.warning(f"Found {outliers.sum()} outliers in {col}")
                elif method == 'zscore':
                    z_scores = np.abs((result[col] - transformer['mean']) / transformer['std'])
                    outliers = z_scores > transformer['threshold']
                    if outliers.any():
                        self.logger.warning(f"Found {outliers.sum()} outliers in {col}")
                elif method == 'percentile':
                    outliers = (result[col] < transformer['lower']) | (result[col] > transformer['upper'])
                    if outliers.any():
                        self.logger.warning(f"Found {outliers.sum()} outliers in {col}")

            elif operation == 'onehot':
                encoded = transformer.transform(result[[col]])
                encoded_cols = [f"{col}_{cat}" for cat in transformer.categories_[0]]
                result = pd.concat([
                    result.drop(columns=[col]),
                    pd.DataFrame(encoded, columns=encoded_cols, index=result.index)
                ], axis=1)

            elif operation == 'label':
                result[col] = transformer.transform(result[col])

            elif operation == 'ordinal':
                result[col] = transformer.transform(result[[col]])

            elif operation == 'scale':
                result[col] = transformer.transform(result[[col]])

        return result

    def save(self, path: str) -> None:
        """Save the fitted preprocessor to disk."""
        import joblib
        joblib.dump({
            'transformers': self._transformers,
            'pending_operations': self._pending_operations,
            'trained': self._trained
        }, path)

    def load(self, path: str) -> 'DataPreprocessor':
        """Load a fitted preprocessor from disk."""
        import joblib
        data = joblib.load(path)
        self._transformers = data['transformers']
        self._pending_operations = data['pending_operations']
        self._trained = data['trained']
        return self
