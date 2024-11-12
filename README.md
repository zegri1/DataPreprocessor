# DataPreprocessor Documentation

## Table of Contents
- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Core Operations](#core-operations)
  - [Missing Values](#missing-values)
  - [Outliers](#outliers)
  - [Encoding](#encoding)
  - [Scaling](#scaling)
- [Using with New Data](#using-with-new-data)
- [Saving and Loading](#saving-and-loading)

## Overview

DataPreprocessor is a library that simplifies data preprocessing for machine learning through a fluent, chainable API. It handles common preprocessing tasks like missing values, outliers, encoding, and scaling.

## Basic Usage

```python
from datapreprocessor import DataPreprocessor

# Create and configure preprocessor
preprocessor = (DataPreprocessor()
    .fill_missing()
        .numeric(['age', 'salary'])
            .with_mean()
        .categorical(['education'])
            .with_mode()
        .done()
    .scale()
        .standard(['age', 'salary'])
        .done())

# Apply to data
processed_df = preprocessor.fit_transform(df)
```

## Core Operations

### Missing Values

The `.fill_missing()` operation handles missing values in your dataset. It provides separate methods for numeric and categorical columns.

#### Method Chain Structure
```python
.fill_missing()
    .numeric(columns)      # Select numeric columns
    .categorical(columns)  # Select categorical columns
    .done()               # Complete the operation
```

#### Selecting Columns
Both `.numeric()` and `.categorical()` accept either a single column name or a list of columns:
```python
.numeric('age')                      # Single column
.numeric(['age', 'salary'])          # Multiple columns
```

#### Filling Strategies

For numeric columns:
- `.with_mean()`: Fill with column mean
- `.with_median()`: Fill with column median
- `.with_knn(n_neighbors=5)`: Use KNN imputation

For categorical columns:
- `.with_mode()`: Fill with most frequent value

#### Using .where()
`.where()` defines specific fill values or strategies for individual columns. It can be used:
1. After a strategy method to override it for specific columns:
```python
.numeric(['age', 'salary'])
    .with_mean()                     # Default strategy
    .where(age='median', salary=0)   # Override for specific columns
```

2. Directly after column selection to set strategies without a default:
```python
.numeric(['age', 'salary'])
    .where(age='mean', salary=0)     # Set strategy for each column
```

Accepted values in .where():
- For numeric columns:
  - Strings: 'mean', 'median'
  - Numbers: Any constant value
  - Custom functions: Function that takes a Series and returns a value
- For categorical columns:
  - 'mode': Use most frequent value
  - Any string: Use as constant fill value


### Outliers

The `.handle_outliers()` operation detects and handles outliers in numeric columns. During training, it can remove or clip outliers; during inference, it will only flag them.

#### Method Structure
```python
.handle_outliers()
    .columns(columns)    # Select columns to process
    .using_method()      # Specify outlier detection method
    .done()
```

#### Available Methods

1. **IQR Method**
```python
.handle_outliers()
    .columns(['salary', 'age'])
    .using_iqr(k=1.5)    # Values outside Q1 - k*IQR, Q3 + k*IQR are outliers
    .done()
```
- `k`: Multiplier for IQR range (default=1.5)
- Identifies outliers using the interquartile range method
- More robust to extreme values than z-score

2. **Z-Score Method**
```python
.handle_outliers()
    .columns(['score'])
    .using_zscore(threshold=3)    # Values beyond 3 standard deviations
    .done()
```
- `threshold`: Number of standard deviations (default=3)
- Identifies outliers based on distance from mean in standard deviations
- Assumes normally distributed data

3. **Percentile Method**
```python
.handle_outliers()
    .columns(['rating'])
    .using_percentile(lower=1, upper=99)    # Keep only 1st to 99th percentile
    .done()
```
- `lower`: Lower percentile boundary (default=1)
- `upper`: Upper percentile boundary (default=99)
- Identifies outliers based on percentile thresholds
- Useful when you want to keep a specific proportion of the data

### Encoding

The `.encode()` operation transforms categorical variables into numeric format. Multiple encoding methods can be chained together.

#### Method Structure
```python
.encode()
    .method(columns)    # Apply encoding method to columns
    .done()
```

#### Available Methods

1. **One-Hot Encoding**
```python
.encode()
    .onehot(['department', 'city'])    # Creates binary columns for each category
    .done()
```
- Creates dummy variables (0/1) for each category
- Handles new categories during inference using `handle_unknown='ignore'`
- Column names follow pattern: `{original_column}_{category}`

2. **Label Encoding**
```python
.encode()
    .label(['gender', 'color'])    # Converts categories to integers
    .done()
```
- Converts categories to sequential integers (0, 1, 2...)
- Maintains the same mapping for inference
- Best for binary categories or when ordinal relationship doesn't matter

3. **Ordinal Encoding**
```python
.encode()
    .ordinal('education', 
            order=['HS', 'BS', 'MS', 'PhD'])    # Specify explicit ordering
    .done()
```
- Converts categories to ordered integers based on specified sequence
- Requires explicit ordering of categories
- Preserves ordinal relationships between categories

### Scaling

The `.scale()` operation normalizes numeric features. Different scaling methods can be applied to different columns.

#### Method Structure
```python
.scale()
    .method(columns)    # Apply scaling method to columns
    .done()
```

#### Available Methods

1. **Standard Scaling**
```python
.scale()
    .standard(['age', 'salary'])    # Scale to zero mean and unit variance
    .done()
```
- Transforms features to have zero mean and unit variance
- `x_scaled = (x - mean) / std`
- Best for algorithms that assume normally distributed data (e.g., neural networks)

2. **Min-Max Scaling**
```python
.scale()
    .minmax(['score'], range=(0, 10))    # Scale to specific range
    .done()
```
- Scales features to a fixed range
- `x_scaled = (x - min) * (range[1] - range[0]) / (max - min) + range[0]`
- Default range is (0, 1)
- Best when you need bounded values

3. **Robust Scaling**
```python
.scale()
    .robust(['rating', 'score'])    # Scale using statistics robust to outliers
    .done()
```
- Uses statistics that are robust to outliers (median and IQR)
- `x_scaled = (x - median) / IQR`
- Best when data contains outliers

#### Combining Multiple Operations

You can combine all operations in a single chain, maintaining a clear preprocessing pipeline:

```python
preprocessor = (DataPreprocessor()
    .fill_missing()
        .numeric(['age', 'salary'])
            .with_mean()
            .where(salary=0)
        .categorical(['education'])
            .with_mode()
        .done()
    .handle_outliers()
        .columns(['salary'])
        .using_iqr(k=2.0)
        .done()
    .encode()
        .onehot(['department'])
        .ordinal('education', ['HS', 'BS', 'MS', 'PhD'])
        .done()
    .scale()
        .standard(['age'])
        .minmax('salary', range=(0, 1))
        .done())
```

### Transform Behavior

When using `.transform()` on new data:

1. **Missing Values**: Applied the same way as training
2. **Outliers**: Only flagged, not transformed
3. **Encoding**: 
   - One-hot: New categories are handled based on `handle_unknown`
   - Label/Ordinal: New categories raise an error
4. **Scaling**: Uses training data statistics

```python
# During training
train_processed = preprocessor.fit_transform(train_df)

# During inference
test_processed = preprocessor.transform(test_df)
```
## Saving and Loading

Save and load preprocessor configurations:

```python
# Save
preprocessor.save('preprocessor.joblib')

# Load
loaded_preprocessor = DataPreprocessor().load('preprocessor.joblib')
```

## Best Practices

1. Handle missing values before outliers
2. Encode categorical variables before scaling
3. Use consistent strategies across related features
4. Save preprocessor after fitting for consistent transformations

