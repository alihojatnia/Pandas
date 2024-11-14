### Pandas

#### Handling Missing Values
```
import pandas as pd

# Replace missing values in a numeric column with its mean
df['numeric_column'].fillna(df['numeric_column'].mean(), inplace=True)

# Replace missing values in a categorical column with a placeholder
df['category_column'].fillna('Unknown', inplace=True)

# Drop rows with any missing values
df.dropna(inplace=True)

```

#### Creating New Features
```
# Create a new feature by multiplying two columns
df['new_feature'] = df['column_A'] * df['column_B']

# Extract year, month, and day from a datetime column
df['year'] = pd.to_datetime(df['date_column']).dt.year
df['month'] = pd.to_datetime(df['date_column']).dt.month
df['day'] = pd.to_datetime(df['date_column']).dt.day

```

#### Binning and Discretization
```
# Bin a numeric column into categories
df['binned_feature'] = pd.cut(df['numeric_column'], bins=[0, 10, 20, 30, 100], 
                              labels=['low', 'medium', 'high', 'very_high'])

```

#### One-Hot Encoding
```
# One-hot encode a categorical column
df = pd.get_dummies(df, columns=['category_column'], drop_first=True)

```

#### Label Encoding
```
from sklearn.preprocessing import LabelEncoder

# Apply label encoding to a categorical column
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category_column'])

```

#### Normalization and Scaling
```
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Normalize numeric columns using Min-Max Scaling
scaler = MinMaxScaler()
df[['numeric_column_A', 'numeric_column_B']] = scaler.fit_transform(df[['numeric_column_A', 'numeric_column_B']])

# Standardize numeric columns using Standard Scaler
standardizer = StandardScaler()
df[['numeric_column_A', 'numeric_column_B']] = standardizer.fit_transform(df[['numeric_column_A', 'numeric_column_B']])

```

#### Log Transformation
```
# Apply log transformation to reduce skewness in a numeric column
df['log_transformed'] = df['numeric_column'].apply(lambda x: np.log1p(x))

```

#### Interaction Features
```
# Create an interaction feature by multiplying two columns
df['interaction_feature'] = df['column_A'] * df['column_B']

```

#### Date and Time Features
```
# Extract features from a datetime column
df['day_of_week'] = pd.to_datetime(df['date_column']).dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

```

#### Target Encoding
```
# Perform target encoding for a categorical column based on the mean of the target
mean_encoded = df.groupby('category_column')['target_column'].mean()
df['encoded_feature'] = df['category_column'].map(mean_encoded)

```

#### Feature Hashing
```
from sklearn.feature_extraction import FeatureHasher

# Apply feature hashing to a high-cardinality categorical column
hasher = FeatureHasher(input_type='string', n_features=10)
hashed_features = hasher.transform(df['high_cardinality_column'].astype(str))
hashed_df = pd.DataFrame(hashed_features.toarray())

```

#### Outlier Removal
```
# Remove outliers from a numeric column using the IQR method
Q1 = df['numeric_column'].quantile(0.25)
Q3 = df['numeric_column'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['numeric_column'] >= (Q1 - 1.5 * IQR)) & (df['numeric_column'] <= (Q3 + 1.5 * IQR))]

```

#### Polynomial Features
```
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features for numeric columns
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['numeric_column_A', 'numeric_column_B']])
poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['numeric_column_A', 'numeric_column_B']))

```

#### Groupby Aggregation Features
```
# Create new features using groupby aggregation
df['mean_feature_by_group'] = df.groupby('group_column')['numeric_column'].transform('mean')
df['sum_feature_by_group'] = df.groupby('group_column')['numeric_column'].transform('sum')

```
