import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from app.files_preprocessing.readers.base_reader import DataReader


class DataPreprocessor:
  data = None
  def __init__(self, reader:DataReader ):
      self.reader = reader
      self.data = None
      self.preprocessed_data = None
      self.preprocessed_label = None

  def read_data(self, url: str):
      self.data = self.reader.read_data(url)

  def preprocess_data_for_predict(self):
    df = pd.DataFrame(self.data)
    numeric = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
    categorical = df.select_dtypes(include=['object', 'category', 'bool', 'datetime64']).columns.to_list()
    numeric_transformer = Pipeline(
      steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
      ]
    )
    categorical_transformer = Pipeline(
      steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
      ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
          ('num', numeric_transformer, numeric),
          ('cat', categorical_transformer, categorical)
      ]
    )
    self.preprocessed_data = preprocessor.fit_transform(df)
    print(self.preprocessed_data)
    
  def automatic_preprocess_data(self, prediction_column):
    df = pd.DataFrame(self.data)
    print(df)
    X = df.drop(prediction_column, axis=1)
    y = df[[prediction_column]]
    numeric = X.select_dtypes(include=['int64', 'float64']).columns.to_list()
    categorical = X.select_dtypes(include=['object', 'category', 'bool', 'datetime64']).columns.to_list()
    numeric_transformer = Pipeline(
      steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
      ]
    )
    categorical_transformer = Pipeline(
      steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
      ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
          ('num', numeric_transformer, numeric),
          ('cat', categorical_transformer, categorical)
      ]
    )
    print(y)
    self.preprocessed_data = preprocessor.fit_transform(X)
    self.preprocessed_label = categorical_transformer.fit_transform(y)
    self.label_encoder = OneHotEncoder(sparse_output=False)
    self.preprocessed_label = self.label_encoder.fit_transform(y.values.reshape(-1, 1))  
    self.classes = self.label_encoder.categories_[0]

