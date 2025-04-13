import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('c:\\Users\\irina\\Desktop\\dataset.csv')

train, test = train_test_split(data, test_size=0.5, stratify=data['NObeyesdad'], random_state=42)

train.to_csv('train_data.csv', index=False)
test.to_csv('test_data.csv', index=False)

print("Данные успешно разделены на train_data.csv и test_data.csv")