# import pandas as pd
# from sklearn.model_selection import train_test_split

# data = pd.read_csv('test_data.csv')
# data.drop("NObeyesdad", axis=1, inplace=True)
# # train, test = train_test_split(data, test_size=0.5, stratify=data['NObeyesdad'], random_state=42)

# # train.to_csv('train_data.csv', index=False)
# # test.to_csv('test_data.csv', index=False)
# data.to_csv('test_daya.csv', index=False)
# print("Данные успешно разделены на train_data.csv и test_data.csv")

import pickle

with open("c:\\Users\\irina\\Desktop\\diplom\\1.pkl", 'rb') as f:
  saved_objects = pickle.load(f)
  model = saved_objects["neural_network"]


print(f"pickle {model.layers[0].weights}")