from util.get_dataset import TrainDataset, TestDataset
from sklearn.model_selection import train_test_split
import numpy as np
def TrainDataset_prepared():
    x_train, x_val, y_train, y_val = TrainDataset()
    return x_train, x_val, y_train, y_val

def TestDataset_prepared():
    x_test, y_test = TestDataset()
    return x_test, y_test
x_train, x_val, y_train, y_val = TrainDataset_prepared()
y_train = y_train.astype(np.int64)
y_val = y_val.astype(np.int64)


x_train_20, x_temp, y_train_20, y_temp = train_test_split(x_train, y_train, test_size=0.80, random_state=42)
x_val_20, x_val_temp, y_val_20, y_val_temp = train_test_split(x_val, y_val, test_size=0.80, random_state=42)

x_train_15, x_temp_15, y_train_15, y_temp_15 = train_test_split(x_train_20, y_train_20, test_size=0.25, random_state=42)
x_val_15, x_val_temp_15, y_val_15, y_val_temp_15 = train_test_split(x_val_20, y_val_20, test_size=0.25, random_state=42)

x_train_10, x_temp_10, y_train_10, y_temp_10 = train_test_split(x_train_15, y_train_15, test_size=1/3, random_state=42)
x_val_10, x_val_temp_10, y_val_10, y_val_temp_10 = train_test_split(x_val_15, y_val_15, test_size=1/3, random_state=42)

x_train_5, x_temp_5, y_train_5, y_temp_5 = train_test_split(x_train_10, y_train_10, test_size=0.5, random_state=42)
x_val_5, x_val_temp_5, y_val_5, y_val_temp_5 = train_test_split(x_val_10, y_val_10, test_size=0.5, random_state=42)

x_train_3, x_temp_3, y_train_3, y_temp_3 = train_test_split(x_train_5, y_train_5, test_size=0.4, random_state=42)
x_val_3, x_val_temp_3, y_val_3, y_val_temp_3 = train_test_split(x_val_5, y_val_5, test_size=0.4, random_state=42)
