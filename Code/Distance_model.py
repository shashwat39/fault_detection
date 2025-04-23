import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
from tensorflow import keras
from keras import layers 
from keras import callbacks
from sklearn.compose import make_column_transformer 
from sklearn.model_selection import ShuffleSplit, train_test_split 
from sklearn.utils import shuffle

LineData = pd.read_csv("../Data/DataLine1.csv")
# LineData = pd.read_csv("../Data/DataLine2.csv")
# LineData = pd.read_csv("../Data/DataLine3.csv") 

# Shuffling
LineData_suffled = shuffle(LineData, random_state = 2)
LineData_suffled.head()

features_col = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"] 
target_col = ["Distance"]
X = LineData_suffled[features_col] 
Y = LineData_suffled[target_col] 

scalling = StandardScaler()
X_scalled = scalling.fit_transform(X, Y) 

x_train, x_val, y_train, y_val  = train_test_split(X_scalled, Y, test_size=0.15, random_state=10) 

Dis_model = keras.Sequential([
    layers.Dense(60, activation='relu', input_shape=[6]), 
    layers.BatchNormalization(),
    layers.Dense(100, activation='tanh'), 
    layers.BatchNormalization(),
    layers.Dense(80, activation='relu'),
    layers.BatchNormalization(),  
    layers.Dense(80, activation='relu'), 
    layers.Dense(1) 
])


Dis_model.compile(
    optimizer = 'adam',
    loss = 'mae',
)

history = Dis_model.fit(
    x_train, y_train,
    validation_data = (x_val, y_val),
    batch_size = 120,  
    epochs = 1000,
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Train Loss: {}".format(history_df['loss'].min()))
print("Minimum Validation Loss: {}".format(history_df['val_loss'].min()))


y_pred = Dis_model.predict(x_val)
y_val_series = y_val.squeeze()
y_pred_series = y_pred.squeeze()
comparison = pd.DataFrame({
    "True": np.array(y_val_series), 
    "Predicted": np.array(y_pred_series)  
})

# Showing the True values and Predicted values on Validation data
comparison.head()