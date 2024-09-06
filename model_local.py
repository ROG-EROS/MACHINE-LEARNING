# imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# loading data
file_name ="Iris.csv"
dataset = pd.read_csv(file_name)
X=dataset.drop(['Id','Species'],axis=1)
y=dataset['Species']
lbl_clf = LabelEncoder()
Y_encoded = lbl_clf.fit_transform(y)
Y_final = tf.keras.utils.to_categorical(Y_encoded)

x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.20,  random_state=42, stratify=Y_encoded,  shuffle=True)

std_clf = StandardScaler()
x_train_new = std_clf.fit_transform(x_train)
x_test_new = std_clf.transform(x_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_dim=4, activation=tf.nn.relu, kernel_initializer='he_normal', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(7, activation=tf.nn.relu, kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(5, activation=tf.nn.relu, kernel_initializer='he_normal',  kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))
model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
iris_model = model.fit(x_train_new, y_train, epochs=200, batch_size=5,validation_data = (x_test_new, y_test))



print("\n********************************************************************")
print("\t\t\t Model validation ..")
print("********************************************************************")
y_pred = model.predict(x_test_new)
y_pred_c = [val.argmax() for val in y_pred]
y_act_c = [val.argmax() for val in y_test]
print("======================================================================")
print("Accuracy : ")
print(round(accuracy_score(y_act_c, y_pred_c)*100,2), '%')
print("======================================================================")

print("======================================================================")
print("Confusion Matrix : ")
print(confusion_matrix(y_act_c, y_pred_c))
print("======================================================================")

print("======================================================================")
print("Classification Report : ")
print(classification_report(y_act_c, y_pred_c))
print("======================================================================")

fig = plt.figure(figsize=(16, 12))

# Plot Accuracy levels during traing
fig.add_subplot(2, 1, 1)
plt.plot(iris_model.history['accuracy'])
plt.plot(iris_model.history['val_accuracy'])
plt.legend(['Train', 'Validation'], loc='lower left')
plt.ylabel('Accuracy')
plt.xlabel('Epoch Number')
plt.title('Model accuracy during training SGD')
plt.show()

# Plot loss levels during traing
fig.add_subplot(2, 1, 2)
plt.plot(iris_model.history['loss'])
plt.plot(iris_model.history['val_loss'])
plt.legend(['Train', 'Validation'], loc='upper right')
plt.ylabel('Loss')
plt.xlabel('Epoch Number')
plt.title('Model loss during training SGD')
plt.show()