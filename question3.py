import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import MiniProjectPath1 as data

brooklyn_traffic = data.dataset_1['Brooklyn Bridge']
manhattan_traffic = data.dataset_1['Manhattan Bridge']
williamsburg_traffic = data.dataset_1['Williamsburg Bridge']
queensboro_traffic = data.dataset_1['Queensboro Bridge']

total_traffic = data.dataset_1['Total']
high_temp = data.dataset_1['High Temp']
low_temp = data.dataset_1['Low Temp']
precip = data.dataset_1['Precipitation']
day = data.dataset_1['Day']

X = []
y = []

for i in range(len(total_traffic)):
    X.append([total_traffic[i], brooklyn_traffic[i], manhattan_traffic[i], williamsburg_traffic[i], queensboro_traffic[i], high_temp[i], low_temp[i], precip[i]])

for i in range(len(day)):
    if(day[i] == "Sunday"):
        y.append(0)
    elif(day[i] == "Monday"):
        y.append(1)
    elif(day[i] == "Tuesday"):
        y.append(2)
    elif(day[i] == "Wednesday"):
        y.append(3)
    elif(day[i] == "Thursday"):
        y.append(4)
    elif(day[i] == "Friday"):
        y.append(5)
    elif(day[i] == "Saturday"):
        y.append(6)

# Figure comparing total traffic, high temp, low temp, and precipitation
plt.figure(1, figsize=(20,4))
plt.subplot(1, 3, 1)
plt.scatter(total_traffic, high_temp, c = y , cmap='rainbow')
plt.title('Total Traffic vs High Temperature (°F)')
plt.xlabel('High Temp (°F)')
plt.ylabel('Bike Traffic')

plt.subplot(1, 3, 2)
plt.scatter(total_traffic, low_temp, c = y , cmap='rainbow')
plt.title('Total Traffic vs Low Temperature (°F)')
plt.xlabel('Low Temp (°F)')
plt.ylabel('Bike Traffic')

plt.subplot(1, 3, 3)
plt.scatter(total_traffic, precip, c = y , cmap='rainbow')
plt.title('Total Traffic vs Precipitation (in.)')
plt.xlabel('Precipitation (in.)')
plt.ylabel('Bike Traffic')

plt.savefig('q3-traffic-temp-precip-clusters.png')

# Figure comparing total bike traffic to each bridge's
plt.figure(2, figsize=(10,10))
plt.subplot(2, 2, 1)
plt.scatter(total_traffic, brooklyn_traffic, c = y , cmap='rainbow')
plt.title('Total Traffic vs Brooklyn Bridge Traffic')
plt.xlabel('Brooklyn Bridge Traffic')
plt.ylabel('Bike Traffic')

plt.subplot(2, 2, 2)
plt.scatter(total_traffic, manhattan_traffic, c = y , cmap='rainbow')
plt.title('Total Traffic vs Manhattan Bridge Traffic')
plt.xlabel('Manhattan Bridge Traffic')
plt.ylabel('Bike Traffic')

plt.subplot(2, 2, 3)
plt.scatter(total_traffic, williamsburg_traffic, c = y , cmap='rainbow')
plt.title('Total Traffic vs Williamsburg Bridge Traffic')
plt.xlabel('Williamsburg Bridge Traffic')
plt.ylabel('Bike Traffic')

plt.subplot(2, 2, 4)
plt.scatter(total_traffic, queensboro_traffic, c = y , cmap='rainbow')
plt.title('Total Traffic vs Queensboro Bridge Traffic')
plt.xlabel('Queensboro Bridge Traffic')
plt.ylabel('Bike Traffic')
plt.legend(labels=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])

plt.savefig('q3-traffic-bridges-clusters.png')

# Train KNN classification model and predict
y = day.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

k_vals = list(range(1, 51))
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k) 
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro')
    f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy_list.append(accuracy)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1_score)

plt.figure(3, figsize=(10,10))

plt.subplot(2, 2, 1)
plt.scatter(k_vals, accuracy_list, color='red', label='Accuracy')
plt.plot(k_vals, accuracy_list, color='red')
plt.title('Model Accuracy at Different K Values')
plt.xlabel('K Values')
plt.ylabel('Accuracy')

plt.subplot(2, 2, 2)
plt.scatter(k_vals, precision_list, color='blue', label='Precision')
plt.plot(k_vals, precision_list, color='blue')
plt.title('Model Precision at Different K Values')
plt.xlabel('K Values')
plt.ylabel('Precision')

plt.subplot(2, 2, 3)
plt.scatter(k_vals, recall_list, color='green', label='Recall')
plt.plot(k_vals, recall_list, color='green')
plt.title('Model Recall at Different K Values')
plt.xlabel('K Values')
plt.ylabel('Recall')

plt.subplot(2, 2, 4)
plt.scatter(k_vals, f1_list, color='purple', label='F1 Score')
plt.plot(k_vals, f1_list, color='purple')
plt.title('Model F1 Score at Different K Values')
plt.xlabel('K Values')
plt.ylabel('F1 Score')

plt.savefig('q3-model-k-value-metrics.png')

# Evaluate Model
knn = KNeighborsClassifier(n_neighbors=10) 
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred, average='macro')
f1_score = 2 * (precision * recall) / (precision + recall)

print('Accuracy Score:  ', accuracy)
print('Precision Score: ', precision)
print('Recall Score:    ', recall)
print('F1 Score:        ', f1_score)