import numpy as np
from matplotlib import pyplot as plt
import MiniProjectPath1 as data

total_traffic = data.dataset_1['Total']
high_temp = data.dataset_1['High Temp']
low_temp = data.dataset_1['Low Temp']
precip = data.dataset_1['Precipitation']

def normalize(data):
    norm_data = []
    avg = sum(data) / len(data)
    stdev = np.std(np.array(data))

    for i in range(len(data)):
        norm_data.append((data[i] - avg) / stdev)

    return norm_data

high_temp_norm = normalize(high_temp)
low_temp_norm = normalize(low_temp)
precipitation_norm = normalize(precip)

temp_list = []
for i in range(len(total_traffic)):
    temp_list.append(1)

all_list = [high_temp_norm, low_temp_norm, precipitation_norm, temp_list]
x_trans = np.array(all_list)
x_orig = np.transpose(x_trans)
traffic_list = np.array(total_traffic)
inverted = np.linalg.inv(np.matmul(x_trans, x_orig))
beta = np.matmul((np.matmul(inverted, x_trans)), np.transpose(traffic_list))
print(f"beta: {beta}")

predicted_traffic = np.matmul(x_orig, np.transpose(beta)).tolist()

MSE = 0
var = 0

for k in range(len(predicted_traffic)):
    MSE += (total_traffic[k] - predicted_traffic[k]) ** 2
    var += (total_traffic[k] - (sum(total_traffic) / len(total_traffic))) ** 2

r_squared = 1 - (MSE / var)

print(r_squared)

plt.figure(1)
plt.title('Total Daily Bike Traffic vs. High Temp (°F)')
plt.xlabel('High Temp (°F)')
plt.ylabel('Total Daily Bike Traffic')
plt.scatter(high_temp, total_traffic, color='green')
plt.scatter(high_temp, predicted_traffic, color='blue')
plt.legend(labels=['Real Traffic', 'Predicted Traffic'])
plt.savefig('q2-scatter-high-temp.png')

plt.figure(2)
plt.title('Total Daily Bike Traffic vs. Low Temp (°F)')
plt.xlabel('Low Temp (°F)')
plt.ylabel('Total Bike Traffic')
plt.scatter(low_temp, total_traffic, color='green')
plt.scatter(low_temp, predicted_traffic, color='blue')
plt.legend(labels=['Real Traffic', 'Predicted Traffic'])
plt.savefig('q2-scatter-low-temp.png')

plt.figure(3)
plt.title('Total Daily Bike Traffic vs. Precipitation')
plt.xlabel('Precipitation')
plt.ylabel('Total Bike Traffic')
plt.scatter(precip, total_traffic, color='green')
plt.scatter(precip, predicted_traffic, color='blue')
plt.legend(labels=['Real Traffic', 'Predicted Traffic'])
plt.savefig('q2-scatter-precipitation.png')