import numpy as np
from matplotlib import pyplot as plt
import MiniProjectPath1 as data

# Lists of bike traffic on each bridge
brooklyn_traffic = data.dataset_1['Brooklyn Bridge']
manhattan_traffic = data.dataset_1['Manhattan Bridge']
williamsburg_traffic = data.dataset_1['Williamsburg Bridge']
queensboro_traffic = data.dataset_1['Queensboro Bridge']
total_traffic = data.dataset_1['Total']

sum_traffic = [sum(brooklyn_traffic), sum(manhattan_traffic), sum(williamsburg_traffic), sum(queensboro_traffic), sum(total_traffic)]
len_data = len(brooklyn_traffic)

# Average Bike Traffic on Each Bridge and in Total
avg_brooklyn_traffic = round(sum_traffic[0] / len_data)
avg_manhattan_traffic = round(sum_traffic[1] / len_data)
avg_williamsburg_traffic = round(sum_traffic[2] / len_data)
avg_queensboro_traffic = round(sum_traffic[3] / len_data)
avg_traffic = round(sum_traffic[4] / (len_data * 4))

bridge_names = ['Brooklyn', 'Manhattan', 'Williamsburg', 'Queensboro', 'All Bridges']
avg_list = [avg_brooklyn_traffic, avg_manhattan_traffic, avg_williamsburg_traffic, avg_queensboro_traffic, avg_traffic]

# Creates Bar Graph for Average Bike Traffic on Each Bridge and in Total
plt.figure(1)
plt.title('Average Bike Traffic on Bridges in NYC from April through October')
plt.xticks(np.arange(5), bridge_names, fontsize=8)
plt.ylabel('Average Daily Bike Traffic')
plt.bar(bridge_names, avg_list, color='green')
plt.savefig('average.png')

# Prints Averages
for i in range(len(bridge_names)):
    print("Average Bike Traffic on " + bridge_names[i] + ": " + str(avg_list[i]))