import numpy as np
from matplotlib import pyplot as plt
import MiniProjectPath1 as data

# Lists of bike traffic on each bridge
brooklyn_traffic = data.dataset_1['Brooklyn Bridge']
manhattan_traffic = data.dataset_1['Manhattan Bridge']
williamsburg_traffic = data.dataset_1['Williamsburg Bridge']
queensboro_traffic = data.dataset_1['Queensboro Bridge']
total_traffic = data.dataset_1['Total']

# Month names and monthly traffic across all four bridges
month_names = ['Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']

brooklyn_month = [round(sum(brooklyn_traffic[0:30]) / len(brooklyn_traffic[0:30])),
                  round(sum(brooklyn_traffic[30:61]) / len(brooklyn_traffic[30:61])),
                  round(sum(brooklyn_traffic[61:91]) / len(brooklyn_traffic[61:91])),
                  round(sum(brooklyn_traffic[91:122]) / len(brooklyn_traffic[91:122])),
                  round(sum(brooklyn_traffic[122:153]) / len(brooklyn_traffic[122:153])),
                  round(sum(brooklyn_traffic[153:183]) / len(brooklyn_traffic[153:183])),
                  round(sum(brooklyn_traffic[183:214]) / len(brooklyn_traffic[183:214]))]

manhattan_month = [round(sum(manhattan_traffic[0:30]) / len(manhattan_traffic[0:30])),
                  round(sum(manhattan_traffic[30:61]) / len(manhattan_traffic[30:61])),
                  round(sum(manhattan_traffic[61:91]) / len(manhattan_traffic[61:91])),
                  round(sum(manhattan_traffic[91:122]) / len(manhattan_traffic[91:122])),
                  round(sum(manhattan_traffic[122:153]) / len(manhattan_traffic[122:153])),
                  round(sum(manhattan_traffic[153:183]) / len(manhattan_traffic[153:183])),
                  round(sum(manhattan_traffic[183:214]) / len(manhattan_traffic[183:214]))]

williamsburg_month = [round(sum(williamsburg_traffic[0:30]) / len(williamsburg_traffic[0:30])),
                  round(sum(williamsburg_traffic[30:61]) / len(williamsburg_traffic[30:61])),
                  round(sum(williamsburg_traffic[61:91]) / len(williamsburg_traffic[61:91])),
                  round(sum(williamsburg_traffic[91:122]) / len(williamsburg_traffic[91:122])),
                  round(sum(williamsburg_traffic[122:153]) / len(williamsburg_traffic[122:153])),
                  round(sum(williamsburg_traffic[153:183]) / len(williamsburg_traffic[153:183])),
                  round(sum(williamsburg_traffic[183:214]) / len(williamsburg_traffic[183:214]))]

queensboro_month = [round(sum(queensboro_traffic[0:30]) / len(queensboro_traffic[0:30])),
                  round(sum(queensboro_traffic[30:61]) / len(queensboro_traffic[30:61])),
                  round(sum(queensboro_traffic[61:91]) / len(queensboro_traffic[61:91])),
                  round(sum(queensboro_traffic[91:122]) / len(queensboro_traffic[91:122])),
                  round(sum(queensboro_traffic[122:153]) / len(queensboro_traffic[122:153])),
                  round(sum(queensboro_traffic[153:183]) / len(queensboro_traffic[153:183])),
                  round(sum(queensboro_traffic[183:214]) / len(queensboro_traffic[183:214]))]

# Plot figure
plt.figure(1)
plt.title('Bike Traffic on Bridges in New York City from April through October')
plt.xticks(np.arange(7), month_names, fontsize=8)
plt.xlabel('Month')
plt.ylabel('Average Daily Bike Traffic')

# Brooklyn Bridge
plt.scatter(month_names, brooklyn_month, color='red', label='Brooklyn')
plt.plot(month_names, brooklyn_month, color='red')

# Manhattan Bridge
plt.scatter(month_names, manhattan_month, color='blue', label='Manhattan')
plt.plot(month_names, manhattan_month, color='blue')

# Williamsburg Bridge
plt.scatter(month_names, williamsburg_month, color='green', label='Williamsburg')
plt.plot(month_names, williamsburg_month, color='green')

# Queensboro Bridge
plt.scatter(month_names, queensboro_month, color='purple', label='Queensboro')
plt.plot(month_names, queensboro_month, color='purple')

# Create legend and save
plt.legend(fontsize=8, loc='best')
plt.savefig('q1_bike_traffic_months.png')