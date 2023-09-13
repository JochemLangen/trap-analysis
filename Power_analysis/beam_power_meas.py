# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 14:21:27 2022

@author: joche
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fontsize = 13

Excel_file = pd.read_csv("Laser-beam-power-2.csv")
# print(Excel_file)
#The arrays corresponding to the first point along the z-axis:
time = np.array(Excel_file["S121C"][3:], dtype=float)
power = np.array(Excel_file["14/11/2022 15:33"][3:], dtype=float)
data_length = len(time)

averaging_factor = 10
reshaped_length = int(data_length/averaging_factor)-1
remainder = np.remainder(data_length,averaging_factor)
used_shape = averaging_factor*reshaped_length
print("The averages are taken over {} points, though the last one may be different and is here taken over {} points.".format(averaging_factor,averaging_factor+remainder))

avg_power = np.reshape(power[:used_shape], (reshaped_length,averaging_factor))
avg_power = np.mean(avg_power, axis=0)
avg_power = np.append(avg_power, np.mean(power[used_shape:]))
avg_time = np.reshape(time[:used_shape], (reshaped_length,averaging_factor))
avg_time = np.mean(avg_time, axis=0)
# print(avg_time)
avg_time = np.append(avg_time, np.mean(time[used_shape:]))

power_steps = avg_power[1:]-avg_power[:-1]



# remainder = np.remainder(y_length,interval)

# dimension = int((y_length-remainder)/interval)-1
# used_shape = dimension*interval


# used_y = y_data[:-(interval+remainder)]
# remainder_y = y_data[used_shape:]

# reshaped_y = np.reshape(used_y, (dimension,interval))
# average_array = np.average(reshaped_y, axis=1)
# averages = np.dstack([average_array]*interval)[0]
# reshaped_y -= averages

# y_data = np.reshape(reshaped_y, used_shape)
# last_average = np.mean(remainder_y)
# y_data = np.append(y_data, remainder_y-last_average)
# average_array = np.append(average_array, last_average)



fig = plt.figure()
figure_ratio = 2
colours = ['red', 'blue', 'green']
ax = fig.add_axes((0,0,1,1))
ax.plot(avg_time[1:],power_steps)

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
ax.plot(avg_time,avg_power)

fig = plt.figure()
ax = fig.add_axes((0,0,1,1))
ax.plot(time,power)
plt.ylabel("Power (mW)",fontsize=fontsize)
plt.xlabel("Time from start-up (s)",fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
ax.minorticks_on()

