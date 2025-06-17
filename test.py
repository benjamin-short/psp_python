#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 18:12:59 2025

@author: besh2109
"""

from sklearn.linear_model import LinearRegression
import numpy as np
import psp_regions as pr
import matplotlib.pyplot as plt

q_dur = np.array([])
mean_q = np.array([])
med_q = np.array([])
q_v = np.array([])
nq_v = np.array([])
for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    q_durations, mean_q_vel, med_q_vel, q_vel, nq_vel = pr.footpoint_velocity(enc=i,return_nums=True,plot=False)
    q_dur = np.append(q_dur,q_durations)
    mean_q = np.append(mean_q,mean_q_vel)
    med_q = np.append(med_q,med_q_vel)
    q_v = np.append(q_v,q_vel)
    nq_v = np.append(nq_v,nq_vel)

# Assuming q_dur, med_q, and mean_q are 1D arrays
# Filter out NaN values from med_q and corresponding entries in q_dur and mean_q
mask = ~np.isnan(med_q)  # Boolean mask: True where med_q is not NaN
q_dur_clean = q_dur[mask]  # Keep only non-NaN entries
med_q_clean = med_q[mask]
mean_q_clean = mean_q[mask]

# Reshape q_dur_clean to 2D for sklearn
q_dur_reshaped = q_dur_clean.reshape(-1, 1)  # Shape: (n_samples, 1)

# Fit the model
model = LinearRegression()
model.fit(q_dur_reshaped, med_q_clean)

# Get slope and intercept
m, b = model.coef_[0], model.intercept_

# Predict values for the line
y_pred = model.predict(q_dur_reshaped)  # Predicted values

# Create the plot
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)

# ax.scatter(q_dur_clean, mean_q_clean, color='orange', label='Mean Footpoint Velocity')
ax.scatter(q_dur_clean, med_q_clean, color='tab:blue', label='Median Footpoint Velocity')
# ax.plot(q_dur_clean, y_pred, color='red', label='Fitted Line')

ax.set_ylabel('PSP Footpoint Velocity (Mm/s)', fontsize=15)
ax.set_xlabel('Quiescent Region Durations', fontsize=15)
ax.set_title('Footpoint Velocity vs Quiescent Regions Duration', fontsize=18)

ax.set_ylim(-0.003, 0.06)
ax.set_xlim(-1000,20000)
plt.legend()
plt.show()
