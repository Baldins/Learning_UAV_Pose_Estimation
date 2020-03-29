from __future__ import print_function, division

import shutil

import os
import sys
import numpy as np
import glob
import random
from PIL import Image
from tqdm import tqdm
import argparse
from bisect import bisect_right, bisect_left
import pandas as pd
from skimage import io, transform
import time
from datetime import datetime

import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#Import csv data from folders

imu_data = pd.read_csv('/home/francesca/euroc/imu0/data.csv')
camera_data = pd.read_csv('/home/francesca/euroc/cam0/data.csv')
vicon_data = pd.read_csv('/home/francesca/euroc/vicon0/data.csv')

# Drop the first row of the camera

camera_data=camera_data.drop(camera_data.index[[0]])

# re-assign name to columns

vicon_data.columns = ['timestamp','px','py','pz','qw','qx','qy','qz']
camera_data.columns = ['timestamp','filename']
imu_data.columns =['timestamp','wx','wy','wz','ax','ay','az']

# Get the timestamp of the camera and drop the index
camera_timestamps = camera_data['timestamp']
camera_timestamps = camera_timestamps.reset_index(drop=True)

### Interpolate Vicon data to get data at the same timestamp of the camera and IMU
# Vicon data

vicondata = vicon_data.copy() #create a copy

for i in range(len(camera_timestamps)):
    new_row=pd.DataFrame([[camera_timestamps[i],np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]], columns = ['timestamp','px','py','pz','qw','qx','qy','qz'])
    vicondata = pd.concat([vicondata, pd.DataFrame(new_row)], ignore_index=True)
    
vicondata.sort_values(by=['timestamp'],inplace=True, ascending=True)
vicondata = vicondata.reset_index(drop = True)
vicondata = vicondata.interpolate()

# Create a new vicon dataframe with the same timestamps of the camera

idx_vicon=return_idx(camera_timestamps,vicondata)
new_vicon = vicondata.ix[idx_vicon].reset_index(drop=True)

vicon_timestamp = new_vicon['timestamp'].values

# Create a new vicon dataframe with the same timestamps of the camera

idx_vicon=return_idx(camera_timestamps,vicondata)
new_vicon = vicondata.ix[idx_vicon].reset_index(drop=True)

vicon_timestamp = new_vicon['timestamp'].values

# Create a new IMU dataframe with the same timestamps of the camera

idx_imu = return_idx(camera_timestamps, imu_data)
new_imu = imu_data.ix[idx_imu].reset_index(drop=True)
imu_timestamp = new_imu['timestamp'].values

# Create new dataframe for images and save it

image_data = pd.merge(camera_data,new_vicon, on='timestamp').reset_index(drop=True)
image_data.to_csv(r'/home/francesca/euroc/cam0/image_data.csv',index = None, header=True)



file = pd.read_csv('/home/francesca/euroc/cam0/image_data.csv')
file.drop(file.tail(4).index,inplace=True) # drop last n rows
file.to_csv('/home/francesca/euroc/cam0/file.csv',index = None, header=True)


img_name = file.iloc[:, 1]
labels = file.iloc[:, 2:].as_matrix()


camera_timestamps = camera_timestamps.reset_index(drop=True)

c_t = np.array(camera_timestamps)
new_imu_values = np.zeros((len(c_t),10,7))
# new_imu.drop(imu_data.tail(1).index,inplace=True) # drop last n rows


# Create a new IMU dataframe with the same timestamps of the camera
# imu_matrix = imu_data.as_matrix()
for i in range(len(c_t)):
    idx_imu = return_idx_single(camera_timestamps.iloc[i], imu_data)
    new_imu_values[i][:][:] = imu_data.ix[idx_imu:idx_imu+9].reset_index(drop=True).values


new_imu_values = new_imu_values[:2144]

# Create new dataframe for images and save it

image_data = pd.merge(camera_data,new_vicon, on='timestamp').reset_index(drop=True)
image_data.to_csv(r'/home/francesca/euroc/cam0/image_data.csv',index = None, header=True)