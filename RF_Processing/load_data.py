### Produced by Carolyn J. Swinney and John C. Woods for use with the 'DroneDetect' dataset ###

### MIN + 11 + 00 + 00 = MIN_1100_00.dat
# [0] MIN - DJI Mavic Mini
#     AIR - DJI Mavic 2 Air S
#     DIS - Parrot Disco
#     INS - DJI Inspire 2
#     MP1 - DJI Mavic Pro
#     MP2 - DJI Mavic Pro 2
#     PHA - DJI Phantom 4
# [1] 00 - ClEAN
#     01 - BLUE
#     10 - WIFI
#     11 - BOTH
# [2] 00 - ON
#     01 - HO
#     02 - FLY
# [3] IMAGE NUMBER
import os
import numpy as np
from numpy import sum,isrealobj,sqrt
from numpy.random import standard_normal

file = r"C:\Users\DiepHM\Documents\AI\RF_Processing\CLEAN\MP1_ON\MA1_0000_00.dat"                            # path to file location
f = open(file, "rb")                                        # open file
data = np.fromfile(f, dtype="float32",count=240000000)      # read the data into numpy array
data = data.astype(np.float32).view(np.complex64)           # view as complex
data_norm = (data-np.mean(data))/(np.sqrt(np.var(data)))    # normalise
newarr = np.array_split(data_norm, 100)	                    # split the array, 100 will equate to a sample length of 20ms


i=0	      						    # initialise counter
while i < 100:					            # loop through each split
    e = newarr[i]
    print(e)
     
# inside this loop you can save new smaller files or produce graphs

    i=i+1


