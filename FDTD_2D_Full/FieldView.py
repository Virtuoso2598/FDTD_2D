import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

FrameRate = 5
LowLim = -25
HighLim = 25

directory = Path("")

Location_Srv = directory / "FieldSaveDat.txt"
Location_Data = directory / "Ez.txt"

# read service information from FDTD core, need for get data
service_inf = np.loadtxt(Location_Srv, delimiter=' ', dtype=np.int)
print("part size = ", service_inf[0], "quant = ", service_inf[1])

# read field data from FDTD core
#data = np.genfromtxt(Location_Data, dtype=np.float)
data = pd.read_csv(Location_Data, delim_whitespace=True, header=None)

data = data.values

#LowLim = data.min()
#HighLim = data.max()

# get size of calculated field
x_s = int(service_inf[0])
y_s = int(data.shape[1])
z_s = int(service_inf[1])

# Create 3D array for FDTD data
FrameData = np.zeros((x_s, y_s, z_s))
print("FrameData: ", FrameData.shape)

# alignment raw data into frames
i=0
while i < z_s :
    FrameData[:,:,i] = data[(x_s*i) : (x_s*(i+1)), :]
    i=i+1    
print("Cycle end")

#create grid for colormap
y_bound, x_bound = np.mgrid[slice(0, x_s+1, 1),
                            slice(0, y_s+1, 1)]

#Create figure and axis for plotting
fig  = plt.figure()
ax   = plt.subplot(111)

    #mng = plt.get_current_fig_manager()
    #mng.resize(*mng.window.maxsize())

cax = ax.pcolormesh(x_bound, y_bound, FrameData[:,:,0], vmin=LowLim, vmax=HighLim, cmap = 'jet')

# create finc fore animation
def animate(t):
    #print(t)
    ax.cla()
    cax = ax.pcolormesh(x_bound, y_bound, FrameData[:,:,t], vmin=LowLim, vmax=HighLim, cmap = 'jet')
    return cax,

ani = animation.FuncAnimation(fig, animate, frames=z_s, interval=FrameRate, repeat_delay=1000)
ax.set_aspect('equal', 'box')
plt.show()

