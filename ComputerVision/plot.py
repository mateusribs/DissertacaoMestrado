from threading import Thread
import serial
import time
import collections
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as numpy
from tkinter import *

isReceive = False
isRun = True
value = 0.0

def getData():
    time.sleep(1.0)
    serialConnection.reset_input_buffer()

    while(isRun):
        global isReceive
        global value
        value = float(serialConnection.readline().strip())
        isReceive = True

def plotData(self, Samples, serialConnection, lines, lineValueText, lineLabel):
    data.append(value)
    lines.set_data(range(Samples), data)
    lineValueText.set_text(lineLabel + '=' + str(round(value, 2)))


serialPort = 'COM3'
baudRate = 9600

try:
    serialConnection = serial.Serial(serialPort, baudRate)
except:
    print('Cannot connect to the port')


Samples = 100
data = collections.deque([0]*Samples, maxlen=Samples)
sampleTime = 100

xmin = 0
xmax = Samples
ymin = 0
ymax = 30

fig = plt.figure(figsize=(13,6))
ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
plt.title('Real-time Sensor Reading')
ax.set_xlabel('Samples')
ax.set_ylabel('Attitude')

lineLabel = 'Degrees'
lines = ax.plot([], [], label = lineLabel)[0]
lineValueText = ax.text(0.85, 0.95, '', transform = ax.transAxes)

thread = Thread(target=getData)
thread.start()

while isReceive != True:
    print('Starting receive data')
    time.sleep(0.1)

anim = animation.FuncAnimation(fig, plotData, fargs=(Samples, serialConnection, lines, lineValueText, lineLabel), interval = sampleTime)
plt.show()

isRun = False
thread.join()
serialConnection.close()