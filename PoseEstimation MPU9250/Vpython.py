# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 01:16:04 2020

@author: Mateus Ribeiro
"""
from vpython import *
import time
import math
import numpy as np
import serial

arduino = serial.Serial('COM5', 115200)
time.sleep(1)


scene.range = 5
d2r = np.pi/180
r2d = 1/d2r
scene.forward = vector(-1,-1,-1)
scene.width = 600
scene.heigth = 600

x = box(opacity=.5)
xArrow = arrow(length=2, shaftwidth=.1, axis=vector(1,0,0), color=color.red)
yArrow = arrow(length=2, shaftwidth=.1, axis=vector(0,1,0), color=color.green)
zArrow = arrow(length=2, shaftwidth=.1, axis=vector(0,0,1), color=color.blue)

frontArrow = arrow(length=2.5, shaftwidth=.1, color=color.purple, axis=vector(1,0,0))
upArrow = arrow(length=2, shaftwidth=.1, color=color.magenta, axis=vector(0,1,0))
sideArrow = arrow(length=2, shaftwidth=.1, color=color.orange, axis=vector(0,0,1))

while True:
    while (arduino.inWaiting()==0):
        pass
    data = arduino.readline()
    data = str(data, 'utf-8')
    splitdata = data.split(',')
    roll = float(splitdata[0])*d2r
    pitch = float(splitdata[1])*d2r
    yaw = 0*d2r
    
    rate(50)
    k = vector(np.cos(yaw)*np.cos(pitch), np.sin(pitch), np.sin(yaw)*np.cos(pitch))
    y = vector(0,1,0)
    s = cross(k, y)
    v = cross(s, k)
    v_rotated = v*np.cos(roll)+cross(k,v)*np.sin(roll)

    frontArrow.axis = k
    sideArrow.axis = s
    upArrow.axis = v_rotated

    frontArrow.length = 2.5
    sideArrow.length = 2
    upArrow.length = 2

    x.axis = k
    x.up = v_rotated




        