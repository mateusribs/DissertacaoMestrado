# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 00:38:27 2020

@author: Mateus Ribeiro
"""
import serial
import time

arduinoData = serial.Serial('COM5', 115200)
time.sleep(1)
while(True):
    while(arduinoData.inWaiting()==0):
        pass
    data = arduinoData.readline()
    data = str(data, 'utf-8')
    splitdata = data.split(',')
    phi = float(splitdata[0])
    theta= float(splitdata[1])
    print("Phi = ", phi, "Theta = ", theta)