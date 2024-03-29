#!/bin/python

from pyModbusTCP.server import ModbusServer, DataBank
from time import sleep
from random import uniform

# Create an instance of ModbusServer
server = ModbusServer("10.131.115.170", 1234, no_block=True)

try:
    print("Start server...")
    server.start()
    print("Server is online")
    state = [0]
    while True:
        DataBank.set_words(1, [int(uniform(0, 100))])
        DataBank.set_words(2, [int(200)])
        if state != DataBank.get_words(1):
            state = DataBank.get_words(1)
            print("Value of Register 1 has changed to " +str(state))
        sleep(0.5)


except:
    print("Shutdown server ...")
    server.stop()
    print("Server is offline")
