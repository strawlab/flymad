# USB DAC device for FlyMAD

This is an [Arduino Uno](http://arduino.cc/en/Main/arduinoBoardUno)
device with a custom [Arduino
ProtoShield](http://www.sparkfun.com/products/7914) containing an [MCP
4822
DAC](http://www.microchip.com/stellent/idcplg?IdcService=SS_GET_PAGE&nodeId=1335&dDocName=en024016).

## principle of operation

The Arduino program accepts messages like "[123,456]" over the serial
port and converts these strings to integers and then outputs that
value to the MCP 4822 chip, which converts them to an analog
signal. The valid range of values for this 12 bit chip is 0-4095. This
gives voltages from 0 to 4.2V.

