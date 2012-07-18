EESchema Schematic File Version 2  date Wed 18 Jul 2012 11:34:01 AM CEST
LIBS:myshield
LIBS:arduino_shieldsNCL
LIBS:power
LIBS:device
LIBS:transistors
LIBS:conn
LIBS:linear
LIBS:regul
LIBS:74xx
LIBS:cmos4000
LIBS:adc-dac
LIBS:memory
LIBS:xilinx
LIBS:special
LIBS:microcontrollers
LIBS:dsp
LIBS:microchip
LIBS:analog_switches
LIBS:motorola
LIBS:texas
LIBS:intel
LIBS:audio
LIBS:interface
LIBS:digital-audio
LIBS:philips
LIBS:display
LIBS:cypress
LIBS:siliconi
LIBS:opto
LIBS:atmel
LIBS:contrib
LIBS:valves
LIBS:shield-cache
EELAYER 25  0
EELAYER END
$Descr A4 11700 8267
encoding utf-8
Sheet 1 1
Title ""
Date "18 jul 2012"
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
Wire Wire Line
	7200 4150 7250 4150
Wire Wire Line
	7250 3850 6900 3850
Wire Wire Line
	6900 3850 6900 4900
Wire Wire Line
	6900 4900 8400 4900
Connection ~ 10100 3350
Wire Wire Line
	10100 3700 10100 3350
Connection ~ 9750 2050
Connection ~ 9750 3350
Wire Wire Line
	9750 2050 9750 3350
Connection ~ 9650 2550
Wire Wire Line
	8900 5000 8900 4650
Wire Wire Line
	8900 4650 10700 4650
Wire Wire Line
	10700 4650 10700 2550
Wire Wire Line
	10700 2550 9450 2550
Wire Wire Line
	8400 4900 8400 5000
Wire Wire Line
	8100 5000 6850 5000
Wire Wire Line
	5400 5150 5250 5150
Connection ~ 6350 2450
Wire Wire Line
	3550 4050 3550 3450
Wire Wire Line
	3550 3450 5700 3450
Wire Wire Line
	5700 3450 5700 2450
Wire Wire Line
	5700 2450 7250 2450
Connection ~ 6400 2550
Wire Wire Line
	5250 4250 6400 4250
Wire Wire Line
	6400 4250 6400 1250
Connection ~ 8900 2550
Wire Wire Line
	8900 1250 8900 2950
Wire Wire Line
	6950 2950 7250 2950
Wire Wire Line
	10150 3350 7250 3350
Wire Wire Line
	7250 3350 7250 3150
Wire Wire Line
	6950 1650 7250 1650
Connection ~ 9050 1150
Wire Wire Line
	9050 2450 9050 800 
Wire Wire Line
	9050 2450 8800 2450
Wire Wire Line
	8800 3150 10000 3150
Wire Wire Line
	8800 1850 10000 1850
Wire Wire Line
	7250 1150 6350 1150
Wire Wire Line
	6450 2650 7250 2650
Wire Wire Line
	6400 2550 7250 2550
Wire Wire Line
	7250 1550 7050 1550
Wire Wire Line
	7050 1550 7050 2750
Wire Wire Line
	7050 2750 7250 2750
Wire Wire Line
	6400 1250 7250 1250
Wire Wire Line
	7250 1350 6450 1350
Wire Wire Line
	7250 1450 6500 1450
Wire Wire Line
	6350 1150 6350 2450
Wire Wire Line
	7250 1850 7250 2050
Wire Wire Line
	7250 2050 10150 2050
Wire Wire Line
	8900 1250 8800 1250
Wire Wire Line
	8900 2550 8800 2550
Wire Wire Line
	9050 800  3300 800 
Wire Wire Line
	9050 1150 8800 1150
Wire Wire Line
	7250 3050 7200 3050
Wire Wire Line
	7200 3050 7200 1750
Wire Wire Line
	7200 1750 7250 1750
Connection ~ 7200 2150
Wire Wire Line
	9650 2550 9650 2600
Wire Wire Line
	6450 1350 6450 4350
Wire Wire Line
	6450 4350 5250 4350
Connection ~ 6450 2650
Wire Wire Line
	6500 1450 6500 4150
Wire Wire Line
	6500 4150 5250 4150
Wire Wire Line
	3300 800  3300 5150
Wire Wire Line
	3300 5150 3550 5150
Wire Wire Line
	8200 5000 8300 5000
Wire Wire Line
	7200 2150 9900 2150
Wire Wire Line
	9900 2150 9900 2450
Wire Wire Line
	9900 2450 11050 2450
Wire Wire Line
	11050 2450 11050 4850
Wire Wire Line
	11050 4850 9000 4850
Wire Wire Line
	9000 4850 9000 5000
Connection ~ 9450 2150
Wire Wire Line
	8900 2950 10450 2950
Wire Wire Line
	10450 2950 10450 4400
Wire Wire Line
	10450 4400 8800 4400
Wire Wire Line
	8800 4400 8800 5000
Connection ~ 9450 2950
Wire Wire Line
	10100 3900 10200 3900
Wire Wire Line
	6850 5000 6850 3750
Wire Wire Line
	6850 3750 7250 3750
Wire Wire Line
	5250 4550 7100 4550
Wire Wire Line
	7100 4550 7100 4050
Wire Wire Line
	7100 4050 7250 4050
$Comp
L GND #PWR01
U 1 1 500682D3
P 7200 4150
F 0 "#PWR01" H 7200 4150 30  0001 C CNN
F 1 "GND" H 7200 4080 30  0001 C CNN
	1    7200 4150
	0    1    1    0   
$EndComp
$Comp
L CONN_2 P4
U 1 1 50065BB0
P 9750 3800
F 0 "P4" V 9700 3800 40  0000 C CNN
F 1 "CONN_2" V 9800 3800 40  0000 C CNN
	1    9750 3800
	-1   0    0    1   
$EndComp
Text Notes 9750 4100 0    60   ~ 0
GND
Text Notes 9750 3600 0    60   ~ 0
AGND\n
$Comp
L GND #PWR02
U 1 1 500657E8
P 10200 3900
F 0 "#PWR02" H 10200 3900 30  0001 C CNN
F 1 "GND" H 10200 3830 30  0001 C CNN
	1    10200 3900
	0    -1   -1   0   
$EndComp
NoConn ~ 8700 5000
NoConn ~ 8600 5000
NoConn ~ 8500 5000
$Comp
L CONN_10 P3
U 1 1 5005C3C5
P 8550 5350
F 0 "P3" V 8500 5350 60  0000 C CNN
F 1 "CONN_10" V 8600 5350 60  0000 C CNN
	1    8550 5350
	0    1    1    0   
$EndComp
$Comp
L GND #PWR03
U 1 1 5005B4D1
P 5400 5150
F 0 "#PWR03" H 5400 5150 30  0001 C CNN
F 1 "GND" H 5400 5080 30  0001 C CNN
	1    5400 5150
	0    -1   -1   0   
$EndComp
NoConn ~ 3550 4150
NoConn ~ 3550 4250
NoConn ~ 3550 4350
NoConn ~ 3550 4450
NoConn ~ 3550 4550
NoConn ~ 3550 4650
NoConn ~ 3550 4750
NoConn ~ 3550 4850
NoConn ~ 3550 4950
NoConn ~ 3550 5050
NoConn ~ 3550 5250
NoConn ~ 3550 5350
NoConn ~ 3550 5450
NoConn ~ 5250 4050
NoConn ~ 5250 4450
NoConn ~ 5250 4650
NoConn ~ 5250 4750
NoConn ~ 5250 4850
NoConn ~ 5250 4950
NoConn ~ 5250 5050
NoConn ~ 5250 5250
NoConn ~ 5250 5350
NoConn ~ 5250 5450
$Comp
L ARDUINO_NANO U1
U 1 1 5005B378
P 4400 4700
F 0 "U1" H 3950 5650 60  0000 C CNN
F 1 "ARDUINO_NANO" H 4400 3850 60  0000 C CNN
	1    4400 4700
	-1   0    0    1   
$EndComp
$Comp
L GND #PWR04
U 1 1 50059A79
P 6950 2950
F 0 "#PWR04" H 6950 2950 30  0001 C CNN
F 1 "GND" H 6950 2880 30  0001 C CNN
	1    6950 2950
	0    1    1    0   
$EndComp
$Comp
L GND #PWR05
U 1 1 50059A01
P 9650 2600
F 0 "#PWR05" H 9650 2600 30  0001 C CNN
F 1 "GND" H 9650 2530 30  0001 C CNN
	1    9650 2600
	1    0    0    -1  
$EndComp
$Comp
L CP1 C2
U 1 1 500593DA
P 9450 2750
F 0 "C2" H 9500 2850 50  0000 L CNN
F 1 "CP1" H 9500 2650 50  0000 L CNN
	1    9450 2750
	1    0    0    -1  
$EndComp
$Comp
L CP1 C1
U 1 1 50059383
P 9450 2350
F 0 "C1" H 9500 2450 50  0000 L CNN
F 1 "CP1" H 9500 2250 50  0000 L CNN
	1    9450 2350
	1    0    0    -1  
$EndComp
NoConn ~ 7250 2850
$Comp
L DAC714 DAC2
U 1 1 50056ECA
P 7950 2850
F 0 "DAC2" H 7700 3400 60  0000 C CNN
F 1 "DAC714" H 8000 2400 60  0000 C CNN
	1    7950 2850
	1    0    0    -1  
$EndComp
NoConn ~ 8800 1550
NoConn ~ 8800 1650
NoConn ~ 8800 1450
NoConn ~ 8800 1350
NoConn ~ 8800 1750
NoConn ~ 8800 2650
NoConn ~ 8800 2750
NoConn ~ 8800 2850
NoConn ~ 8800 2950
NoConn ~ 8800 3050
$Comp
L GND #PWR06
U 1 1 500591C2
P 6950 1650
F 0 "#PWR06" H 6950 1650 30  0001 C CNN
F 1 "GND" H 6950 1580 30  0001 C CNN
	1    6950 1650
	0    1    1    0   
$EndComp
$Comp
L BNC P2
U 1 1 50057289
P 10150 3150
F 0 "P2" H 10160 3270 60  0000 C CNN
F 1 "BNC" V 10260 3090 40  0000 C CNN
	1    10150 3150
	1    0    0    -1  
$EndComp
$Comp
L BNC P1
U 1 1 5005727C
P 10150 1850
F 0 "P1" H 10160 1970 60  0000 C CNN
F 1 "BNC" V 10260 1790 40  0000 C CNN
	1    10150 1850
	1    0    0    -1  
$EndComp
$Comp
L SSRELAY RELAY1
U 1 1 50057034
P 7600 3600
F 0 "RELAY1" H 7700 3550 60  0000 C CNN
F 1 "SSRELAY" H 7750 2950 60  0000 C CNN
	1    7600 3600
	1    0    0    -1  
$EndComp
$Comp
L DAC714 DAC1
U 1 1 50056EAB
P 7950 1550
F 0 "DAC1" H 7700 2100 60  0000 C CNN
F 1 "DAC714" H 8000 1100 60  0000 C CNN
	1    7950 1550
	1    0    0    -1  
$EndComp
$EndSCHEMATC
