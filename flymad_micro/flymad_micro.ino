/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#include <SPI.h>
#include "dac714.h"

const unsigned short AOUT_A0 = 10;
const unsigned short AOUT_A1 = 9;
const unsigned short LASER = 7;

DAC714 analogOut = DAC714(AOUT_A0,AOUT_A1);
char inByte = 0;         // incoming serial byte
uint8_t state=0;

#define LASER_BIT 0X02
#define DEBUG_BIT 0X01

void setup() {
  // initialize the digital pin as an output.
  pinMode(LASER, OUTPUT);
  // start serial port at 9600 bps:
  Serial.begin(115200);

  SPI.setDataMode(SPI_MODE0);
  SPI.setBitOrder(MSBFIRST);
  SPI.setClockDivider(SPI_CLOCK_DIV8);
  SPI.begin();
}

void loop() {
  while (Serial.available() > 0) {
    uint8_t cmd = Serial.parseInt();
    uint16_t dacA = Serial.parseInt();
    uint16_t dacB = Serial.parseInt();

    if (Serial.read() == '\n') {
		if (cmd & LASER_BIT) {
			digitalWrite(LASER, HIGH);   // set the output on
		} else {
			digitalWrite(LASER, LOW);      // set the output off
		}
		analogOut.setValue_AB(dacA,dacB);

  
      if (cmd & DEBUG_BIT) {
        // print the three numbers in one string as hexadecimal:
        Serial.print(cmd, HEX);
        Serial.write(" ");
        Serial.print(dacA, HEX);
        Serial.write(" ");

        Serial.println(dacB, HEX);
      }
    }
  }
}

