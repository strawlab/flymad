/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#include <SPI.h>
#include "dac714.h"

const unsigned short AOUT_A0 = 10;
const unsigned short AOUT_A1 = 9;
const unsigned short LASER = 7;

DAC714 analogOut = DAC714(AOUT_A0,AOUT_A1);

// global vars for the state of the DACs
uint8_t velocity_mode=0;
uint16_t posA=0;
uint16_t posB=0;

uint16_t rateA=0;
uint16_t rateB=0;
int8_t signA=1;
int8_t signB=1;

unsigned long last_stampA;
unsigned long last_stampB;

#define VELOCITY_BIT 0X04
#define LASER_BIT    0X02
#define DEBUG_BIT    0X01

#define stamp_func micros
#define timescale 5

void setup() {
  // initialize the digital pin as an output.
  pinMode(LASER, OUTPUT);
  // start serial port at 9600 bps:
  Serial.begin(115200);

  SPI.setDataMode(SPI_MODE0);
  SPI.setBitOrder(MSBFIRST);
  SPI.setClockDivider(SPI_CLOCK_DIV2);
  SPI.begin();

  digitalWrite(LASER, LOW);
  analogOut.setValue_AB(posA,posB);
}

void set_stuff( uint16_t dac, uint16_t* rate, int8_t *sign ) {
	int16_t speed = (int16_t)dac; // cast uint to int

	if (speed==0) {
		*sign = 0;
		*rate = 0xFFFF;
		return;
	}

	if (speed < 0) {
		*sign = -1;
		speed = -speed;
	} else {
		*sign = 1;
	}

	float num = 65536.0f;
	float ratef = num/(float)speed;

	*rate = ratef; //cast float to uint
}

void loop() {

  if (velocity_mode) {
    unsigned long cur_stamp = stamp_func();
    unsigned long dtA = (cur_stamp-last_stampA)>>timescale;
    unsigned long dtB = (cur_stamp-last_stampB)>>timescale;

	uint8_t newvals = 0;
    if (dtA>0) {
		int16_t stepsA = (int16_t)(dtA/rateA);

		if (stepsA != 0) {
			posA += signA*stepsA;
			newvals = 1;
			last_stampA += (stepsA*rateA)<<timescale;
		}
    }
	if (dtB>0) {
		int16_t stepsB = (int16_t)(dtB/rateB);
		if (stepsB != 0) {
			posB += signB*stepsB;
			newvals = 1;
			last_stampB += (stepsB*rateB)<<timescale;
		}
	}

	if (newvals) {
		analogOut.setValue_AB(posA,posB);
	}

  }

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
		if (cmd & VELOCITY_BIT) {
                  velocity_mode=1;
				  set_stuff( dacA, &rateA, &signA );
				  set_stuff( dacB, &rateB, &signB );

				  unsigned long cur_stamp = stamp_func();
				  last_stampA = cur_stamp-rateA; // trigger update next cycle
				  last_stampB = cur_stamp-rateB;

				  Serial.write("sign, rate (A): ");
				  Serial.print(signA, DEC);
				  Serial.write(" ");
				  Serial.println(rateA, DEC);

				  Serial.write("sign, rate (B): ");
				  Serial.print(signB, DEC);
				  Serial.write(" ");
				  Serial.println(rateB, DEC);

                } else {
                  velocity_mode=0;
                  posA = dacA; posB = dacB;
                  analogOut.setValue_AB(posA,posB);
                }

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

