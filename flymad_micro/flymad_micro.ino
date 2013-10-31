/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#include <SPI.h>
#include "dac714.h"

const unsigned short AOUT_A0 = 10;
const unsigned short AOUT_A1 = 9;
const unsigned short LASER = 7;

DAC714 analogOut = DAC714(AOUT_A0,AOUT_A1);

// global vars for the state of the DACs
uint8_t velocity_mode=0;
uint8_t enable_adc=0;
uint16_t posA=0;
uint16_t posB=0;
uint16_t adcVal=0;

uint32_t rateA=0;
uint32_t rateB=0;
int8_t signA=1;
int8_t signB=1;

unsigned long time;

unsigned long last_stampA;
unsigned long last_stampB;

const unsigned long COMM_HZ = (1000000/10); //100Hz, in microseconds

#define ADC_EN_BIT 	 0X08
#define VELOCITY_BIT 0X04
#define LASER_BIT    0X02
#define DEBUG_BIT    0X01

void setup() {
  // initialize the digital pin as an output.
  pinMode(LASER, OUTPUT);
  // start serial port at 115200 bps:
  Serial.begin(115200);

  SPI.setDataMode(SPI_MODE0);
  SPI.setBitOrder(MSBFIRST);
  SPI.setClockDivider(SPI_CLOCK_DIV2);
  SPI.begin();

  digitalWrite(LASER, LOW);
  analogOut.setValue_AB(posA,posB);

  analogReference(INTERNAL); //1.1V

  time = micros();
}

void set_stuff( int32_t speed, uint32_t* rate, int8_t *sign ) {

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

	// magic constant found by trial and error
	float num =   1000000.0f;

	float ratef = num/(float)speed;

	*rate = ratef; //cast float to int
}

void loop() {
  uint8_t cmd = 0;
  unsigned long cur_stamp = micros();

  if (velocity_mode) {
	unsigned long dtA = (cur_stamp-last_stampA);
	unsigned long dtB = (cur_stamp-last_stampB);

	uint8_t newvals = 0;
    if (dtA>0) {
		int32_t stepsA = (int32_t)(dtA/rateA);

		if (stepsA != 0) {
			posA += signA*stepsA;
			newvals = 1;
			last_stampA += (stepsA*rateA);
		}
    }
	if (dtB>0) {
		int32_t stepsB = (int32_t)(dtB/rateB);
		if (stepsB != 0) {
			posB += signB*stepsB;
			newvals = 1;
			last_stampB += (stepsB*rateB);
		}
	}

	if (newvals) {
		analogOut.setValue_AB(posA,posB);
	}

  }

  while (Serial.available() > 0) {
    cmd = Serial.parseInt();

    int32_t cmdA = Serial.parseInt();
    int32_t cmdB = Serial.parseInt();

    if (Serial.read() == '\n') {
		if (cmd & LASER_BIT) {
			digitalWrite(LASER, HIGH);   // set the output on
		} else {
			digitalWrite(LASER, LOW);      // set the output off
		}
		if (cmd & VELOCITY_BIT) {
			velocity_mode=1;
			set_stuff( cmdA, &rateA, &signA );
			set_stuff( cmdB, &rateB, &signB );

			unsigned long cur_stamp = micros();
			last_stampA = cur_stamp-rateA; // trigger update next cycle
			last_stampB = cur_stamp-rateB;

		} else {
			velocity_mode=0;
			posA = (uint16_t)cmdA;
			posB = (uint16_t)cmdB;
			analogOut.setValue_AB(posA,posB);
		}

		enable_adc = cmd & ADC_EN_BIT;

    }
  }

  unsigned long dt = cur_stamp - time;
  if (cmd || (dt > COMM_HZ)) {

		if (enable_adc) {
			adcVal = analogRead(A0);
		}

		Serial.print(velocity_mode, DEC);
		Serial.write(" ");
		Serial.print(posA, DEC);
		Serial.write(" ");
		Serial.print(posB, DEC);
		Serial.write(" ");
		Serial.println(adcVal, DEC);

        time = cur_stamp;
  }

}

