/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
#include <SPI.h>
#include "dac714.h"

const unsigned short AOUT_A0 	= 10;
const unsigned short AOUT_A1 	= 9;
const unsigned short LASER 		= 7;
const unsigned short PIN_PWM 	= 3;

DAC714 analogOut = DAC714(AOUT_A0,AOUT_A1);

// global vars for the state of the DACs
uint16_t posA=0;
uint16_t posB=0;
uint16_t adcVal=0;

#define STATE_INITIALIZED 			(0x1)
#define IS_INITIALIZED(_v) 			(_v & STATE_INITIALIZED)
#define STATE_ADC_ENABLED			(0x2)
#define IS_ADC_ENABLED(_v) 			(_v & STATE_ADC_ENABLED)
#define STATE_VELOCITY_MODE			(0x4)
#define IS_VELOCITY_MODE(_v)		(_v & STATE_VELOCITY_MODE)
#define STATE_LASER_MODULATABLE		(0x8)
#define IS_LASER_MODULATABLE(_v)	(_v & STATE_LASER_MODULATABLE)
uint32_t state=0;

// for tracking velocity mode
uint32_t rateA=0;
uint32_t rateB=0;
int8_t signA=1;
int8_t signB=1;

unsigned long time;

unsigned long last_stampA;
unsigned long last_stampB;

const unsigned long COMM_HZ = (1000000/10); //100Hz, in microseconds

#define VELOCITY_BIT 0X02
#define SETUP_BIT    0X01

#define SETUP_ENABLE_ADC			0x01
#define SETUP_LASER_MODULATABLE		0x02

void setup() {
  state = 0;

  // initialize the digital pin as an output.
  pinMode(LASER, OUTPUT);
  // start serial port at 115200 bps:
  Serial.begin(115200);

  pinMode(PIN_PWM, OUTPUT);

  SPI.setDataMode(SPI_MODE0);
  SPI.setBitOrder(MSBFIRST);
  SPI.setClockDivider(SPI_CLOCK_DIV2);
  SPI.begin();

  // currently, at statup this is always true, but I can imagine loading
  // this from EEPROM if the laser takes especially long to turn on/off
  if ( IS_LASER_MODULATABLE(state) ) {
	;
  } else {
    digitalWrite(LASER, LOW);
  }

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

  if ( IS_VELOCITY_MODE(state) ) {
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
    uint8_t cmdC = Serial.parseInt();

    if (Serial.read() == '\n') {
		if (cmd == SETUP_BIT) {
			state |= STATE_INITIALIZED;

			if (cmdA & SETUP_ENABLE_ADC)
				state |= STATE_ADC_ENABLED;

			if (cmdA & SETUP_LASER_MODULATABLE)
				state |= STATE_LASER_MODULATABLE;

			posA = posB = adcVal = 0;

		} else {
			if (cmd & VELOCITY_BIT) {
				state |= STATE_VELOCITY_MODE;
				set_stuff( cmdA, &rateA, &signA );
				set_stuff( cmdB, &rateB, &signB );

				unsigned long cur_stamp = micros();
				last_stampA = cur_stamp-rateA; // trigger update next cycle
				last_stampB = cur_stamp-rateB;

			} else {
				state &= (~STATE_VELOCITY_MODE);
				posA = (uint16_t)cmdA;
				posB = (uint16_t)cmdB;
				analogOut.setValue_AB(posA,posB);
			}
		}

		if ( IS_LASER_MODULATABLE(state) ) {
			analogWrite(PIN_PWM, cmdC);
		} else {
			digitalWrite(LASER, cmdC > 0);
		}
    }
  }

  unsigned long dt = cur_stamp - time;
  if (cmd || (dt > COMM_HZ)) {

		if ( IS_ADC_ENABLED(state) ) {
			adcVal = analogRead(A0);
		}

		Serial.print(state, DEC);
		Serial.write(" ");
		Serial.print(posA, DEC);
		Serial.write(" ");
		Serial.print(posB, DEC);
		Serial.write(" ");
		Serial.println(adcVal, DEC);

        time = cur_stamp;
  }

}

