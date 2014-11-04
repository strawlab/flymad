#include <inttypes.h>

#include <UDEV.h>

#include "TimerOne.h"
#include "TLC5620CN.h"
#include "dac714.h"

const int PIN_LED = A5;

const int PIN_LASERDAC_DATA = A4;
const int PIN_LASERDAC_CLK  = A3;
const int PIN_LASERDAC_LOAD = A2;
const int PIN_LASERDAC_LDAC = A1;

const int PIN_GYRODAC_ADDR0 = 10;
const int PIN_GYRODAC_ADDR1 =  9;

const int PIN_LASER_PWR0    =  4;
const int PIN_LASER_PWM0    =  3;
const int PIN_LASER_SENSE0  = A0;

const int PIN_LASER_PWR1    =  7;
const int PIN_LASER_PWM1    =  5;
const int PIN_LASER_SENSE1  = A6;

const int PIN_LASER_PWR2    =  8;
const int PIN_LASER_PWM2    =  6;
const int PIN_LASER_SENSE2  = A7;

class Laser {
public:
    Laser(int pin_pwr, int pin_pwm, int pin_sense, uint8_t dac_channel, TLC5620CN &dac) :
        _pin_pwr(pin_pwr), _pin_pwm(pin_pwm), _pin_sense(pin_sense),
        _dac_channel(dac_channel),
        _interval_ms(0), _tick_ms(0), _on(0), _en(0),
        _dac(dac) {}

    void begin(void) {
        _dac.begin();
        pinMode(_pin_pwr, OUTPUT);
        pinMode(_pin_pwm, OUTPUT);
        analogReference(INTERNAL);

        set_frequency(0);
        set_on(0);
        set_enable(0);
    }

    void set_intensity(uint8_t v) {
        _dac.set_channel(_dac_channel, v);
        analogWrite(_pin_pwm, v); 
    }

    void set_frequency(float f) {
        if (f > 0)
            _interval_ms = 1000.0/f; //convert to ms interval
        else
            _interval_ms = 0;
    }

    void set_on(uint8_t on) {
        _on = on > 0;
        if (_interval_ms == 0) digitalWrite(_pin_pwr, _on && _en);
    }

    void set_enable(uint8_t en) {
        _en = en > 0;
        if (_interval_ms == 0) digitalWrite(_pin_pwr, _on && _en);
    }


    uint16_t read_current(void) {
        return analogRead(_pin_sense);
    }

    void update(uint8_t en, float frequency, uint8_t intensity) {
        set_enable(en);
        set_frequency(frequency);
        set_intensity(intensity);
        _tick_ms = 0;
    }

    void tick_1ms(void) {
        if (_interval_ms > 0) {
            if (++_tick_ms >= _interval_ms) {
                digitalWrite(_pin_pwr, _on && _en && (0x01 ^ digitalRead(_pin_pwr)));
                _tick_ms = 0;
            }
        }
    }

private:
    int _pin_pwr, _pin_pwm, _pin_sense;
    uint8_t _dac_channel;
    uint16_t _interval_ms;
    uint16_t _tick_ms;
    uint8_t _on, _en;
    TLC5620CN &_dac;
};

typedef struct {
    uint16_t pos;
    unsigned long last_stamp;
    uint32_t rate;
    int8_t sign;
} GyroState_t;


    

TLC5620CN dac_laser(PIN_LASERDAC_DATA, PIN_LASERDAC_CLK, PIN_LASERDAC_LOAD, PIN_LASERDAC_LDAC);
DAC714    dac_gyro(PIN_GYRODAC_ADDR0, PIN_GYRODAC_ADDR1);

Laser     laser0(PIN_LASER_PWR0, PIN_LASER_PWM0, PIN_LASER_SENSE0, 0, dac_laser);
Laser     laser1(PIN_LASER_PWR1, PIN_LASER_PWM1, PIN_LASER_SENSE1, 1, dac_laser);
Laser     laser2(PIN_LASER_PWR2, PIN_LASER_PWM2, PIN_LASER_SENSE2, 2, dac_laser);

UDEV      udev(Serial);

bool            velocity_mode;
GyroState_t     gyroA, gyroB;
unsigned long   time;

const unsigned long COMM_HZ = (1000000/10); //100Hz, in microseconds

static char serial_read_blocking() {
    while (Serial.available() == 0) {
        delay(1);
    }
    return Serial.read();
}

void velocity_bookeeping(int32_t speed, unsigned long cur_stamp, GyroState_t &gyro) {
	if (speed == 0) {
		gyro.sign = 0;
		gyro.rate = 0xFFFF;
		return;
	}

	if (speed < 0) {
		gyro.sign = -1;
		speed = -speed;
	} else {
		gyro.sign = 1;
	}

	// magic constant found by trial and error
	float num =   1000000.0f;
	float ratef = num/(float)speed;

	gyro.rate = (uint32_t)ratef;
    gyro.last_stamp = cur_stamp - gyro.rate;
}

void lasers_tick_1ms(void) {
  laser0.tick_1ms();
  laser1.tick_1ms();
  laser2.tick_1ms();
}

void setup() {
  pinMode(PIN_LED, OUTPUT);

  //ensure lasers are off
  laser0.begin();
  laser1.begin();
  laser2.begin();

  Timer1.initialize(1000);
  Timer1.attachInterrupt(lasers_tick_1ms);

  // start serial port at 115200 bps:
  Serial.begin(115200);

  //wait for 5 seconds for an id
  udev.begin();
  udev.setup(PIN_LED);

  dac_gyro.begin();

  time = micros();
  gyroA.pos  = gyroB.pos  = 0;
  gyroA.rate = gyroB.rate = 0;
  gyroA.sign = gyroB.sign = 1;
  gyroA.last_stamp = gyroB.last_stamp = time;
  velocity_mode = false;

}


void loop() {
  uint8_t ok = 0;
  uint8_t cmd = 0;
  uint8_t val = 0;
  unsigned long cur_stamp = micros();

  int32_t cmdA, cmdB;
  uint8_t cmdC;

  if (Serial.available() > 1) {
    cmd = Serial.read();
    val = Serial.read();

    if ((cmd == 'P') && (val == '=')) {
        cmdA = Serial.parseInt();   //galvo A
        cmdB = Serial.parseInt();   //galvo B
        cmdC = Serial.parseInt();   //laser on/off (bitwise)
        if (serial_read_blocking() == '\n') {
		    gyroA.pos = (uint16_t)cmdA;
		    gyroB.pos = (uint16_t)cmdB;
            dac_gyro.setValue_AB(gyroA.pos, gyroB.pos);

            laser0.set_on(cmdC & 0x01);
            laser1.set_on(cmdC & 0x02);
            laser2.set_on(cmdC & 0x04);

            ok = 1;
	    }
    }
    else if ((cmd == 'V') && (val == '=')) {
        cmdA = Serial.parseInt();   //galvo A
        cmdB = Serial.parseInt();   //galvo B
        cmdC = Serial.parseInt();   //laser on/off (bitwise)
        if (serial_read_blocking() == '\n') {
            velocity_mode = true;
            velocity_bookeeping(cmdA, cur_stamp, gyroA);
            velocity_bookeeping(cmdB, cur_stamp, gyroB);

            laser0.set_on(cmdC & 0x01);
            laser1.set_on(cmdC & 0x02);
            laser2.set_on(cmdC & 0x04);

            ok = 1;
        }
    }
    else if ((cmd == 'L') && (val == '=')) {
        cmdA = Serial.parseInt(); //laser number
        cmdB = Serial.parseInt(); //enabled
        float f = Serial.parseFloat(); //frequency
        cmdC = Serial.parseInt(); //intensity (0-255)
        ok = serial_read_blocking() == '\n';

        if (ok) {
            if (cmdA == 0)
                laser0.update(cmdB,f,cmdC);
            else if (cmdA == 1)
                laser1.update(cmdB,f,cmdC);
            else if (cmdA == 2)
                laser2.update(cmdB,f,cmdC);
            else
                ok = 0;
        }
    }
    else if ((cmd == 'v') && (val == '?')) {
		Serial.write("v=");
		Serial.println(2, DEC);
        ok = serial_read_blocking() == '\n';
    }
    else if (cmd=='N') {
        ok = udev.process(cmd, val) != ID_FAIL_CRC;
    }
  }

  if (velocity_mode) {
    ;
  }

  if (cmd)
    digitalWrite(PIN_LED, ok);

  unsigned long dt = cur_stamp - time;
  if (cmd || (dt > COMM_HZ)) {
		Serial.write("S=");
		Serial.print(gyroA.pos, DEC);
		Serial.write(" ");
		Serial.print(gyroB.pos, DEC);
		Serial.write(" ");
        Serial.print(laser0.read_current(),DEC);
		Serial.write(" ");
        Serial.print(laser1.read_current(),DEC);
        Serial.write(" ");
        Serial.println(laser2.read_current(),DEC);
        time = cur_stamp;

  }

}

