#include "TLC5620CN.h"

TLC5620CN::TLC5620CN(int data, int clk, int load, int ldac) :
    _pin_data(data), _pin_clk(clk),
    _pin_load(load), _pin_ldac(ldac) {
}

void TLC5620CN::begin(void) {
  pinMode(_pin_data, OUTPUT);
  pinMode(_pin_clk, OUTPUT);
  pinMode(_pin_load, OUTPUT);
  pinMode(_pin_ldac, OUTPUT);
}

void TLC5620CN::set_channel(uint8_t chan, uint8_t v) {
    uint8_t rng = 0x00; //don't double the range
    uint16_t val = 0x00;

    val = ((chan & 0x3) << 9) | ((rng & 0x01) << 8) | v;
    digitalWrite(_pin_load, HIGH);
    digitalWrite(_pin_ldac, LOW);
    TLC5620CN::shiftOut(_pin_data, _pin_clk, MSBFIRST, val, 11, 1);
    digitalWrite(_pin_load, LOW);
    delayMicroseconds(1);
    digitalWrite(_pin_ldac, HIGH);
    delayMicroseconds(1);
}

void TLC5620CN::shiftOut(uint8_t dataPin, uint8_t clockPin, uint8_t bitOrder, int val, uint8_t bits, uint8_t del) {
    uint8_t i;
    for (i = 0; i < bits; i++)  {
        if (bitOrder == LSBFIRST)
            digitalWrite(dataPin, !!(val & (1 << i)));
        else    
            digitalWrite(dataPin, !!(val & (1 << ((bits - 1 - i)))));
        digitalWrite(clockPin, HIGH);
        delayMicroseconds(del);
        digitalWrite(clockPin, LOW);            
    }
}


