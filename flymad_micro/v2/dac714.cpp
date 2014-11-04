#include "Arduino.h"
#include "SPI.h"
#include "dac714.h"

DAC714::DAC714(int a0Pin, int a1Pin) {
    // Configure chip select and latch pins
    a0 = a0Pin;
    a1 = a1Pin;
}

void DAC714::begin(void) {
    pinMode(a0,OUTPUT);
    pinMode(a1,OUTPUT);
    digitalWrite(a0,HIGH);
    digitalWrite(a1,HIGH);

    SPI.setDataMode(SPI_MODE0);
    SPI.setBitOrder(MSBFIRST);
    SPI.setClockDivider(SPI_CLOCK_DIV2);
    SPI.begin();
}

void DAC714::setValue_AB(uint16_t value_A, uint16_t value_B) {
  uint8_t byte0, byte1, byte2, byte3;

  digitalWrite(a0,LOW);

  byte0 = (value_B >> 8);
  byte1 = value_B & 0xFF;
  byte2 = (value_A >> 8);
  byte3 = value_A & 0xFF;

  SPI.transfer(byte0);
  SPI.transfer(byte1);
  SPI.transfer(byte2);
  SPI.transfer(byte3);
  digitalWrite(a0,HIGH);

  digitalWrite(a1,LOW);
  digitalWrite(a1,HIGH);
}

void DAC714::setValue_ABC(uint16_t value_A, uint16_t value_B, uint16_t value_C) {
  uint8_t byte0, byte1, byte2, byte3, byte4, byte5;

  digitalWrite(a0,LOW);

  byte0 = (value_C >> 8);
  byte1 = value_C & 0xFF;
  byte2 = (value_B >> 8);
  byte3 = value_B & 0xFF;
  byte4 = (value_A >> 8);
  byte5 = value_A & 0xFF;

  SPI.transfer(byte0);
  SPI.transfer(byte1);
  SPI.transfer(byte2);
  SPI.transfer(byte3);
  SPI.transfer(byte4);
  SPI.transfer(byte5);
  digitalWrite(a0,HIGH);

  digitalWrite(a1,LOW);
  digitalWrite(a1,HIGH);
}
