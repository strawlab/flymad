#include "Arduino.h"
#include "SPI.h"
#include "dac714.h"

DAC714::DAC714(int a0Pin, int a1Pin) {
    // Configure chip select and latch pins
    a0 = a0Pin;
    a1 = a1Pin;
    pinMode(a0,OUTPUT);
    pinMode(a1,OUTPUT);
    digitalWrite(a0,HIGH);
    digitalWrite(a1,HIGH);
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
