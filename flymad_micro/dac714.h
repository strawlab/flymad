// ----------------------------------------------------------------------------
// dac714.h
//
// Provides an SPI based interface to two DAC714 DACs with cascaded
// serial bus connection with synchronous update.
//
// ----------------------------------------------------------------------------
#ifndef _DAC714_H_
#define _DAC714_H_

class DAC714 {
private:
    int a0;
    int a1;
public:
    DAC714(int a0Pin, int a1Pin);
    void setValue_AB(uint16_t value_A, uint16_t value_B);
};
#endif
