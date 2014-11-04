#include <Arduino.h>
#include <inttypes.h>

class TLC5620CN {
public:
    TLC5620CN(int data, int clk, int load, int ldac);
    void begin(void);
    void set_channel(uint8_t chan, float v) { set_channel(chan, (uint8_t)(v*255.0) ); }
    void set_channel(uint8_t chan, uint8_t v);
private:
    int _pin_data;
    int _pin_clk;
    int _pin_load;
    int _pin_ldac;
    void shiftOut(uint8_t dataPin, uint8_t clockPin, uint8_t bitOrder, int val, uint8_t bits = 8, uint8_t del = 0);
};


