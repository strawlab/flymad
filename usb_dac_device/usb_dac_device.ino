#include <LOG.h>
#include <SPI.h>
#include <mcp4822.h>
#include <Streaming.h>
#include <SerialReceiver.h>

// set pin 10 as the slave select for the digital pot:
const int slaveSelectPin = 10; // D10
const int latchPin = 9; //D9

MCP4822 analogOut = MCP4822(slaveSelectPin,latchPin);
SerialReceiver receiver = SerialReceiver();

void setup() {
    // Setup serial and SPI communications
    Serial.begin(115200);
    analogOut.begin();
    SPI.setDataMode(SPI_MODE0);
    SPI.setBitOrder(MSBFIRST);
    SPI.setClockDivider(SPI_CLOCK_DIV8);
    SPI.begin();

    // Configure analog outputs
    analogOut.setGain2X_AB();
}


void loop() {
    int a,b;
    
    while (Serial.available() > 0) {
      receiver.process(Serial.read());
      if (receiver.messageReady()) {
        a = receiver.readInt(0);
        b = receiver.readInt(1);
        analogOut.setValue_AB(a,b);
        receiver.reset();
      }
    }

}

