#include <DS1307RTC.h>
#include <TimeLib.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include "Magellan_BC95.h"
#include "DFRobot_EC.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DallasTemperature.h>
Magellan_BC95 magel;

#define EC_PIN A1
#define ONE_WIRE_BUS 2
float voltage, ecValue, temperature = 100;
DFRobot_EC ec;
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);
File myFile;
//const int chipSelect = 10;
String payload;
String timeNow;
tmElements_t tm;
String testd = "5";

void setup() {
  Serial.begin(115200);
  while (!Serial) ; // wait for serial
  Serial.println(F("DataLogger Shield Test."));
  pinMode(SS, OUTPUT);

  if (!SD.begin(10, 11, 12, 13)) {
    Serial.println(F("SD Card initialization failed!"));
    return;
  }
  Serial.println(F("SD Card OK."));
  //ReadText();

  magel.begin();
  ec.begin();
  sensors.begin();
}

void loop() {
  String payloadPrintln = "";

  timeNow = Now();
  voltage = analogRead(EC_PIN) / 1024.0 * 5000; // read the voltage
  sensors.requestTemperatures();
  temperature = sensors.getTempCByIndex(0);   // read your temperature sensor to execute temperature compensation
  ecValue =  ec.readEC(voltage, temperature); // convert voltage to EC with temperature compensation
  payloadPrintln+="date: ";
  payloadPrintln+=timeNow;
  payloadPrintln+=" temperature: ";
  payloadPrintln+=temperature;
  payloadPrintln+=" EC: ";
  payloadPrintln+=ecValue;
  payloadPrintln+="ms/cm";
  ec.calibration(voltage,temperature);
  String ECValue = String(ecValue);
  String Temperature = String(temperature);
  String Timerec = String(timeNow);
  String testdd = String(timeNow);

  payload = "{\"temperature\":"+Temperature+",\"EC\":"+ECValue+",\"timerec\":"+Timerec+"}"; //,\"timerec\":"+Timerec+"
  magel.report(payload);
  WriteText(payloadPrintln);
  Serial.println(payloadPrintln);
  Serial.println("");
  for(int count = 0; count < 57 ; count++){
    delay(1000);
  }
  delay(120);
}


void ReadText() {
  // re-open the file for reading:
  myFile = SD.open("DataLogs.txt");
  if (myFile) {
    Serial.println(F("DataLogs.txt:"));

    // read from the file until there's nothing else in it:
    while (myFile.available()) {
      Serial.write(myFile.read());
    }
    // close the file:
    myFile.close();
  }
  else {
    // if the file didn't open, print an error:
    Serial.println(F("error opening DataLogs.txt"));
  }
}

void WriteText(String txt) {
  myFile = SD.open("DataLogs.txt", FILE_WRITE);
  if (myFile) {
    myFile.println(txt);
    myFile.close();
  }
  else {
    // if the file didn't open, print an error:
    Serial.println(F("error opening DataLogs.txt"));
  }
}

String Now(){
  String timeNow = "";
  if (RTC.read(tm)) {
    timeNow+=tmYearToCalendar(tm.Year);
    timeNow+=leadZero(tm.Month);
    timeNow+=leadZero(tm.Day);
    timeNow+=leadZero(tm.Hour);
    timeNow+=leadZero(tm.Minute);
    timeNow+=leadZero(tm.Second);
  } 
  else {
    timeNow = "NO";
    if (RTC.chipPresent()) {
      Serial.println("The DS1307 is stopped.  Please run the SetTime");
      Serial.println("example to initialize the time and begin running.");
      Serial.println();
    } 
    else {
      Serial.println("DS1307 read error!  Please check the circuitry.");
      Serial.println();
    }
  }
  return timeNow;
}

String leadZero(int num)
{
  char buffernum[4];
  if (num < 10) {
    sprintf(buffernum, "0%d", num);
  }
  else {
    sprintf(buffernum, "%d", num);
  }
  String bufferNum = String(buffernum);
  return bufferNum;
}

double round2Text(double value) {
   return (int)(value * 1000 + 0.5) / 1000.0;
}
