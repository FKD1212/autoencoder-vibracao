/*
  Firmware de referência: ESP32 + acelerômetro ADXL345 -> sensor_app.py

  Ligações (I2C):
    ADXL345 VCC -> 3V3        ADXL345 GND -> GND
    ADXL345 SDA -> GPIO 21    ADXL345 SCL -> GPIO 22
    ADXL345 CS  -> 3V3 (modo I2C)
    ADXL345 SDO -> GND (endereço 0x53)
  Fixe o sensor rígido no ponto de medição (parafuso ou base magnética; fita
  dupla-face atenua as frequências altas).

  Saída pela USB, 921600 baud, uma amostra por linha, em g:
    # fs=1600 channels=x,y,z units=g sensor=ADXL345    (repetido a cada 2 s)
    0.0123,-0.0045,0.9981
  A cadência vem do relógio do próprio ADXL345: a FIFO em modo stream guarda até
  32 amostras, então nada se perde se a serial atrasar um pouco.

  No PC:  python sensor_app.py --serial COM3   (ou escolha a porta na interface)

  Outra placa/sensor serve, desde que siga o mesmo protocolo: taxa fixa, a linha
  '# fs=...' e uma linha 'v1,v2,...' por amostra.
  Texto a 1600 Hz x 3 eixos ~ 40 kB/s: exige 921600 baud. Num Arduino Uno/Nano,
  use FS_CODE 0x0C (400 Hz), FS_HZ 400 e BAUD 230400.

  Firmware de referência — não testado em hardware nesta versão.
*/
#include <Wire.h>

const uint8_t ADXL = 0x53;
const long BAUD = 921600;
const uint8_t FS_CODE = 0x0E;      // BW_RATE: 0x0F=3200 Hz, 0x0E=1600, 0x0D=800, 0x0C=400, 0x0B=200
const int FS_HZ = 1600;            // precisa bater com FS_CODE
const float G_PER_LSB = 0.0039f;   // resolução plena: 3,9 mg/LSB em qualquer faixa

void writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADXL);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(ADXL);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL, (uint8_t)1);
  return Wire.read();
}

void sendHeader() {
  Serial.print("# fs=");
  Serial.print(FS_HZ);
  Serial.println(" channels=x,y,z units=g sensor=ADXL345");
}

void setup() {
  Serial.begin(BAUD);
  Wire.begin();            // ESP32: SDA=21, SCL=22
  Wire.setClock(400000);   // I2C rápido: ~300 us por amostra
  delay(100);

  while (readReg(0x00) != 0xE5) {   // DEVID do ADXL345
    Serial.println("# erro: ADXL345 nao encontrado - confira ligacoes e endereco");
    delay(1000);
  }
  writeReg(0x2D, 0x00);     // POWER_CTL: standby enquanto configura
  writeReg(0x31, 0x0B);     // DATA_FORMAT: resolução plena, +-16 g
  writeReg(0x2C, FS_CODE);  // BW_RATE: taxa de amostragem
  writeReg(0x38, 0x80);     // FIFO_CTL: modo stream
  writeReg(0x2D, 0x08);     // POWER_CTL: medindo
  sendHeader();
}

void loop() {
  static uint32_t lastHeader = 0;
  uint8_t entries = readReg(0x39) & 0x3F;   // FIFO_STATUS: amostras prontas

  for (uint8_t i = 0; i < entries; i++) {
    Wire.beginTransmission(ADXL);
    Wire.write(0x32);                        // DATAX0..DATAZ1
    Wire.endTransmission(false);
    Wire.requestFrom(ADXL, (uint8_t)6);
    uint8_t b[6];
    for (uint8_t j = 0; j < 6; j++) b[j] = Wire.read();
    int16_t x = (int16_t)((b[1] << 8) | b[0]);
    int16_t y = (int16_t)((b[3] << 8) | b[2]);
    int16_t z = (int16_t)((b[5] << 8) | b[4]);

    Serial.print(x * G_PER_LSB, 4);
    Serial.print(',');
    Serial.print(y * G_PER_LSB, 4);
    Serial.print(',');
    Serial.println(z * G_PER_LSB, 4);
  }

  if (millis() - lastHeader > 2000) {       // quem conectar depois também recebe fs/eixos
    sendHeader();
    lastHeader = millis();
  }
}
