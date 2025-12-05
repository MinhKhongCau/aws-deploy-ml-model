# Lab 2 – IoT Project (LAB2)

This folder (`LAB2`) contains the materials, code, and documentation for **Lab 2** of the IoT course at PTIT.

---

## 📋 Contents

```
LAB2/
├── src/ # Source code (Arduino, ESP, MicroPython, etc.)
├── docs/ # Documentation, diagrams, datasheets, etc.
├── wiring/ # Circuit diagrams, breadboard layout, images
├── model/ # folder model
└── README.md # (This file)
```

---

## 🎯 Objective

The aim of **Lab 2** is to:

- Practice interfacing with sensors (e.g. temperature, humidity, light, etc.)  
- Read sensor values via microcontroller  
- Transmit or display the sensor data (e.g. via Serial Monitor, LCD, or over WiFi)  
- Understand and debug hardware-software integration  

*(You should replace the above with the real objectives of your Lab 2.)*

---

## 🛠 Hardware Setup

| Component             | Connection / Pin | Notes |
|----------------------|------------------|-------|
| Sensor A (e.g. DHT11) | GPIO 14 (D5)     | Data pin to D5; VCC to 3.3V, GND to GND |
| Sensor B (e.g. Light) | ADC pin 34        | Analog read on pin 34 |
| LED / Indicator        | GPIO 26           | Optional: for status feedback |
| …                    | …                | … |

> Ensure correct wiring of VCC / GND to avoid damaging components.

---

## 📂 Source Code

Inside the `src/` folder, you can find:

- `main.ino` (or equivalent) — the main sketch or program  
- Utility / helper files for sensor reading, data formatting, etc.  
- Comments explaining reading intervals, calibration, and patterns  

### Usage

1. Open `main.ino` in the Arduino IDE (or appropriate IDE)  
2. Adjust configuration constants (e.g. pin assignments, sampling interval)  
3. Compile & upload to your board  
4. Open Serial Monitor (baud rate e.g. 115200)  
5. Observe sensor readings and debug messages  

---

## 📘 Documentation

Inside `docs/`, you may find:

- Technical descriptions of sensors used (e.g. datasheet summary)  
- Flowcharts or state diagrams of the software  
- Step-by-step explanation of logic, timing, and error handling  

Inside `wiring/`:

- Circuit diagrams (schematics or images)  
- Breadboard layout photos  
- Pin mapping illustrations  

---
## Install
- unzip file model.zip
- install requirements.txt
```
pip install -r requirement.txt
```
- test with computer webcam
```
python3 test_with_web_cam.py
```
- test with esp32-CAM
- make sure running esp32 cam
```
python3 display_esp32cam_by_opencv.py
```
---

## 🔧 Troubleshooting & Tips

- Make sure power supply is stable (3.3 V / 5 V)  
- Check all wiring — especially data and ground lines  
- Add small delays between readings to avoid flooding sensor  
- Use serial print statements to debug intermediate values  
- Calibrate sensor readings if needed (e.g. smoothing, offsets)  

---

## 🚀 Possible Extensions

You could extend the basic Lab 2 by:

- Sending the sensor data to a cloud or server (e.g. via WiFi or MQTT)  
- Visualizing data on a web dashboard  
- Adding multiple sensors and aggregating readings  
- Triggering actuators (e.g. fans, LED, relay) based on thresholds  

---

## 👤 Author & Credits

- **Author / Lab Owner:** Minh Khong Cau  
- **GitHub:** [MinhKhongCau](https://github.com/MinhKhongCau)  

---

If you send me the actual source code (e.g. `main.ino`) and hardware details (which board, which sensors) for LAB2, I can help you populate this README with exact pinouts, diagrams, and usage instructions. Do you want me to generate a version filled with your actual code details?
::contentReference[oaicite:0]{index=0}
