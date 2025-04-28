# -*- coding:utf-8 -*-
"""
   @file dataLogger_MQTTpublisher.py
   @brief Log 3-axis vibration data (X, Y, Z) from LIS2DW12 sensor to SQLite database and publish to MQTT.
   @n Logs 4000 samples (~1s) to the database and publishes 200 samples (~0.5s) to MQTT.
"""

import sys
import sqlite3
import time
import json
from datetime import datetime
from DFRobot_LIS2DW12 import *
import paho.mqtt.client as mqtt

# Add library path
sys.path.append("/home/dmvnpdmpoc/Desktop/vibrationLivePlot")

# I2C configuration
I2C_BUS = 0x01
ADDRESS_1 = 0x19
SAMPLE_PERIOD = 1.0 / 400  # 400 Hz

# Database configuration
DB_NAME = '/media/dmvnpdmpoc/ssdDisk/Data/vibration_data.db'
DB_SAMPLE_COUNT = 4000  # Update database every 4000 samples (~10s)

# MQTT configuration
BROKER = 'localhost'
PORT = 1883
TOPIC = 'vibration/data'
MQTT_SAMPLE_COUNT = 200  # Update MQTT every 200 samples (~0.5s)

def init_database():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vibration_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            x_value INTEGER NOT NULL,
            y_value INTEGER NOT NULL,
            z_value INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def init_sensor():
    """Initialize LIS2DW12 sensor."""
    try:
        acce = DFRobot_LIS2DW12_I2C(I2C_BUS, ADDRESS_1)
        if not acce.begin():
            raise Exception("Sensor initialization failed")
        print(f"Chip ID: {hex(acce.get_id())}")
        acce.soft_reset()
        acce.set_range(acce.RANGE_2G)
        acce.contin_refresh(True)
        acce.set_data_rate(acce.RATE_400HZ)  # 400 Hz
        acce.set_filter_path(acce.LPF)
        acce.set_filter_bandwidth(acce.RATE_DIV_2)
        acce.set_power_mode(acce.HIGH_PERFORMANCE_14BIT)
        time.sleep(0.1)
        print("Sensor initialized successfully")
        return acce
    except Exception as e:
        print(f"Sensor initialization failed: {e}")
        exit(1)

def on_connect(client, userdata, flags, rc, properties=None):
    """MQTT connect callback."""
    print(f"Connected to MQTT broker with code {rc}")

def on_publish(client, userdata, mid):
    """MQTT publish callback."""
    print(f"Message {mid} published")

def main():
    # Initialize components
    init_database()
    acce = init_sensor()

    # Separate buffers for database and MQTT: (x, y, z)
    db_buffer = []
    mqtt_buffer = []
    last_sample = time.time()
    next_sample = last_sample + SAMPLE_PERIOD

    # Connect to database
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Set up MQTT client
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_publish = on_publish
    try:
        client.connect(BROKER, PORT, keepalive=60)
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        conn.close()
        return

    client.loop_start()

    try:
        while True:
            current_time = time.time()

            # Sample at ~400 Hz
            if current_time >= next_sample:
                try:
                    x = acce.read_acc_x()
                    y = acce.read_acc_y()
                    z = acce.read_acc_z()
                    db_buffer.append((x, y, z))
                    mqtt_buffer.append((x, y, z))
                    last_sample = current_time
                    next_sample = last_sample + SAMPLE_PERIOD
                except Exception as e:
                    with open("error_log.txt", "a") as f:
                        f.write(f"{datetime.now().isoformat()} - Sensor read error: {e}\n")
                    continue

            # Update database every 4000 samples (~1s)
            if len(db_buffer) >= DB_SAMPLE_COUNT:
                data_to_insert = db_buffer[:DB_SAMPLE_COUNT]
                try:
                    cursor.executemany(
                        'INSERT INTO vibration_readings (x_value, y_value, z_value) VALUES (?, ?, ?)',
                        data_to_insert
                    )
                    conn.commit()
                    print(f"Logged {len(data_to_insert)} samples at {datetime.now().isoformat()}")
                    db_buffer = db_buffer[DB_SAMPLE_COUNT:]  # Remove used samples
                except sqlite3.Error as e:
                    print(f"Database error: {e}")

            # Publish to MQTT every MQTT_SAMPLE_COUNT samples
            if len(mqtt_buffer) >= MQTT_SAMPLE_COUNT:
                data_to_publish = mqtt_buffer[:MQTT_SAMPLE_COUNT]
                x_values = [x for x, _, _ in data_to_publish]
                y_values = [y for _, y, _ in data_to_publish]
                z_values = [z for _, _, z in data_to_publish]
                data = {'x_values': x_values, 'y_values': y_values, 'z_values': z_values}
                message = json.dumps(data)
                try:
                    result = client.publish(TOPIC, message, qos=1)
                    if result.rc != mqtt.MQTT_ERR_SUCCESS:
                        print(f"Publish failed with code {result.rc}")
                except Exception as e:
                    print(f"MQTT publish error: {e}")
                mqtt_buffer = mqtt_buffer[MQTT_SAMPLE_COUNT:]  # Remove used samples

            # Dynamic sleep to reduce CPU usage
            sleep_time = max(0, next_sample - time.time())
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        client.loop_stop()
        client.disconnect()
        conn.close()
        acce.soft_reset()  # Assuming this resets the sensor

if __name__ == '__main__':
    main()
