import sys
import time
import sqlite3
import json
import paho.mqtt.client as mqtt
from DFRobot_LIS2DW12 import *

# === CONFIGURATIONS ===
DB_NAME = 'Data.db'
MQTT_BROKER = '192.168.196.122'  
MQTT_PORT = 1883
MQTT_TOPIC = 'sensor/acceleration'

# === SENSOR SETUP ===
I2C_BUS = 0x01
ADDRESS_1 = 0x19
acce = DFRobot_LIS2DW12_I2C(I2C_BUS, ADDRESS_1)

import socket

def is_broker_alive(host, port, timeout=2):
    try:
        sock = socket.create_connection((host, port), timeout)
        sock.close()
        return True
    except Exception:
        return False

# Kiem tra xem broker co hoat dong ko
if is_broker_alive("192.168.196.122", 1883):
    print("âœ… Broker is alive")
else:
    print("âŒ Broker is not reachable")



def init_sensor():
    acce.begin()
    acce.soft_reset()
    acce.set_range(acce.RANGE_2G)
    acce.contin_refresh(True)
    acce.set_data_rate(acce.RATE_400HZ)
    acce.set_filter_path(acce.LPF)
    acce.set_filter_bandwidth(acce.RATE_DIV_4)
    acce.set_power_mode(acce.HIGH_PERFORMANCE_14BIT)
    time.sleep(0.1)

# === SQLITE SETUP ===
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS acceleration (
                    timestamp TEXT,
                    x REAL,
                    y REAL,
                    z REAL
                 )''')
    conn.commit()
    return conn

# === MQTT SETUP ===
def init_mqtt():
    client = mqtt.Client(protocol=mqtt.MQTTv311)
    client.enable_logger() 
    return client

# === MAIN LOOP ===
def main_loop():
    conn = init_db()
    cursor = conn.cursor()
    mqtt_client = init_mqtt()

    buffer = []
    last_commit_time = time.time()

    while True:
        try:
            x = acce.read_acc_x()
            y = acce.read_acc_y()
            z = acce.read_acc_z()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            buffer.append((timestamp, x, y, z))

            payload = json.dumps({"timestamp": timestamp, "x": x, "y": y, "z": z})
            try:
                mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
                mqtt_client.publish(MQTT_TOPIC, payload)
                mqtt_client.disconnect()
            except Exception as e:
                print(f"[{timestamp}] MQTT publish failed: {e}")

            print(f"[{timestamp}] X={x:.2f} mg, Y={y:.2f} mg, Z={z:.2f} mg")


            if time.time() - last_commit_time >= 0.5 and buffer:
                cursor.executemany("INSERT INTO acceleration (timestamp, x, y, z) VALUES (?, ?, ?, ?)", buffer)
                conn.commit()
                buffer.clear()
                last_commit_time = time.time()

            time.sleep(0.08)
        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print("Error:", e)
            time.sleep(1)

    # Ghi ná»‘t buffer cÃ²n láº¡i trÆ°á»›c khi thoÃ¡t
    if buffer:
        cursor.executemany("INSERT INTO acceleration (timestamp, x, y, z) VALUES (?, ?, ?, ?)", buffer)
        conn.commit()

    conn.close()


if __name__ == '__main__':
    init_sensor()
    main_loop()

