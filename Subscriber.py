import json
import paho.mqtt.client as mqtt
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib as mpl

plt.style.use('ggplot')
mpl.rcParams['font.family'] = 'Comic Sans MS'
mpl.rcParams['axes.edgecolor'] = '#E0E0E0'
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.facecolor'] = '#FFF7FB'
mpl.rcParams['figure.facecolor'] = '#FFF7FB'
mpl.rcParams['grid.color'] = '#FFB6C1'
mpl.rcParams['axes.titleweight'] = 'bold'

# === MQTT SETUP ===
MQTT_BROKER = '192.168.196.122'
MQTT_PORT = 1883
MQTT_TOPIC = 'sensor/acceleration'
BUFFER_SIZE = 100

timestamps = deque(maxlen=BUFFER_SIZE)
x_values = deque(maxlen=BUFFER_SIZE)
y_values = deque(maxlen=BUFFER_SIZE)
z_values = deque(maxlen=BUFFER_SIZE)

def on_connect(client, userdata, flags, rc):
    print("Connected with result code", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        timestamps.append(data.get("timestamp", ""))
        x_values.append(data.get("x", 0))
        y_values.append(data.get("y", 0))
        z_values.append(data.get("z", 0))
    except Exception as e:
        print("Error processing message:", e)

client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# === FIGURE SETUP ===
fig, axs = plt.subplots(3, 2, figsize=(10, 6))
fig.suptitle("Real-Time Sensor & FFT Visualization", fontsize=18, color="#BA55D3")

((ax_xtime, ax_xfft), (ax_ytime, ax_yfft), (ax_ztime, ax_zfft)) = axs

# Plot lines
line_xtime, = ax_xtime.plot([], [], color='#FF6F91', label='X (Time)', linewidth=2)
line_ytime, = ax_ytime.plot([], [], color='#6A67CE', label='Y (Time)', linewidth=2)
line_ztime, = ax_ztime.plot([], [], color='#17BEBB', label='Z (Time)', linewidth=2)

line_xfft, = ax_xfft.plot([], [], color='#FF6F91', linestyle='--', label='X (FFT)', linewidth=1.5)
line_yfft, = ax_yfft.plot([], [], color='#6A67CE', linestyle='--', label='Y (FFT)', linewidth=1.5)
line_zfft, = ax_zfft.plot([], [], color='#17BEBB', linestyle='--', label='Z (FFT)', linewidth=1.5)

# Decorate axes
for ax, label in zip([ax_xtime, ax_ytime, ax_ztime], ['X', 'Y', 'Z']):
    ax.set_xlim(0, BUFFER_SIZE)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylabel(f"{label} Time", fontsize=10, color='#555')

for ax, label in zip([ax_xfft, ax_yfft, ax_zfft], ['X', 'Y', 'Z']):
    ax.set_xlim(0, 1.0)
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylabel(f"{label} FFT", fontsize=10, color='#666')

ax_ztime.set_xlabel("Sample Index", fontsize=10)
ax_zfft.set_xlabel("Frequency (Hz)", fontsize=10)

# === FFT Calculation ===
def compute_fft(data):
    if len(data) < 10:
        return np.array([]), np.array([])
    arr = np.array(data) - np.mean(data)
    fft_vals = np.abs(np.fft.rfft(arr))
    fft_freqs = np.fft.rfftfreq(len(arr), d=0.5)
    return fft_freqs, fft_vals

# === INIT + ANIMATE ===
def init():
    ax_xtime.set_ylim(-2000, 2000)
    ax_ytime.set_ylim(-2000, 2000)
    ax_ztime.set_ylim(-2000, 2000)

    for ax in [ax_xfft, ax_yfft, ax_zfft]:
        ax.set_ylim(0, 500)

    return line_xtime, line_ytime, line_ztime, line_xfft, line_yfft, line_zfft

def animate(frame):
    xdata = list(range(len(x_values)))
    line_xtime.set_data(xdata, list(x_values))
    line_ytime.set_data(xdata, list(y_values))
    line_ztime.set_data(xdata, list(z_values))

    # FFT
    fx, fftx = compute_fft(x_values)
    fy, ffty = compute_fft(y_values)
    fz, fftz = compute_fft(z_values)

    line_xfft.set_data(fx, fftx)
    line_yfft.set_data(fy, ffty)
    line_zfft.set_data(fz, fftz)

    # Dynamic FFT Y limit
    if len(fftx): ax_xfft.set_ylim(0, max(fftx) * 1.1)
    if len(ffty): ax_yfft.set_ylim(0, max(ffty) * 1.1)
    if len(fftz): ax_zfft.set_ylim(0, max(fftz) * 1.1)

    return line_xtime, line_ytime, line_ztime, line_xfft, line_yfft, line_zfft

# === RUN ===
ani = animation.FuncAnimation(fig, animate, init_func=init, interval=500, blit=True)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

client.loop_stop()
client.disconnect()
