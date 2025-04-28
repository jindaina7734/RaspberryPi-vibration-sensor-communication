# -*- coding:utf-8 -*-
"""
   @file vibration_mqtt_plotter.py
   @brief Subscribe to 3-axis vibration data (X, Y, Z) via MQTT and plot live.
   @n Uses first MQTT message to calculate offsets for zeroing sensor data.
   @n Plots zeroed raw acceleration (Y range -2000 to 2000 mg), FFT (10s and 60s), and FFT heatmap (60s history).
   @n Raw data updates per MQTT message (~0.1s, 40 samples) at 400 Hz.
   @n FFT and heatmap use 0-100 Hz range, heatmap updates every 10s with magma colormap.
   @n Spacebar stores latest 60s FFT as purple baseline in FFT plots, updates on next press.
"""

import json
import time
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import paho.mqtt.client as mqtt
import logging

# MQTT configuration
BROKER = '192.168.50.27'  # Adjust to your broker's IP (e.g., Raspberry Pi's IP)
PORT = 1883
TOPIC = 'vibration/data'
RAW_PLOT_WINDOW = 400  # 1 second at 400 Hz for raw data
FFT_BUFFER_SIZE = 4000  # 10 seconds at 400 Hz for FFT
FFT_BUFFER_SIZE_LONG = 24000  # 60 seconds at 400 Hz for FFT
SAMPLE_RATE = 400  # Hz
MIN_REDRAW_INTERVAL = 0.01  # Allow fast redraws for responsiveness
HEATMAP_HISTORY = 60  # Seconds
HEATMAP_STEPS = HEATMAP_HISTORY // 10  # 6 steps (10s each)
FREQ_LIMIT = 100  # Hz, limit for FFT and heatmap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_plot():
    """Initialize live plot with 9 subplots: raw data, FFT, and heatmap for X, Y, Z."""
    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    (ax_x_raw, ax_x_fft, ax_x_heatmap), (ax_y_raw, ax_y_fft, ax_y_heatmap), (ax_z_raw, ax_z_fft, ax_z_heatmap) = axes

    # Initialize deques for raw data (1s)
    data_x = deque([0] * RAW_PLOT_WINDOW, maxlen=RAW_PLOT_WINDOW)
    data_y = deque([0] * RAW_PLOT_WINDOW, maxlen=RAW_PLOT_WINDOW)
    data_z = deque([0] * RAW_PLOT_WINDOW, maxlen=RAW_PLOT_WINDOW)

    # Initialize buffers for FFT data (10s)
    fft_buffer_x = deque(maxlen=FFT_BUFFER_SIZE)
    fft_buffer_y = deque(maxlen=FFT_BUFFER_SIZE)
    fft_buffer_z = deque(maxlen=FFT_BUFFER_SIZE)

    # Initialize long FFT buffers (60s)
    fft_buffer_x_long = deque(maxlen=FFT_BUFFER_SIZE_LONG)
    fft_buffer_y_long = deque(maxlen=FFT_BUFFER_SIZE_LONG)
    fft_buffer_z_long = deque(maxlen=FFT_BUFFER_SIZE_LONG)

    # Initialize heatmap data (6 time steps x 101 freq bins up to 100 Hz)
    freq_bins = int(FREQ_LIMIT * FFT_BUFFER_SIZE / SAMPLE_RATE) + 1  # 101 bins
    heatmap_data_x = np.zeros((HEATMAP_STEPS, freq_bins))
    heatmap_data_y = np.zeros((HEATMAP_STEPS, freq_bins))
    heatmap_data_z = np.zeros((HEATMAP_STEPS, freq_bins))

    # Time points for raw data X-axis (-1 to 0 seconds)
    time_points = np.linspace(-1, 0, RAW_PLOT_WINDOW)

    # Frequency points for FFT (0 to 100 Hz)
    freqs = np.fft.rfftfreq(FFT_BUFFER_SIZE, d=1/SAMPLE_RATE)[:freq_bins]
    freqs_long = np.fft.rfftfreq(FFT_BUFFER_SIZE_LONG, d=1/SAMPLE_RATE)[:int(FREQ_LIMIT * FFT_BUFFER_SIZE_LONG / SAMPLE_RATE) + 1]

    # Time points for heatmap Y-axis (-60 to 0 seconds, newest at top)
    heatmap_times = np.linspace(-HEATMAP_HISTORY, 0, HEATMAP_STEPS)

    # Raw data plots
    line_x_raw, = ax_x_raw.plot(time_points, list(data_x), color='orange', label='X-axis')
    ax_x_raw.set_xlabel('Time (s)')
    ax_x_raw.set_ylabel('X Acceleration (mg)')
    ax_x_raw.set_ylim(-2000, 2000)  # Static Y range
    ax_x_raw.legend()
    ax_x_raw.grid(True)

    line_y_raw, = ax_y_raw.plot(time_points, list(data_y), color='green', label='Y-axis')
    ax_y_raw.set_xlabel('Time (s)')
    ax_y_raw.set_ylabel('Y Acceleration (mg)')
    ax_y_raw.set_ylim(-2000, 2000)  # Static Y range
    ax_y_raw.legend()
    ax_y_raw.grid(True)

    line_z_raw, = ax_z_raw.plot(time_points, list(data_z), color='blue', label='Z-axis')
    ax_z_raw.set_xlabel('Time (s)')
    ax_z_raw.set_ylabel('Z Acceleration (mg)')
    ax_z_raw.set_ylim(-2000, 2000)  # Static Y range
    ax_z_raw.legend()
    ax_z_raw.grid(True)

    # FFT plots
    line_x_fft, = ax_x_fft.plot(freqs, np.zeros(len(freqs)), color='orange', label='X-axis (10s)')
    line_x_fft_long, = ax_x_fft.plot(freqs_long, np.zeros(len(freqs_long)), color='black', linestyle='--', alpha=0.7, label='X-axis (60s)')
    line_x_baseline, = ax_x_fft.plot(freqs_long, np.zeros(len(freqs_long)), color='purple', linestyle='-', alpha=0.7, label='Baseline')
    ax_x_fft.set_xlabel('Frequency (0-100 Hz)')
    ax_x_fft.set_ylabel('X Amplitude (mg)')
    ax_x_fft.set_xlim(0, FREQ_LIMIT)
    ax_x_fft.legend()
    ax_x_fft.grid(True)

    line_y_fft, = ax_y_fft.plot(freqs, np.zeros(len(freqs)), color='green', label='Y-axis (10s)')
    line_y_fft_long, = ax_y_fft.plot(freqs_long, np.zeros(len(freqs_long)), color='black', linestyle='--', alpha=0.7, label='Y-axis (60s)')
    line_y_baseline, = ax_y_fft.plot(freqs_long, np.zeros(len(freqs_long)), color='purple', linestyle='-', alpha=0.7, label='Baseline')
    ax_y_fft.set_xlabel('Frequency (0-100 Hz)')
    ax_y_fft.set_ylabel('Y Amplitude (mg)')
    ax_y_fft.set_xlim(0, FREQ_LIMIT)
    ax_y_fft.legend()
    ax_y_fft.grid(True)

    line_z_fft, = ax_z_fft.plot(freqs, np.zeros(len(freqs)), color='blue', label='Z-axis (10s)')
    line_z_fft_long, = ax_z_fft.plot(freqs_long, np.zeros(len(freqs_long)), color='black', linestyle='--', alpha=0.7, label='Z-axis (60s)')
    line_z_baseline, = ax_z_fft.plot(freqs_long, np.zeros(len(freqs_long)), color='purple', linestyle='-', alpha=0.7, label='Baseline')
    ax_z_fft.set_xlabel('Frequency (0-100 Hz)')
    ax_z_fft.set_ylabel('Z Amplitude (mg)')
    ax_z_fft.set_xlim(0, FREQ_LIMIT)
    ax_z_fft.legend()
    ax_z_fft.grid(True)

    # Heatmap plots
    heatmap_x = ax_x_heatmap.imshow(heatmap_data_x, aspect='auto', origin='lower',
                                   extent=[0, FREQ_LIMIT, -HEATMAP_HISTORY, 0],
                                   cmap='magma')
    ax_x_heatmap.set_xlabel('X Frequency (0-100 Hz)')
    ax_x_heatmap.set_ylabel('Time (s)')
    fig.colorbar(heatmap_x, ax=ax_x_heatmap, label='Amplitude (mg)')

    heatmap_y = ax_y_heatmap.imshow(heatmap_data_y, aspect='auto', origin='lower',
                                   extent=[0, FREQ_LIMIT, -HEATMAP_HISTORY, 0],
                                   cmap='magma')
    ax_y_heatmap.set_xlabel('Y Frequency (0-100 Hz)')
    ax_y_heatmap.set_ylabel('Time (s)')
    fig.colorbar(heatmap_y, ax=ax_y_heatmap, label='Amplitude (mg)')

    heatmap_z = ax_z_heatmap.imshow(heatmap_data_z, aspect='auto', origin='lower',
                                   extent=[0, FREQ_LIMIT, -HEATMAP_HISTORY, 0],
                                   cmap='magma')
    ax_z_heatmap.set_xlabel('Z Frequency (0-100 Hz)')
    ax_z_heatmap.set_ylabel('Time (s)')
    fig.colorbar(heatmap_z, ax=ax_z_heatmap, label='Amplitude (mg)')

    # Set column titles
    # Make the titles bold and centered
    ax_x_raw.set_title("Raw Data", fontweight='bold', fontsize=16, loc='center')      
    ax_x_fft.set_title("FFT", fontweight='bold', fontsize=16, loc='center')
    ax_x_heatmap.set_title("Heatmap", fontweight='bold', fontsize=16, loc='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return (fig, ax_x_raw, ax_y_raw, ax_z_raw, ax_x_fft, ax_y_fft, ax_z_fft, ax_x_heatmap, ax_y_heatmap, ax_z_heatmap,
            line_x_raw, line_y_raw, line_z_raw,
            line_x_fft, line_y_fft, line_z_fft,
            line_x_fft_long, line_y_fft_long, line_z_fft_long,
            line_x_baseline, line_y_baseline, line_z_baseline,
            heatmap_x, heatmap_y, heatmap_z,
            data_x, data_y, data_z,
            fft_buffer_x, fft_buffer_y, fft_buffer_z,
            fft_buffer_x_long, fft_buffer_y_long, fft_buffer_z_long,
            heatmap_data_x, heatmap_data_y, heatmap_data_z,
            freqs, freqs_long)

def on_connect(client, userdata, flags, reason_code, properties=None):
    """Callback for when the client connects to the broker."""
    if reason_code == 0:
        logging.info(f"Connected to MQTT broker at {BROKER}:{PORT}")
        client.subscribe(TOPIC, qos=0)
        logging.info(f"Subscribed to {TOPIC}")
    else:
        logging.error(f"Connection failed with code {reason_code}")

def on_disconnect(client, userdata, rc, properties=None):
    """Callback for when the client disconnects."""
    if rc != 0:
        logging.warning(f"Unexpected disconnection. Reason code: {rc}. Attempting to reconnect...")

def on_key(event, userdata):
    """Handle key press events."""
    if event.key == ' ':
        if userdata['baseline_ready']:
            # Use stored 60s FFT amplitudes
            userdata['line_x_baseline'].set_ydata(userdata['baseline_x'])
            userdata['line_y_baseline'].set_ydata(userdata['baseline_y'])
            userdata['line_z_baseline'].set_ydata(userdata['baseline_z'])

            # Auto-scale FFT Y-axes
            userdata['ax_x_fft'].relim()
            userdata['ax_x_fft'].autoscale_view()
            userdata['ax_y_fft'].relim()
            userdata['ax_y_fft'].autoscale_view()
            userdata['ax_z_fft'].relim()
            userdata['ax_z_fft'].autoscale_view()

            # Redraw
            userdata['fig'].canvas.draw()
            userdata['fig'].canvas.flush_events()
            logging.info("Baseline updated with latest 60s FFT")
        else:
            logging.warning(f"No 60s FFT available for baseline")

def on_message(client, userdata, msg):
    """Callback for when a message is received. Updates raw data, FFT, and heatmap."""
    try:
        start_time = time.time()
        if start_time - userdata['last_redraw'] < MIN_REDRAW_INTERVAL:
            return  # Skip if redraw is too frequent

        data = json.loads(msg.payload.decode('utf-8'))
        if all(k in data for k in ('x_values', 'y_values', 'z_values')):
            if (isinstance(data['x_values'], list) and
                isinstance(data['y_values'], list) and
                isinstance(data['z_values'], list)):
                all_values = data['x_values'] + data['y_values'] + data['z_values']
                if all(isinstance(v, (int, float)) for v in all_values):
                    x_values = np.array(data['x_values'], dtype=float)
                    y_values = np.array(data['y_values'], dtype=float)
                    z_values = np.array(data['z_values'], dtype=float)

                    # Calculate offsets from first message
                    if not userdata['offsets_calculated']:
                        if len(x_values) > 0 and len(y_values) > 0 and len(z_values) > 0:
                            x_offset = np.mean(x_values)
                            y_offset = np.mean(y_values)
                            z_offset = np.mean(z_values)
                            userdata['x_offset'] = x_offset
                            userdata['y_offset'] = y_offset
                            userdata['z_offset'] = z_offset
                            userdata['offsets_calculated'] = True
                            logging.info(f"Offsets calculated: X={x_offset:.2f} mg, Y={y_offset:.2f} mg, Z={z_offset:.2f} mg")
                            # Plot first message as zeros
                            x_zeroed = np.zeros_like(x_values)
                            y_zeroed = np.zeros_like(y_values)
                            z_zeroed = np.zeros_like(z_values)
                        else:
                            logging.warning("Empty values in first message")
                            return
                    else:
                        # Zero the data
                        x_zeroed = x_values - userdata['x_offset']
                        y_zeroed = y_values - userdata['y_offset']
                        z_zeroed = z_values - userdata['z_offset']

                    # Update raw data deques
                    for x_val in x_zeroed:
                        userdata['data_x'].append(x_val)
                    for y_val in y_zeroed:
                        userdata['data_y'].append(y_val)
                    for z_val in z_zeroed:
                        userdata['data_z'].append(z_val)

                    # Update FFT buffers
                    for x_val, y_val, z_val in zip(x_zeroed, y_zeroed, z_zeroed):
                        userdata['fft_buffer_x'].append(x_val)
                        userdata['fft_buffer_y'].append(y_val)
                        userdata['fft_buffer_z'].append(z_val)
                        userdata['fft_buffer_x_long'].append(x_val)
                        userdata['fft_buffer_y_long'].append(y_val)
                        userdata['fft_buffer_z_long'].append(z_val)

                    # Calculate FFT if 10s buffer is full
                    if len(userdata['fft_buffer_x']) >= FFT_BUFFER_SIZE:
                        fft_x = np.fft.rfft(list(userdata['fft_buffer_x']))
                        fft_y = np.fft.rfft(list(userdata['fft_buffer_y']))
                        fft_z = np.fft.rfft(list(userdata['fft_buffer_z']))

                        # Scale amplitudes (2/N for two-sided spectrum)
                        amp_x = 2 * np.abs(fft_x) / FFT_BUFFER_SIZE
                        amp_y = 2 * np.abs(fft_y) / FFT_BUFFER_SIZE
                        amp_z = 2 * np.abs(fft_z) / FFT_BUFFER_SIZE

                        # Limit to 0-100 Hz
                        freq_bins = int(FREQ_LIMIT * FFT_BUFFER_SIZE / SAMPLE_RATE) + 1
                        amp_x = amp_x[:freq_bins]
                        amp_y = amp_y[:freq_bins]
                        amp_z = amp_z[:freq_bins]

                        # Update FFT lines
                        userdata['line_x_fft'].set_ydata(amp_x)
                        userdata['line_y_fft'].set_ydata(amp_y)
                        userdata['line_z_fft'].set_ydata(amp_z)

                        # Update heatmap data
                        userdata['heatmap_data_x'] = np.roll(userdata['heatmap_data_x'], -1, axis=0)
                        userdata['heatmap_data_y'] = np.roll(userdata['heatmap_data_y'], -1, axis=0)
                        userdata['heatmap_data_z'] = np.roll(userdata['heatmap_data_z'], -1, axis=0)
                        userdata['heatmap_data_x'][-1, :] = amp_x
                        userdata['heatmap_data_y'][-1, :] = amp_y
                        userdata['heatmap_data_z'][-1, :] = amp_z

                        # Update heatmap plots
                        userdata['heatmap_x'].set_data(userdata['heatmap_data_x'])
                        userdata['heatmap_y'].set_data(userdata['heatmap_data_y'])
                        userdata['heatmap_z'].set_data(userdata['heatmap_data_z'])

                        # Auto-scale heatmap color limits
                        userdata['heatmap_x'].autoscale()
                        userdata['heatmap_y'].autoscale()
                        userdata['heatmap_z'].autoscale()

                        # Auto-scale FFT Y-axes
                        userdata['ax_x_fft'].relim()
                        userdata['ax_x_fft'].autoscale_view()
                        userdata['ax_y_fft'].relim()
                        userdata['ax_y_fft'].autoscale_view()
                        userdata['ax_z_fft'].relim()
                        userdata['ax_z_fft'].autoscale_view()

                        # Flush 10s buffers
                        userdata['fft_buffer_x'].clear()
                        userdata['fft_buffer_y'].clear()
                        userdata['fft_buffer_z'].clear()

                    # Calculate long FFT when buffer is full
                    if len(userdata['fft_buffer_x_long']) >= FFT_BUFFER_SIZE_LONG:
                        fft_x_long = np.fft.rfft(list(userdata['fft_buffer_x_long']))
                        fft_y_long = np.fft.rfft(list(userdata['fft_buffer_y_long']))
                        fft_z_long = np.fft.rfft(list(userdata['fft_buffer_z_long']))

                        # Scale amplitudes
                        amp_x_long = 2 * np.abs(fft_x_long) / FFT_BUFFER_SIZE_LONG
                        amp_y_long = 2 * np.abs(fft_y_long) / FFT_BUFFER_SIZE_LONG
                        amp_z_long = 2 * np.abs(fft_z_long) / FFT_BUFFER_SIZE_LONG

                        # Limit to 0-100 Hz
                        freq_bins_long = int(FREQ_LIMIT * FFT_BUFFER_SIZE_LONG / SAMPLE_RATE) + 1
                        amp_x_long = amp_x_long[:freq_bins_long]
                        amp_y_long = amp_y_long[:freq_bins_long]
                        amp_z_long = amp_z_long[:freq_bins_long]

                        # Store for baseline
                        userdata['baseline_x'] = amp_x_long
                        userdata['baseline_y'] = amp_y_long
                        userdata['baseline_z'] = amp_z_long
                        userdata['baseline_ready'] = True

                        # Update long FFT lines
                        userdata['line_x_fft_long'].set_ydata(amp_x_long)
                        userdata['line_y_fft_long'].set_ydata(amp_y_long)
                        userdata['line_z_fft_long'].set_ydata(amp_z_long)

                        # Clear long FFT buffers
                        userdata['fft_buffer_x_long'].clear()
                        userdata['fft_buffer_y_long'].clear()
                        userdata['fft_buffer_z_long'].clear()

                    # Update raw plot lines
                    userdata['line_x_raw'].set_ydata(list(userdata['data_x']))
                    userdata['line_y_raw'].set_ydata(list(userdata['data_y']))
                    userdata['line_z_raw'].set_ydata(list(userdata['data_z']))

                    # Redraw plot
                    userdata['fig'].canvas.draw()
                    userdata['fig'].canvas.flush_events()
                    userdata['last_redraw'] = start_time
                else:
                    logging.warning("Non-numeric values found in data")
            else:
                logging.warning("Message values are not lists")
        else:
            logging.warning("Message missing required keys: x_values, y_values, z_values")
    except json.JSONDecodeError:
        logging.error("Invalid JSON received")
    except Exception as e:
        logging.error(f"Unexpected error in on_message: {e}")

def main():
    # Initialize plot
    (fig, ax_x_raw, ax_y_raw, ax_z_raw, ax_x_fft, ax_y_fft, ax_z_fft, ax_x_heatmap, ax_y_heatmap, ax_z_heatmap,
     line_x_raw, line_y_raw, line_z_raw,
     line_x_fft, line_y_fft, line_z_fft,
     line_x_fft_long, line_y_fft_long, line_z_fft_long,
     line_x_baseline, line_y_baseline, line_z_baseline,
     heatmap_x, heatmap_y, heatmap_z,
     data_x, data_y, data_z,
     fft_buffer_x, fft_buffer_y, fft_buffer_z,
     fft_buffer_x_long, fft_buffer_y_long, fft_buffer_z_long,
     heatmap_data_x, heatmap_data_y, heatmap_data_z,
     freqs, freqs_long) = init_plot()

    # Initialize userdata for storing plot objects and offsets
    user_data = {
        'fig': fig,
        'ax_x_raw': ax_x_raw,
        'ax_y_raw': ax_y_raw,
        'ax_z_raw': ax_z_raw,
        'ax_x_fft': ax_x_fft,
        'ax_y_fft': ax_y_fft,
        'ax_z_fft': ax_z_fft,
        'ax_x_heatmap': ax_x_heatmap,
        'ax_y_heatmap': ax_y_heatmap,
        'ax_z_heatmap': ax_z_heatmap,
        'line_x_raw': line_x_raw,
        'line_y_raw': line_y_raw,
        'line_z_raw': line_z_raw,
        'line_x_fft': line_x_fft,
        'line_y_fft': line_y_fft,
        'line_z_fft': line_z_fft,
        'line_x_fft_long': line_x_fft_long,
        'line_y_fft_long': line_y_fft_long,
        'line_z_fft_long': line_z_fft_long,
        'line_x_baseline': line_x_baseline,
        'line_y_baseline': line_y_baseline,
        'line_z_baseline': line_z_baseline,
        'heatmap_x': heatmap_x,
        'heatmap_y': heatmap_y,
        'heatmap_z': heatmap_z,
        'data_x': data_x,
        'data_y': data_y,
        'data_z': data_z,
        'fft_buffer_x': fft_buffer_x,
        'fft_buffer_y': fft_buffer_y,
        'fft_buffer_z': fft_buffer_z,
        'fft_buffer_x_long': fft_buffer_x_long,
        'fft_buffer_y_long': fft_buffer_y_long,
        'fft_buffer_z_long': fft_buffer_z_long,
        'heatmap_data_x': heatmap_data_x,
        'heatmap_data_y': heatmap_data_y,
        'heatmap_data_z': heatmap_data_z,
        'freqs': freqs,
        'freqs_long': freqs_long,
        'last_redraw': time.time(),
        'x_offset': 0.0,
        'y_offset': 0.0,
        'z_offset': 0.0,
        'offsets_calculated': False,
        'baseline_x': np.zeros(len(freqs_long)),
        'baseline_y': np.zeros(len(freqs_long)),
        'baseline_z': np.zeros(len(freqs_long)),
        'baseline_ready': False
    }

    # Add key press handler
    def on_key_wrapper(event):
        on_key(event, user_data)

    fig.canvas.mpl_connect('key_press_event', on_key_wrapper)

    # Add a callback to close the program when the plot is closed
    def on_close(event):
        logging.info("Plot window closed. Exiting program...")
        plt.ioff()
        plt.close()
        exit(0)

    fig.canvas.mpl_connect('close_event', on_close)

    # Set up MQTT client
    client = mqtt.Client(protocol=mqtt.MQTTv5, userdata=user_data)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Enable automatic reconnection
    client.reconnect_delay_set(min_delay=1, max_delay=120)

    # Attempt to connect with retry logic
    while True:
        try:
            client.connect(BROKER, PORT, keepalive=15)
            break
        except Exception as e:
            logging.error(f"Failed to connect to MQTT broker: {e}")
            time.sleep(5)

    try:
        client.loop_forever()  # Run MQTT loop indefinitely
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt detected. Stopping...")
    except Exception as e:
        logging.critical(f"Unexpected error in main loop: {e}")
    finally:
        client.loop_stop()
        client.disconnect()
        plt.ioff()
        plt.close()

if __name__ == '__main__':
    main()