import pcap
import dpkt
import keyboard
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import Process
from matplotlib.artist import Artist

#selected_mac = sys.argv[1]

BANDWIDTH = 20

# number of subcarrier
NSUB = int(BANDWIDTH * 3.2)

# : 제외
selected_mac = 'dca6328e1dcb'
show_packet_length = 100
GAP_PACKET_NUM = 20


# for sampling
def truncate(num, n):
    integer = int(num * (10 ** n)) / (10 ** n)
    return float(integer)


def sniffing(nicname, mac_address):
    print('Start Sniifing... @', nicname, 'UDP, Port 5500')
    sniffer = pcap.pcap(name=nicname, promisc=True, immediate=True, timeout_ms=50)
    sniffer.setfilter('udp and port 5500')

    before_ts = 0.0

    # ####### RealTime plot ###############
    x = np.arange(0, show_packet_length, 1)
    y_list = []

    for i in range(0, NSUB):
        y_list.append([0 for j in range(0, show_packet_length)])

    plt.ion()

    fig, ax = plt.subplots(figsize=(12, 8))

    line_list = []

    for y in y_list:
        line, = ax.plot(x, y, alpha=0.5)
        line_list.append(line)

    plt.title('{}'.format(selected_mac), fontsize=18)
    plt.ylabel('Signal Amplitude', fontsize=16)
    plt.xlabel('Packet', fontsize=16)
    plt.ylim(0, 1500)

    # Amp Min-Max gap text on plot figure
    txt = ax.text(40, 1600, 'Amp Min-Max Gap: None', fontsize=14)
    gap_count = 0
    minmax = []

    idx = show_packet_length - 1
    # ####################################################

    for ts, pkt in sniffer:
        if int(ts) == int(before_ts):
            cur_ts = truncate(ts, 1)
            bef_ts = truncate(before_ts, 1)

            if cur_ts == bef_ts:
                before_ts = ts
                continue

        eth = dpkt.ethernet.Ethernet(pkt)
        ip = eth.data
        udp = ip.data

        # MAC Address 추출
        # UDP Payload에서 Four Magic Byte (0x11111111) 이후 6 Byte는 추출된 Mac Address 의미
        mac = udp.data[4:10].hex()

        if mac != mac_address:
            continue


        # Four Magic Byte + 6 Byte Mac Address + 2 Byte Sequence Number + 2 Byte Core and Spatial Stream Number + 2 Byte Chanspac + 2 Byte Chip Version 이후 CSI
        # 4 + 6 + 2 + 2 + 2 + 2 = 18 Byte 이후 CSI 데이터
        csi = udp.data[18:]

        bandwidth = ip.__hdr__[2][2]
        nsub = int(bandwidth * 3.2)

        # Convert CSI bytes to numpy array
        csi_np = np.frombuffer(
            csi,
            dtype=np.int16,
            count=nsub * 2
        )

        # Cast numpy 1-d array to matrix
        csi_np = csi_np.reshape((1, nsub * 2))

        # Convert csi into complex numbers
        csi_cmplx = np.fft.fftshift(
            csi_np[:1, ::2] + 1.j * csi_np[:1, 1::2], axes=(1,)
        )

        csi_data = list(np.abs(csi_cmplx)[0])

        # real time plot
        idx += 1
        for i, y in enumerate(y_list):
            del y[0]
            new_y = csi_data[i]
            y.append(new_y)
            line_list[i].set_xdata(x)
            line_list[i].set_ydata(y)

            # Min-Max Gap
            if gap_count == 0:
                minmax.append([new_y, new_y])
            else:
                # Update min
                if minmax[i][0] > new_y:
                    minmax[i][0] = new_y
                # Update max
                if minmax[i][1] < new_y:
                    minmax[i][1] = new_y

        gap_list = []
        for mm in minmax:
            gap_list.append(mm[1] - mm[0])

        gap = max(gap_list)

        Artist.remove(txt)
        txt = ax.text(40, 1600, 'Amp Min-Max Gap: {}'.format(gap), fontsize=14)
        gap_count += 1
        if gap_count == 20:
            gap_count = 0
            minmax = []

        fig.canvas.draw()
        fig.canvas.flush_events()

        before_ts = ts

        if keyboard.is_pressed('s'):
            print("Stop Collecting...")
            exit()


if __name__ == '__main__':
    sniffing('wlan0', selected_mac)