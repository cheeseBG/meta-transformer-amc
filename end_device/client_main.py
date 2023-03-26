import pcap
import dpkt
import keyboard
import pandas as pd
import numpy as np
import os
from datetime import datetime
import time
import socket

# For send CSI data
HOST = '192.168.0.191'
PORT = 9009


# for sampling
def truncate(num, n):
    integer = int(num * (10 ** n)) / (10 ** n)
    return float(integer)


def sniffing(nicname):
    print('Start Sniifing... @', nicname, 'UDP, Port 5500')
    sniffer = pcap.pcap(name=nicname, promisc=True, immediate=True, timeout_ms=50)
    sniffer.setfilter('udp and port 5500')

    before_ts = 0.0

    for ts, pkt in sniffer:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
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

            csi_data = str(mac) + ' '.join(list(csi_cmplx[0]))

            try:
                sock.connect((HOST, PORT))
                sock.sendall(csi_data.encode())
            finally:
                sock.close()

            before_ts = ts

            if keyboard.is_pressed('s'):
                print("Stop Collecting...")
                break


if __name__ == '__main__':
    sniffing('wlan0')