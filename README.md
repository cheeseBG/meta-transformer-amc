# wifi-sensing-master


## To-do
1. HAR? Presence detection? 방향성 결정
2. benchmark dataset 제공
3. Challenge한 부분이 종속성 문제 외에 더 있는지 확안
4. Server로 올릴때 Federated Learning을 써야할 이유가 있을지? privacy 문제에 대한 확인
5. 문제를 수식으로 정의 할 수 있는가? 설계까지.

## Consider
1. RSSI가 낮을수록 성능이 떨어진다.
2. AP에 가까이 위치할수록 여러 공간에서 측정되었는지, AP앞에 한명이있는건지 구분이 안된다.
3. server로 올리는 데이터의 양이 너무 많다. -> 현재는 일정 패턴을 보일때만 server로 데이터 전송
4. False Positive와 False Negative 중 어떤게 더 중요한지??

### ATHCSIFrame

Reference based on the [Atheros CSI Tool User Guide](https://wands.sg/research/wifi/AtherosCSI/document/Atheros-CSI-Tool-User-Guide.pdf).

- timestamp: Timestamp of seconds since epoch for this CSI frame.
- csi_length: Expected length of the CSI matrix payload.
- tx_channel: Wireless channel the collecting device is receiving on (represented in Hz/frequency).
- err_info: PHY error code. 0 when valid.
- noise_floor: Current noise floor.
- rate: Transmission rate (not yet sure if bitmask).
- bandwidth: Transmission bandwidth (0->20MHz, 1->40MHz)
- num_tones: Number of subcarriers (tones).
- nr: Number of receiving antennas present.
- nc: Number of transmitting antennas present.
- rssi: Total observed RSSI (signal strength in dB).
- rssi_1: Observed RSSI on the first receiving antenna.
- rssi_2: Observed RSSI on the second receiving antenna (if present).
- rssi_3: Observed RSSI on the third receiving antenna (if present).
- payload_length: Expected length of the frame payload.

### IWLCSIFrame

- timestamp_low: Timestamp indicating the current state of the IWL5300's built-in clock.
- bfee_count: Index of the frame out of all those observed during uptime.
- n_rx: Number of receiving antennas present.
- n_tx: Number of transmitting antennas present.
- rssi_a: Observed RSSI (signal strength in dB) on the first receiving antenna.
- rssi_b: Observed RSSI on the second receiving antenna (if present).
- rssi_c: Observed RSSI on the third receiving antenna (if present).
- noise: Current noise floor.
- agc: Automatic gain control setting.
- antenna_sel: Bitmask indicating the permutation setting.
- length: Reported length of a CSI payload.
- rate: Bitmask indicating the rate at which this frame was sent.
#### Hint :
IWLCSIFrame visualization can be count at ./docs


### NEXCSIFrame

This format is based on the modified version of [nexmon_csi](https://github.com/nexmonster/nexmon_csi/tree/pi-5.4.51) (credit [mzakharo](https://github.com/seemoo-lab/nexmon_csi/pull/46)) for BCM43455c0 with support for RSSI and Frame Control. If using the [regular](https://github.com/seemoo-lab/nexmon_csi/) version of `nexmon_csi`, these fields will not contain this data.

- rssi: Observed RSSI (signal strength in dB) on the receiving antenna.
- frame_control: [Frame Control](https://en.wikipedia.org/wiki/802.11_Frame_Types#Frame_Control) bitmask.
- source_mac: MAC address for the device which sent the packet.
- sequence_no: Sequence number of the frame for which CSI was captured.
- core: Binary field indicating the core being used.
- spatial_stream: Binary field indicating the spatial stream being used.
- channel_spec: Channel configuration, hex representation of the selected channel and bandwidth pairing.
- chip: Broadcom chipset version of the collecting device.
