'''
    Configuration of Extractor and plot parameters
'''

EXTRACTOR_CONFIG = {
    'wifi_chip': 'bcm43455c0',  # Raspberry Pi B3+ / B4
    'band': '2.4GHz',  # 5GHz
    'bandwidth': 20,  # 40, 80, 160 (MHz)

    # 여러개의 TX로 부터 CSI를 추출하는 상황에서 특정 tramsmitter에 대해 real time plot을 사용할때
    'use_TX_MAC': False,
    'TX_MAC': 'dca6328e1dcb',  # enter transmitter MAC address

    # For Sampling num per second
    'SAMPLE': 10,

    'null_20MHz': ['_' + str(x+32) for x in [-32, -31, -30, -29,
                                              31,  30,  29,  0]],

    'null_40MHz': ['_' + str(x+64) for x in [-64, -63, -62, -61, -60, -59, -1,
                                          63,  62,  61,  60,  59,  1,  0]],

    'null_80MHz': ['_' + str(x+128) for x in [-128, -127, -126, -125, -124, -123, -1,
                                           127,  126,  125,  124,  123,  1,  0]],

    'null_160MHz': ['_' + str(x+256) for x in [-256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
                                           255,  254,  253,  252,  251,  129,  128,  127,  5,  4,  3,  3,  1,  0]],

    'pilot_20MHz': ['_' + str(x+32) for x in [-21, -7, 21,  7]],

    'pilot_40MHz': ['_' + str(x+64) for x in [-53, -25, -11, 53,  25,  11]],

    'pilot_80MHz': ['_' + str(x+128) for x in [-103, -75, -39, -11, 103,  75,  39,  11]],

    'pilot_160MHz': ['_' + str(x+256)for x in [-231, -203, -167, -139, -117, -89, -53, -25,
                                            231,  203,  167,  139,  117, 89,  53,  25]]
}
