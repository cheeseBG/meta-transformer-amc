import socketserver
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np
import keras
import joblib
import pandas as pd


def realTimeHPDec(share_value):
    HOST = '192.168.0.130'
    PORT = 9010

    global P_COUNT
    P_COUNT = 0
    WINDOW_SIZE = 50
    SUB_NUM = '_30'

    mac = 'dca6328e1dcb'

    columns = []
    for i in range(0, 64):
        columns.append('_' + str(i))

    null_pilot_col_list = ['_' + str(x + 32) for x in [-32, -31, -30, -29, -21, -7, 0, 7, 21, 29, 30, 31]]

    # Load pretrained model
    print('======> Load model')
    model = keras.models.load_model('../pretrained/continual_cnn1d_model')
    model.summary()
    print('======> Success')

    # Load scaler
    print('======> Load scaler')
    scaler = joblib.load('../pretrained/std_scaler.pkl')
    print('======> Success')

    mac_dict = {}
    mac_dict[mac] = pd.DataFrame(columns=columns)
    mac_dict[mac].drop(null_pilot_col_list, axis=1, inplace=True)

    class MyTcpHandler(socketserver.BaseRequestHandler):

        def handle(self):
            #print('{0} is connected'.format(self.client_address[0]))
            buffer = self.request.recv(2048)  # receive data
            buffer = buffer.decode()
            global P_COUNT
            P_COUNT += 1

            if not buffer:
                print("Fail to receive!")
                return
            else:
                recv_csi = [list(map(float, buffer.split(' ')))]
                csi_df = pd.DataFrame(recv_csi, columns=columns)

                share_value.send(-1)

                '''
                    1. Remove null & pilot subcarrier
                    2. Before input the data, scaling with pretrained-fitted scaler
                    3. Keep window_size 50. If 25 packets changed, choose 1 subcarrier and run model.
                '''
                # 1. Remove null & pilot subcarrier
                csi_df.drop(null_pilot_col_list, axis=1, inplace=True)

                new_columns = csi_df.columns

                # 2. Before input the data, scaling with pretrained-fitted scaler
                csi_data = scaler.transform(csi_df)

                # 3. Keep window_size 50. If 25 packets changed, choose 1 subcarrier and run model
                try:
                    mac_dict[mac] = pd.concat([mac_dict[mac], pd.DataFrame(csi_data, columns=new_columns)], ignore_index=True)
                    if len(mac_dict[mac]) == 50 and P_COUNT == 50:
                        c_data = np.array(mac_dict[mac][SUB_NUM].to_list())
                        c_data = c_data.reshape(-1, 50, 1)

                        pred = model.predict(c_data)

                        print('Predict result: {}'.format(pred))
                        thres = self.__selectThreshold(pred)

                        share_value.send(thres)

                        # Drop first row
                        mac_dict[mac].drop(0, inplace=True)
                        mac_dict[mac].reset_index(drop=True, inplace=True)

                        P_COUNT = 0

                    elif len(mac_dict[mac]) == 50 and P_COUNT == 25:
                        c_data = np.array(mac_dict[mac][SUB_NUM].to_list())
                        c_data = c_data.reshape(-1, 50, 1)

                        pred = model.predict(c_data)

                        print('Predict result: {}'.format(pred))
                        thres = self.__selectThreshold(pred)

                        share_value.send(thres)

                        # Drop first row
                        mac_dict[mac].drop(0, inplace=True)
                        mac_dict[mac].reset_index(drop=True, inplace=True)

                        P_COUNT = 0

                    elif len(mac_dict[mac]) == 50:
                        # Drop first row
                        mac_dict[mac].drop(0, inplace=True)
                        mac_dict[mac].reset_index(drop=True, inplace=True)

                    elif len(mac_dict[mac]) > 50:
                        print("Error!")



                except Exception as e:
                    print('Error', e)

        def __selectThreshold(self, predict):
            if predict[0][0] < 0.5:
                return 0.7
            else:
                return 0.2



    def runServer(HOST, PORT):
        print('==== Start Edge Server ====')
        print('==== Exit with Ctrl + C ====')

        try:
            server = socketserver.TCPServer((HOST, PORT), MyTcpHandler)
            server.serve_forever()  # server_forever()메소드를 호출하면 클라이언트의 접속 요청을 받을 수 있음

        except KeyboardInterrupt:
            print('==== Exit Edge server ====')


    runServer(HOST, PORT)
