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
