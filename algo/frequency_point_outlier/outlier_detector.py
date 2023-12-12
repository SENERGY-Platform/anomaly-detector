import threading 
import datetime
import time 

from river import stats

# TODO
# nur sinnvoll bei sensoren die regelmaesig senden
# nicht bei erkennungssensoren z.b 

class FrequencyDetector(threading.Thread):
    def __init__(
        self, 
        kafka_producer
    ):
        threading.Thread.__init__(self)
        self.last_received_ts = None
        self.kafka_producer = kafka_producer
        self.__stop = False

        self.rolling_iqr = stats.RollingIQR(
            q_inf=0.25,
            q_sup=0.75,
            window_size=100
        )
        self.rolling_quantile = stats.RollingQuantile(
            q=.75,
            window_size=100,
        )

    def run(self):
        while not self.__stop:
            if not self.last_received_ts:
                print("Pause check until first input")
                time.sleep(5)
                continue

            now = datetime.datetime.now()
            waiting_time = self.calculate_time_diff(now, self.last_received_ts)
            print(f"Time since last input {waiting_time}")
            if self.duration_is_anomalous(waiting_time):
                self.kafka_producer.produce()
            
            time.sleep(5)

    def duration_is_anomalous(self, waiting_time):
        iqr = self.rolling_iqr.get()
        if iqr:
            upper_border = 1.5 * iqr + self.rolling_quantile.get()
            print(f"{waiting_time} > {upper_border}")
            return waiting_time > upper_border
        return False

    def stop(self):
        self.__stop = True

    def calculate_time_diff(self, ts1, ts2):
        return (ts1 - ts2).total_seconds() / 60

    def register_input(self, data):
        now = datetime.datetime.now()

        if not self.last_received_ts:
            print('First input arrived')
            self.last_received_ts = now
            return 

        waiting_time = self.calculate_time_diff(now, self.last_received_ts)
        print(f"Waiting time of current input: {waiting_time}")
        self.update_model(waiting_time)
        self.last_received_ts = now

    def update_model(self, waiting_time):
        self.rolling_iqr.update(waiting_time)
        self.rolling_quantile.update(waiting_time)
        
#a = Monitor(None)
#a.start()

#for i in range(10):
#    print(f"RUN {i}")
#    a.register_input(i)
#    time.sleep(1)

# time.sleep(5)