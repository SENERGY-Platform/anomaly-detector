import threading 
import datetime
import time 
from algo import utils
import json 

# TODO
# nur sinnvoll bei sensoren die regelmaesig senden
# nicht bei erkennungssensoren z.b 

class FrequencyDetector(threading.Thread, utils.StdPointOutlierDetector):
    def __init__(
        self, 
        kafka_producer,
        operator_id,
        pipeline_id,
        output_topic,
        data_path
    ):
        threading.Thread.__init__(self)
        utils.StdPointOutlierDetector.__init__(self, data_path)

        self.last_received_ts = None
        self.kafka_producer = kafka_producer
        self.pause_event = threading.Event()
        self.__stop = False
        self.operator_id = operator_id
        self.pipeline_id = pipeline_id
        self.output_topic = output_topic
        self.operator_start_time = datetime.datetime.now()

    def run(self):
        while self.__stop:
            if not self.last_received_ts:
                print("Pause check until first input")
                time.sleep(5)
                continue

            if self.pause_event.is_set():
                continue

            now = datetime.datetime.now()
            waiting_time = self.calculate_time_diff(now, self.last_received_ts)
            print(f"Time since last input {waiting_time}")
            if self.point_is_anomalous(waiting_time):
                print("Time since last input was anomalous - either too short or too long")
                self.kafka_producer.produce(
                    self.output_topic,
                        json.dumps(
                            {
                                "pipeline_id": self.pipeline_id,
                                "operator_id": self.operator_id,
                                "analytics": {
                                    "anomaly_occured": True, 
                                    "message": "Time since last input was anomalous - either too short or too long"
                                },
                                "time": "{}Z".format(datetime.datetime.utcnow().isoformat())
                            }
                        ),
                        self.operator_id
                )
            
            time.sleep(5)

    def stop(self):
        self.__stop = True

    def pause(self):
        self.pause_event.set()

    def unpause(self):
        self.pause_event.clear()

    def calculate_time_diff(self, ts1, ts2):
        return (ts1 - ts2).total_seconds() / 60

    def register_input(self, data):
        input_timestamp = utils.todatetime(data['energy_time']).tz_localize(None)

        now = datetime.datetime.now()

        if input_timestamp < self.operator_start_time:
            print('Input is historic -> dont use for outlier detection')
            return 

        if not self.last_received_ts:
            print('First input arrived')
            self.last_received_ts = now
            return 

        waiting_time = self.calculate_time_diff(now, self.last_received_ts)
        print(f"Waiting time of current input: {waiting_time}")
        self.update(waiting_time)
        self.last_received_ts = now

    
        