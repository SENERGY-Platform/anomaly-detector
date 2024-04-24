from confluent_kafka import Producer
import socket
import json 

conf = {'bootstrap.servers': 'localhost:29092',
        'client.id': socket.gethostname()}

producer = Producer(conf)

producer.produce("analytics", key="key", value=json.dumps({
    "device_id": "id",
    "service_id": "analytics",
    "value": 20000,
    "time": "2024-04-11T14:00:25.737774"
}))
producer.flush()
