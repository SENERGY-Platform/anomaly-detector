import unittest
import tempfile

from algo.utils import StdPointOutlierDetector

class TestStdPointOutlierDetector(unittest.TestCase):

    def setUp(self):
        self.tmpDir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        self.tmpDir.cleanup()

    def test_parameters(self):
        tests = [
            {
                "input": 2,
                "expected_mean": 2,
                "expected_std": 0,
                "expected_high_anomaly": False,
                "expected_low_anomaly": False  
            },
            {
                "input": 2.5,
                "expected_mean": 2.25,
                "expected_std": 0.25,
                "expected_high_anomaly": False,
                "expected_low_anomaly": False  
            },
            {
                "input": 1.8,
                "expected_mean": 2.1,
                "expected_std": 0.29,
                "expected_high_anomaly": False,
                "expected_low_anomaly": False  
            },
            # First anomaly
            {
                "input": 9,
                "expected_mean": 3.83,
                "expected_std": 3.0,
                "expected_high_anomaly": True,
                "expected_low_anomaly": False  
            },
            {
                "input": -7,
                "expected_mean": 1.66,
                "expected_std": 5.09,
                "expected_high_anomaly": False,
                "expected_low_anomaly": True  
            }
        ]

        detector = StdPointOutlierDetector(self.tmpDir.name)
        for test in tests:
            print(test)
            input_value = test['input']
            self.assertEqual(test['expected_high_anomaly'], detector.point_is_anomalous_high(input_value))
            self.assertEqual(test['expected_low_anomaly'], detector.point_is_anomalous_low(input_value))

            detector.update(input_value)
            self.assertAlmostEqual(test['expected_mean'], detector.current_mean, 2)
            self.assertAlmostEqual(test['expected_std'], detector.current_stddev, 2)

