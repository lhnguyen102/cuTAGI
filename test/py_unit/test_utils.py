import os
import sys
import unittest

import numpy as np

from pytagi import Utils

# path to binding code
sys.path.append(
    os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
)


class TestUtils(unittest.TestCase):
    def test_get_errors_returns_one_prediction_per_sample(self):
        num_classes, batch_size = 4, 2
        utils = Utils()
        hrc = utils.get_hierarchical_softmax(num_classes)
        ma = np.zeros(batch_size * hrc.len, dtype=np.float32)
        Sa = np.ones(batch_size * hrc.len, dtype=np.float32)
        labels = np.array([0, 3], dtype=np.int32)

        pred, prob = utils.get_errors(ma, Sa, labels, num_classes, batch_size)

        self.assertEqual(len(pred), batch_size)
        self.assertEqual(len(prob), batch_size * num_classes)


if __name__ == "__main__":
    unittest.main()
