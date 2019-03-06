import unittest
import mock


from machine.util.callbacks import EarlyStoppingCallback


class TestEarlyStoppingCallback(unittest.TestCase):
    def setUp(self):
        self.temp_callback = EarlyStoppingCallback()
        self.temp_callback._get_loss_metric = mock.MagicMock(return_value=3)

    def test_train_begin(self):
        self.temp_callback.on_train_begin()

    def test_get_current(self):
        mock_metric = mock.MagicMock()
        mock_metric.name = 'tmp'
        tmp_info = {self.temp_callback.monitor: [mock_metric]}

        # test with no objective name
        self.assertEqual(3, self.temp_callback._get_current(tmp_info))
        # test with present objective name
        self.temp_callback.objective_name = 'tmp'
        self.assertEqual(3, self.temp_callback._get_current(tmp_info))
        # test for missing objective name in monitor
        tmp_info = {self.temp_callback.monitor: []}
        self.assertRaises(
            ValueError, lambda: self.temp_callback._get_current(tmp_info))

    def test_on_epoch_end(self):
        self.temp_callback.wait = 0
        # Update if minimize if True
        self.temp_callback._get_current = mock.MagicMock(return_value=3)
        self.temp_callback.best = 4
        self.temp_callback.on_epoch_end()
        self.assertEqual(3, self.temp_callback.best)

        # Don't Update if minimize and current higher
        self.temp_callback.best = 2
        self.temp_callback.on_epoch_end()
        self.assertEqual(2, self.temp_callback.best)

        # Don't Update if delta
        self.temp_callback.min_delta = 2
        self.temp_callback.best = 4
        self.temp_callback.on_epoch_end()
        self.assertEqual(4, self.temp_callback.best)

        # Update if maximizing
        self.temp_callback.min_delta = 0
        self.temp_callback.minimize = -1
        self.temp_callback.best = 2
        self.temp_callback.on_epoch_end()
        self.assertEqual(3, self.temp_callback.best)

        # Don't Update if maximizing and current lower
        self.temp_callback.best = 4
        self.temp_callback.on_epoch_end()
        self.assertEqual(4, self.temp_callback.best)

        # Don't update if delta
        self.temp_callback.min_delta = 2
        self.temp_callback.best = 2
        self.temp_callback.on_epoch_end()
        self.assertEqual(2, self.temp_callback.best)

    def test_bad_monitor_value(self):
        self.assertRaises(
            ValueError, lambda: EarlyStoppingCallback(monitor='eval'))
        self.assertRaises(
            ValueError, lambda: EarlyStoppingCallback(monitor='val_losses'))


if __name__ == '__main__':
    unittest.main()
