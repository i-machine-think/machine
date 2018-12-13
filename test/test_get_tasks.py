import unittest

from machine.tasks import Task
from machine.tasks import get_task


class TestGetTask(unittest.TestCase):

    def test_get_lookup_table_task(self):
        task = get_task("lookup")
        self.assertEqual(task.name, "lookup")

    def test_get_symbol_rewriting_task(self):
        task = get_task("symbol_rewriting")
        self.assertEqual(task.name, "Symbol Rewriting")
