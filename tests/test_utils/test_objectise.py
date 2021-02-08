"""Test the `objectise` function"""

import unittest
from utils.objectise import snake2camal, objectise

def dummy_fn(x, y=2):
    return x + y

class TestSnake2Camel(unitest.TestCase):
    def test_snake2camel(self):
        assert snake2camel("hello_world") == "HelloWorld"

class TestObjectise(unittest.TestCase):
    def setUp(self):
        DummyClass = objectise(dummy_fn)
        
        self.object = DummyClass(y=2)
        
    def test_objectise(self):
        assert self.object(3) == 5
        
if __name__ == "__main__":
    unittest.main()
