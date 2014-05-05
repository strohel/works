from unittest import TestCase

class TestSimulation(TestCase):

    def test_simple(self):
        from asim.simulation.simple import main
        main()
