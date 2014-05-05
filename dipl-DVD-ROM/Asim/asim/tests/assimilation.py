from unittest import TestCase

# Z in the name to be alphabetically after TestSimulation. Hacky, but works
class TestZAssimilation(TestCase):

    def test_twin(self):
        from asim.assimilation.twin import main
        main()
