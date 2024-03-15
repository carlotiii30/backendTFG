import unittest
from src.conexion import manejador, servidor

class TestConexion(unittest.TestCase):
    def setUp(self):
        self.server = servidor.Server('localhost', 12345)

    def test_server(self):
        self.assertIsNotNone(self.server)

if __name__ == '__main__':
    unittest.main()