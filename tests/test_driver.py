

import os
import unittest
import hashlib

from get_data import NILCDriver


class TestNILCDriver(unittest.TestCase):

    def test_download(self):
        self.driver = NILCDriver(emb_type='word2vec', subtype='cbow', dimension=50)
        self.driver.download()
        md5_hash = hashlib.md5()
        with open(self.driver.output, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                md5_hash.update(byte_block)

        self.assertEqual(md5_hash.hexdigest(), 'fa4345e194bc47f15e5ec194c26db9c2')

    def tearDown(self):
        try:
            os.remove(self.driver.output)
        except OSError as os_err:
            print(os_err)


if __name__ == '__main__':
    unittest.main()
