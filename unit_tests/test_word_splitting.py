import unittest
import word_splitting

class TestBpe(unittest.TestCase):
    def test_create_word_split_sets(self):
        id_verses = {0: ['boy boys girl girls '.split(' ')]}
        steps_to_save = {1}
        output = word_splitting.create_word_split_sets(id_verses, steps_to_save)
        self.assertEqual(1, len(output))
        self.assertEqual(0, list(output.keys())[0])
        self.assertEqual(1, len(output[0]))
        self.assertEqual(1, list(output[0].keys())[0])
        self.assertEqual([['boy', 'boy', 's', 'girl', 'girl', 's']], output[0][1])

if __name__ == "__main__":
    unittest.main()
