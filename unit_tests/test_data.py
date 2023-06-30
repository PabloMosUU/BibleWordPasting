import unittest
import data

class TestData(unittest.TestCase):
    def test_batch(self):
        sequences = [el.split() for el in \
                     ['I love it', 'My name is Earl', 'Whose line is it anyway', 'As you like it', 'Everybody']]
        words = list(set([el for lis in sequences for el in lis]))
        word_index = {wd: ix for ix, wd in enumerate(words)}
        word_index[data.PAD_TOKEN] = 170
        word_index[data.START_OF_VERSE_TOKEN] = 171
        word_index[data.END_OF_VERSE_TOKEN] = 172
        dataset, original_sequence_lengths = data.batch(sequences, 2, word_index)
        self.assertEqual(3, len(dataset))
        self.assertTrue(all([len(el) <= 2 for el in dataset]))
        self.assertTrue(all([len(seq) == 6 for seq in dataset[0]]))
        self.assertTrue(all([len(seq) == 7 for seq in dataset[1]]))
        self.assertTrue(all([all([seq[0] == 171 for seq in b]) for b in dataset]))
        self.assertTrue(all([all([seq[-1] in (170, 172) for seq in b]) for b in dataset]))
        self.assertEqual(172, dataset[0][1][-2])
        self.assertEqual(6, original_sequence_lengths[1][1])

    def test_batch_sorted(self):
        sequences = [el.split() for el in \
                     ['I love it', 'My name is Earl', 'Whose line is it anyway', 'As you like it', 'Everybody']]
        words = list(set([el for lis in sequences for el in lis]))
        word_index = {wd: ix for ix, wd in enumerate(words)}
        index_word = {ix: wd for wd, ix in word_index.items()}
        word_index[data.PAD_TOKEN] = 170
        word_index[data.START_OF_VERSE_TOKEN] = 171
        word_index[data.END_OF_VERSE_TOKEN] = 172
        dataset, _ = data.batch(sequences, 2, word_index)
        self.assertEqual('My', index_word[dataset[0][0][1]])

    def test_pad_batch(self):
        sequences = [el.split() for el in ['I love it', 'My name is Earl']]
        padded = data.pad_batch(sequences)
        self.assertEqual(2, len(padded))
        self.assertTrue(all([len(el) == 4 for el in padded]))
        self.assertEqual(data.PAD_TOKEN, padded[0][-1])


    def test_join_texts(self):
        texts = ['something', 'else', 'is', 'here']
        self.assertEqual(
            '\n\nsomething else is here EOS',
            data.join_texts(texts, prompt='\n\n', separator=' ', eot_token='EOS')
        )

    def test_join_by_hierarchy(self):
        comments = """# language_name:        English
# closest_ISO_639-3:    eng
# ISO_15924:            Latn
# year_short:           1997
# year_long:            
# vernacular_title:     
# english_title:        World English Bible
# URL:                  http://biblehub.com/web/matthew/1.htm
# copyright_short:      Public Domain 1997
# copyright_long:       
# notes:                
"""
        comment_lines = comments.split('\n')
        lines = comment_lines + ['40001001\tFirst verse', '40001002\tSecond verse', '40002001\tNext chapter', '41001001\tAnother book',
                 '67001001\tAnother testament']
        bible = data.parse_pbc_bible_lines(lines, True, 'eng')
        _, by_testament, by_book, by_chapter, _ = bible.join_by_toc()
        self.assertTrue('old' not in by_testament)
        self.assertEqual('First verse_Second verse_Next chapter_Another book'.split('_'),
                         by_testament['new'])
        self.assertEqual(['Another testament'],
                         by_testament['apocryphal'])
        self.assertEqual('First verse_Second verse_Next chapter'.split('_'), by_book[40])
        self.assertEqual(['Another book'], by_book[41])
        self.assertEqual('Another testament'.split('_'), by_book[67])
        self.assertEqual('First verse_Second verse'.split('_'), by_chapter[40001])
        self.assertEqual('Next chapter'.split('_'), by_chapter[40002])
        self.assertEqual('Another book'.split('_'), by_chapter[41001])
        self.assertEqual('Another testament'.split('_'), by_chapter[67001])

    def test_to_dictionaries_repeated_commented_out_lines(self):
        comment_lines = []
        content_lines = [(1, "some lines", False),
                         (2, "some other line", False),
                         (3, "this line is commented out", True),
                         (3, "again but with the same ID", True),
                         (4, "some other line", False)]
        _, _, hidden_content = data.PbcBible.to_dictionaries(comment_lines, content_lines)
        self.assertEqual({3: 'again but with the same ID'}, hidden_content)


if __name__ == "__main__":
    unittest.main()
