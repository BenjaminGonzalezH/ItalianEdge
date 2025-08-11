######### Libraries #########
import unittest                                     # Test interface.
from unittest.mock import patch                     # Replacement of objects.
import pandas as pd                                 # Dataframe managment.
import sys                                          # syscalls.
import io                                           # Input-Output 
from ParetoInsight_CPU.MappingEntrez import (
    chunks,
    query_mygene_chunk,
    ConvertToEntrezID
)

class TestGeneIDConversion(unittest.TestCase):

    def setUp(self):
        # Silent prints.
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    ########################## Tests ##########################
    def test_chunks_behavior(self):
        data = list(range(10))
        blocks = list(chunks(data, 3))
        self.assertEqual(blocks, [
            [0,1,2], [3,4,5], [6,7,8], [9]
        ])
        self.assertTrue(all(isinstance(b, list) for b in blocks))

    @patch("mygene.MyGeneInfo")
    def test_query_mygene_chunk_success(self, MockMG):
        mock_df = pd.DataFrame({
            'entrezgene': [111, 222],
            'notfound': [False, False]
        }, index=['GENE1', 'GENE2'])
        instance = MockMG.return_value
        instance.querymany.return_value = mock_df
        out = query_mygene_chunk(['GENE1','GENE2'], scopes=['symbol'], taxID=9606)
        self.assertEqual(list(out['entrezgene']), [111, 222])
        self.assertFalse(out['notfound'].any())

    @patch("mygene.MyGeneInfo")
    def test_query_mygene_chunk_notfound(self, MockMG):
        mock_df = pd.DataFrame({
            'notfound': [True]
        }, index=['FAKEGENE'])
        instance = MockMG.return_value
        instance.querymany.return_value = mock_df
        out = query_mygene_chunk(['FAKEGENE'], scopes=['symbol'], taxID=9606)
        self.assertTrue(out.empty)

    @patch("requests.exceptions.RequestException", new=Exception)
    @patch("mygene.MyGeneInfo")
    def test_query_mygene_chunk_request_exception(self, MockMG):
        instance = MockMG.return_value
        instance.querymany.side_effect = Exception("network")
        with self.assertRaises(RuntimeError):
            query_mygene_chunk(['GENE1'], scopes=['symbol'], taxID=9606)

    @patch("gprofiler.GProfiler")
    @patch("mygene.MyGeneInfo")
    def test_convert_to_entrez_id_priority(self, MockMG, MockGP):
        gp_instance = MockGP.return_value
        gp_instance.convert.return_value = pd.DataFrame({
            'incoming': ['GENE1'],
            'converted': ['101']
        })
        mg_instance = MockMG.return_value
        mg_instance.querymany.return_value = pd.DataFrame({
            'entrezgene': [202],
            'notfound': [False]
        }, index=['GENE2'])
        ids = ConvertToEntrezID(['GENE1', 'GENE2'], organism_gp='hsapiens', taxID=9606)
        self.assertEqual(ids[0], 'NA')
        self.assertEqual(ids[1], '202')

    @patch("gprofiler.GProfiler")
    @patch("mygene.MyGeneInfo")
    def test_convert_to_entrez_id_all_na(self, MockMG, MockGP):
        gp_instance = MockGP.return_value
        gp_instance.convert.return_value = pd.DataFrame({
            'incoming': ['GENE3'],
            'converted': [None]
        })
        mg_instance = MockMG.return_value
        mg_instance.querymany.return_value = pd.DataFrame({
            'entrezgene': [None],
            'notfound': [True]
        }, index=['GENE3'])
        ids = ConvertToEntrezID(['GENE3'], organism_gp='hsapiens', taxID=9606, na_value='NA')
        self.assertEqual(ids, ['NA'])

    def test_convert_to_entrez_id_empty(self):
        with self.assertRaises(ValueError):
            ConvertToEntrezID([], organism_gp='hsapiens')

if __name__ == "__main__":
    unittest.main()
