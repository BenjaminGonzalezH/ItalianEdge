# Import.
from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.anno.gaf_reader import GafReader
from goatools.associations import read_ncbi_gene2go
from goatools.go_enrichment import GOEnrichmentStudy
import concurrent.futures

from goatools.obo_parser import GODag
from goatools.anno.genetogo_reader import Gene2GoReader
from goatools.go_enrichment import GOEnrichmentStudy

def enrich_go(gene_list, obo_file='go-basic.obo', gene2go_file='gene2go', taxid=3702):
    godag = GODag(obo_file)
    gene2go = Gene2GoReader(gene2go_file, taxids=[taxid]).get_id2gos("BP")
    
    goea = GOEnrichmentStudy(
        gene_list,
        gene2go,
        godag,
        methods=['fdr_bh']
    )
    
    results = goea.run_study(gene_list)
    
    significant_results = [
        (res.GO, res.name, res.p_fdr_bh, res.study_items) 
        for res in results if res.p_fdr_bh < 0.05
    ]
    
    return significant_results

