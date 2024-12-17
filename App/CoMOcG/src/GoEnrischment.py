# Import
import gzip
import os
from Bio import Entrez


def get_entrez_id(gene_symbol, mail, taxonomy):
    Entrez.email = mail
    
    # Search the NCBI Gene database with the gene symbol
    handle = Entrez.esearch(db="gene", term=gene_symbol+"[Gene] AND txid"+str(taxonomy), retmode="xml")
    record = Entrez.read(handle)
    handle.close()

    # Check if results were found
    if record["Count"] == "0":
        print(f"No Entrez ID found for {gene_symbol}")
        return None
    else:
        # Extract the Entrez ID
        entrez_id = record["IdList"][0]
        return entrez_id
    
def get_GOannotation_fromID(entrez_id, mail):
    
    Entrez.email = mail
    fetch_handle = Entrez.efetch(db="gene", id=entrez_id, retmode="xml")
    gene_record = Entrez.read(fetch_handle)
    fetch_handle.close()

    go_annotations = []
    for feature in gene_record[0]['Entrezgene_locus']:
        if 'Gene-commentary_products' in feature:
            for product in feature['Gene-commentary_products']:
                if 'Gene-commentary_comment' in product:
                    for comment in product['Gene-commentary_comment']:
                        if 'Gene-commentary_type' in comment and comment['Gene-commentary_type'] == 'GO':
                            go_annotations.append(comment['Gene-commentary_text'])
    return go_annotations

