import pandas as pd
import pyranges as pr

def bins_to_genes_intersection(bins, genes, genes_cols=['seq_id', 'start', 'end']):
    """
    bins :: pandas.DataFrame : 
        columns: 
            Chromosome: `chr1`,
            Start: `10000`
            End:   `20000`
    genes :: pandas.DataFrame :
        columns in format of `gffpd.read_gff3`
    genes_cols :: list of strings with columns
        which should be put in PyRanges
    """
    format_cols = ['Chromosome', 'Start', 'End']
    buffer_bins = bins.copy()
    buffer_genes = genes.copy()
    buffer_bins.columns = format_cols
    buffer_genes = buffer_genes.rename(columns = dict(zip(genes_cols, format_cols)), 
                                       inplace = False)
    
    if 'chr' in buffer_genes['Chromosome'].apply(str).unique()[0]:
        pass
    else:
        buffer_genes['Chromosome'] = 'chr' + buffer_genes['Chromosome'].apply(str)
    
    if 'chr' in buffer_bins['Chromosome'].apply(str).unique()[0]:
        pass
    else:
        buffer_bins['Chromosome'] = 'chr' + buffer_bins['Chromosome'].apply(str)
        
    a = buffer_genes[format_cols]
    b = buffer_bins

    ar = pr.PyRanges(a)
    br = pr.PyRanges(b)

    result = ar.join(br)
    result = result.df.drop_duplicates()
    
    if len(result) == 0:
        print('Zero intersection result.')
    else:
        return result.merge(buffer_genes, left_on=format_cols, right_on=format_cols)