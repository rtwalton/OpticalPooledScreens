from collections import defaultdict, Counter
import scipy.sparse
import numpy as np
import pandas as pd
import os
from Levenshtein import distance

import ops.utils
from ops.constants import *


# LOAD TABLES

def validate_design(df_design):
    for group, df in df_design.groupby('group'):
        x = df.drop_duplicates(['prefix_length', 'edit_distance'])
        if len(x) > 1:
            cols = ['group', DESIGN, 'prefix_length', 'edit_distance']
            error = 'multiple prefix specifications for group {0}:\n{1}'
            raise ValueError(error.format(group, df[cols]))
            
    return df_design


def load_gene_list(filename):
    return (pd.read_csv(filename, header=None)
     .assign(design=os.path.splitext(filename)[0])
     .rename(columns={0: GENE_ID})
    )


def validate_genes(df_genes, df_sgRNAs):
    missing = set(df_genes[GENE_ID]) - set(df_sgRNAs[GENE_ID])
    if missing:
        error = '{0} gene ids missing from sgRNA table: {1}'
        missing_ids = ', '.join(map(str, missing))
        raise ValueError(error.format(len(missing), missing_ids))

    duplicates = df_genes[[SUBPOOL, GENE_ID]].duplicated(keep=False)
    if duplicates.any():
        error = 'duplicate genes for the same subpool: {0}'
        xs = df_genes.loc[duplicates, [SUBPOOL, GENE_ID]].values
        raise ValueError(error.format(xs))

    return df_genes


def select_prefix_group(df_genes, df_sgRNAs):
    # doesn't shortcut if some genes need less guides
    prefix_length, edit_distance = (
        df_genes[[PREFIX_LENGTH, EDIT_DISTANCE]].values[0])

    return (df_sgRNAs
        .join(df_genes.set_index(GENE_ID), on=GENE_ID, how='inner')
        .pipe(select_guides, prefix_length, edit_distance)
        .sort_values([SUBPOOL, GENE_ID, RANK])
    #     # .set_index(GENE_ID)
    #     # .pipe(df_genes.join, on=GENE_ID, how='outer', rsuffix='_remove')
    #     # .pipe(lambda x: x[[c for c in x.columns if not c.endswith('_remove')]])
        .assign(selected_rank=lambda x: 
            ops.utils.rank_by_order(x, [SUBPOOL, GENE_ID]))
        .query('selected_rank <= sgRNAs_per_gene')
        .sort_values([SUBPOOL, GENE_ID, 'selected_rank'])
        .drop(['selected_rank'], axis=1)
    )


def select_guides(df_input, prefix_length, edit_distance):
    """`df_input` has gene_id, sgRNAs_per_gene
    """
    # TODO: NEEDS EDIT DISTANCE
    if edit_distance != 1:
        raise NotImplementedError('edit distance needs doing')

    selected_guides = (df_input
     .assign(prefix=lambda x: x['sgRNA'].str[:prefix_length])
     .pipe(lambda x: x.join(x[GENE_ID].value_counts().rename('sgRNAs_per_id'), 
         on=GENE_ID))
     .sort_values([RANK, 'sgRNAs_per_id'])
     .drop_duplicates('prefix')
     [SGRNA].pipe(list)
     )
    return df_input.query(loc('{SGRNA} == @selected_guides'))


# FILTER SGRNAS

def filter_sgRNAs(df_sgRNAs, homopolymer=5):
    cut = [has_homopolymer(x, homopolymer) or has_BsmBI_site(x) 
            for x in df_sgRNAs[SGRNA]]
    return df_sgRNAs[~np.array(cut)]

def has_homopolymer(x, n):
    a = 'A'*n in x
    t = 'T'*n in x
    g = 'G'*n in x
    c = 'C'*n in x
    return a | t | g | c
   
def has_BsmBI_site(x):
    x = 'CACCG' + x.upper() + 'GTTT'
    return 'CGTCTC' in x or 'GAGACG' in x


# OLIGOS

def build_sgRNA_oligos(df, dialout_primers, 
                        left='CGTCTCgCACCg', right='GTTTcGAGACG'):

    template = '{fwd}{left}{sgRNA}{right}{rev}'
    arr = []
    for s, d in df[[SGRNA, DIALOUT]].values:
        fwd, rev = dialout_primers[d]
        rev = reverse_complement(rev)
        oligo = template.format(fwd=fwd, rev=rev, sgRNA=s, 
                                left=left, right=right)
        arr += [oligo]
    return arr


def build_test(df_oligos, dialout_primers):
    """Pattern-match sgRNA cloning and dialout primers.
    """
    sites = 'CGTCTC', reverse_complement('CGTCTC')
    pat = ('(?P<dialout_fwd>.*){fwd}.CACCG'
           '(?P<sgRNA_cloned>.*)'
           'GTTT.{rev}(?P<dialout_rev>.*)')
    pat = pat.format(fwd=sites[0], rev=sites[1])

    kosuri = {}
    for i, (fwd, rev) in enumerate(dialout_primers):
        kosuri[fwd] = 'fwd_{0}'.format(i)
        kosuri[rev] = 'rev_{0}'.format(i)

    def validate_design(df):
        if not (df[VECTOR] == 'CROPseq').all():
            raise ValueError('can only validate CROPseq design')
        return df

    return (df_oligos
     .pipe(validate_design)
     .assign(sgRNA=lambda x: x['sgRNA'].str.upper())
     .assign(oligo=lambda x: x['oligo'].str.upper())
     .pipe(lambda x: pd.concat([x, x['oligo'].str.extract(pat)], axis=1))
     .assign(dialout_rev=lambda x: x['dialout_rev'].apply(reverse_complement))
     .assign(dialout_fwd_ix=lambda x: x['dialout_fwd'].apply(kosuri.get))      
     .assign(dialout_rev_ix=lambda x: x['dialout_rev'].apply(kosuri.get))            
     .assign(dialout_ix=lambda x: 
             x['dialout_fwd_ix'].str.split('_').str[1].astype(int))
    )


def validate_test(df_test):
    """Check sgRNA cloning and identiy of dialout primers.
    """
    assert df_test.eval('sgRNA_cloned == sgRNA').all()

    assert (df_test['dialout_fwd_ix'].str[-1] == 
            df_test['dialout_rev_ix'].str[-1]).all()

    assert df_test.eval('dialout_ix== dialout').all()

    print('Looking good!')

    return df_test


def reverse_complement(seq):
    watson_crick = {'A': 'T',
                'T': 'A',
                'C': 'G',
                'G': 'C',
                'U': 'A',
                'N': 'N'}

    watson_crick.update({k.lower(): v.lower() 
        for k, v in watson_crick.items()})

    return ''.join(watson_crick[x] for x in seq)[::-1]


# EXTERNAL

def import_brunello(filename):
    """Import "Brunello Library Target Genes", which can be found at:
    https://www.addgene.org/pooled-library/broadgpp-human-knockout-brunello/
    """
    columns = {'Target Gene ID': GENE_ID
              ,'Target Gene Symbol': GENE_SYMBOL
              ,'sgRNA Target Sequence': SGRNA
              , 'Rule Set 2 score': SGRNA_SCORE
              }

    def reassign_nontargeting(df):
        """Given non-targeting sgRNAs a gene ID of -1.
        """
        new_ids = []
        new_symbols = []
        for i, s in df[[GENE_ID, GENE_SYMBOL]].values:
            if s == 'Non-Targeting Control':
                new_ids.append(-1)
                new_symbols.append('nontargeting')
            else:
                new_ids.append(i)
                new_symbols.append(s)

        return df.assign(**{GENE_ID: new_ids, GENE_SYMBOL: new_symbols})

    df_brunello = (pd.read_csv(filename, sep='\t')
        .rename(columns=columns)
        .pipe(reassign_nontargeting)
        .pipe(ops.utils.cast_cols, int_cols=[GENE_ID])
        .assign(**{SGRNA_SCORE: lambda x: x[SGRNA_SCORE].fillna(0)})
        .assign(**{RANK: lambda x: 
            x.groupby(GENE_ID)[SGRNA_SCORE]
             .rank(ascending=False, method='first').astype(int)})
        [[GENE_ID, GENE_SYMBOL, RANK, SGRNA]]
        .sort_values([GENE_ID, RANK])
        )

    df_brunello.loc[df_brunello.gene_symbol=="AKAP2",'gene_id']=445815
    df_brunello.loc[df_brunello.gene_symbol=="PALM2",'gene_id']=445815
    df_brunello.loc[df_brunello.gene_symbol=="C10orf12",'gene_id']=84458
    df_brunello.loc[df_brunello.gene_symbol=="C10orf131",'gene_id']=387707
    df_brunello.loc[df_brunello.gene_symbol=="C16orf47",'gene_id']=463
    df_brunello.loc[df_brunello.gene_symbol=="C17orf47",'gene_id']=5414
    df_brunello.loc[df_brunello.gene_symbol=="C7orf76",'gene_id']=7979
    df_brunello.loc[df_brunello.gene_symbol=="MIA2",'gene_id']=4253
    df_brunello.loc[df_brunello.gene_symbol=="NARR",'gene_id']=83871
    df_brunello.loc[df_brunello.gene_symbol=="TMEM133",'gene_id']=143872
    df_brunello.loc[df_brunello.gene_symbol=="XAGE1B",'gene_id']=653067


    df_brunello = df_brunello.query('gene_symbol != ["C2orf48","TMEM257","TXNRD3NB"]').copy()

    return df_brunello

def import_brunello_dump(filename):
    df_brunello_dump = pd.read_csv(filename)

    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="AKAP2",'gene_id']=445815
    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="PALM2",'gene_id']=445815
    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="C16orf47",'gene_id']=463
    df_brunello_dump.loc[df_brunello_dump.gene_symbol=="C17orf47",'gene_id']=5414

    df_brunello_dump = df_brunello_dump.query('gene_symbol != ["C2orf48","TMEM257","TXNRD3NB"]').copy()

    return df_brunello_dump


def import_tkov3(filename, df_ncbi):    
    columns = {'GENE': GENE_SYMBOL, 'SEQUENCE': SGRNA}
    symbols_to_ids = df_ncbi.set_index(GENE_SYMBOL)[GENE_ID]
    # symbols_to_ids.index = symbols_to_ids.index.str.upper()
    df_tkov3 = (pd.read_excel(filename)
      .rename(columns=columns)
      [[GENE_SYMBOL, SGRNA]]
      )

    df_tkov3 = df_tkov3.query('gene_symbol!=["C2orf48","TMEM257","TXNRD3NB"]').copy()

    df_tkov3.loc[df_tkov3.gene_symbol=="ADC",'gene_symbol']="AZIN2"
    df_tkov3.loc[df_tkov3.gene_symbol=="AGPAT9",'gene_symbol']="GPAT3"
    df_tkov3.loc[df_tkov3.gene_symbol=="AIM1",'gene_symbol']="CRYBG1"
    df_tkov3.loc[df_tkov3.gene_symbol=="B3GNT1",'gene_symbol']="B4GAT1"
    df_tkov3.loc[df_tkov3.gene_symbol=="C11orf48",'gene_symbol']="LBHD1"
    df_tkov3.loc[df_tkov3.gene_symbol=="C15orf38",'gene_symbol']="ARPIN"
    df_tkov3.loc[df_tkov3.gene_symbol=="C2ORF15",'gene_symbol']="C2orf15"
    df_tkov3.loc[df_tkov3.gene_symbol=="C2orf47",'gene_symbol']="MAIP1"
    df_tkov3.loc[df_tkov3.gene_symbol=="C6ORF165",'gene_symbol']="C6orf165"
    df_tkov3.loc[df_tkov3.gene_symbol=="C7orf55",'gene_symbol']="FMC1"
    df_tkov3.loc[df_tkov3.gene_symbol=="CD97",'gene_symbol']="ADGRE5"
    df_tkov3.loc[df_tkov3.gene_symbol=="CXXC11",'gene_symbol']="RTP5"
    df_tkov3.loc[df_tkov3.gene_symbol=="FLJ27365",'gene_symbol']="MIRLET7BHG"
    df_tkov3.loc[df_tkov3.gene_symbol=="GIF",'gene_symbol']="CBLIF"
    df_tkov3.loc[df_tkov3.gene_symbol=="HN1L",'gene_symbol']="JPT2"
    df_tkov3.loc[df_tkov3.gene_symbol=="HN1",'gene_symbol']="JPT1"
    df_tkov3.loc[df_tkov3.gene_symbol=="KIAA1045",'gene_symbol']="PHF24"
    df_tkov3.loc[df_tkov3.gene_symbol=="NAT6",'gene_symbol']="NAA80"
    df_tkov3.loc[df_tkov3.gene_symbol=="NOV",'gene_symbol']="CCN3"
    df_tkov3.loc[df_tkov3.gene_symbol=="STRA13",'gene_symbol']="CENPX"
    df_tkov3.loc[df_tkov3.gene_symbol=="ZHX1-C8ORF76",'gene_symbol']="ZHX1-C8orf76"
    df_tkov3.loc[df_tkov3.gene_symbol=="MUM1",'gene_symbol']="PWWP3A"
    df_tkov3.loc[df_tkov3.gene_symbol=='CSRP2BP','gene_symbol'] = 'KAT14'
    df_tkov3.loc[df_tkov3.gene_symbol=='C10orf2','gene_symbol'] = 'TWNK'
    df_tkov3.loc[df_tkov3.gene_symbol=='AZI1','gene_symbol'] = 'CEP131'
    df_tkov3.loc[df_tkov3.gene_symbol=='EFTUD1','gene_symbol'] = 'EFL1'
    # df_tkov3[GENE_SYMBOL]=df_tkov3[GENE_SYMBOL].str.upper()
    return (df_tkov3
     .join(symbols_to_ids, on=GENE_SYMBOL, how='left')
     .assign(**{RANK: lambda x: ops.utils.rank_by_order(x, GENE_ID)})
    )

def import_CRISPRfocus(filename,df_ncbi):
    df_CRISPRfocus = pd.read_csv(filename)

    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="ZHX1-C8ORF76",'gene_symbol']="ZHX1-C8orf76"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="TGIF2-C20ORF24",'gene_symbol']="TGIF2-C20orf24"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="RPL17-C18ORF32",'gene_symbol']="RPL17-C18orf32"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="ANXA8L1",'gene_id']=np.nan
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="MUM1",'gene_symbol']="PWWP3A"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="NAT6",'gene_symbol']="NAA80"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="SLC35E2",'gene_symbol']="SLC35E2A"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="GIF",'gene_symbol']="CBLIF"
    df_CRISPRfocus.loc[df_CRISPRfocus.gene_symbol=="NOV",'gene_symbol']="CCN3"


    # df_CRISPRfocus_id = df_CRISPRfocus.dropna(subset=['gene_id'])
    # df_CRISPRfocus_na = df_CRISPRfocus[df_CRISPRfocus.gene_id.isna()]
    symbols_to_ids = df_ncbi.set_index(GENE_SYMBOL)[GENE_ID]

    # symbols_to_ids.index = symbols_to_ids.index.str.upper()
    # df_CRISPRfocus_na = (df_CRISPRfocus_na
    #                      .drop(columns=['gene_id'])
    #                      .join(symbols_to_ids, on=GENE_SYMBOL, how='left')
    #                     )
    df_CRISPRfocus = (df_CRISPRfocus
                         .drop(columns=['gene_id'])
                         .join(symbols_to_ids, on=GENE_SYMBOL, how='left')
                        )

    df_CRISPRfocus= df_CRISPRfocus.query('gene_symbol != ["FAM231C","C2orf48","TMEM257","TXNRD3NB","FAM25B"]').copy()

    # return pd.concat([df_CRISPRfocus_id,df_CRISPRfocus_na],sort=True)[['gene_symbol','gene_id','sgRNA','rank']]
    return df_CRISPRfocus[['gene_symbol','gene_id','sgRNA','rank']]

def import_wang2017(filename,df_ncbi):
    df_wang2017 = (pd.read_excel(filename)
                   .rename(columns={'sgRNA ID':'sgRNA_ID','sgRNA location':'sgRNA_location',
                    'Genomic strand targeted':'Genomic_strand','sgRNA sequence':'sgRNA',
                    'Other genes hits':'Other_gene_hits','Symbol':'gene_symbol'})
                  )

    def group_controls(s):
      if s.sgRNA_ID.startswith('CTRL'):
          s.gene_symbol = 'nontargeting'
      elif 'INTERGENIC' in s.sgRNA_ID:
          s.gene_symbol = 'INTERGENIC'
      return s

    df_wang2017 = df_wang2017.apply(lambda x: group_controls(x),axis=1)
    df_wang2017 = df_wang2017.query('gene_symbol != "INTERGENIC"').copy()

    df_wang2017.loc[df_wang2017.gene_symbol=="NOV",'gene_symbol']="CCN3"
    df_wang2017.loc[df_wang2017.gene_symbol=="GIF",'gene_symbol']="CBLIF"
    df_wang2017.loc[df_wang2017.gene_symbol=="B3GNT1",'gene_symbol']="B4GAT1"
    df_wang2017.loc[df_wang2017.gene_symbol=="C7orf55",'gene_symbol']="FMC1"
    df_wang2017.loc[df_wang2017.gene_symbol=="CXXC11",'gene_symbol']="RTP5"
    df_wang2017.loc[df_wang2017.gene_symbol=="AGPAT9",'gene_symbol']="GPAT3"
    df_wang2017.loc[df_wang2017.gene_symbol=="ZHX1-C8ORF76",'gene_symbol']="ZHX1-C8orf76"
    df_wang2017.loc[df_wang2017.gene_symbol=="AIM1",'gene_symbol']="CRYBG1"
    df_wang2017.loc[df_wang2017.gene_symbol=="NAT6",'gene_symbol']="NAA80"
    df_wang2017.loc[df_wang2017.gene_symbol=="CD97",'gene_symbol']="ADGRE5"
    df_wang2017.loc[df_wang2017.gene_symbol=="C15orf38",'gene_symbol']="ARPIN"
    df_wang2017.loc[df_wang2017.gene_symbol=="C2orf47",'gene_symbol']="MAIP1"
    df_wang2017.loc[df_wang2017.gene_symbol=="STRA13",'gene_symbol']="CENPX"
    df_wang2017.loc[df_wang2017.gene_symbol=="C11orf48",'gene_symbol']="LBHD1"
    df_wang2017.loc[df_wang2017.gene_symbol=="MUM1",'gene_symbol']="PWWP3A"
    df_wang2017.loc[df_wang2017.gene_symbol=="HN1L",'gene_symbol']="JPT2"
    df_wang2017.loc[df_wang2017.gene_symbol=="HN1",'gene_symbol']="JPT1"
    df_wang2017.loc[df_wang2017.gene_symbol=="ADC",'gene_symbol']="AZIN2"
    df_wang2017.loc[df_wang2017.gene_symbol=="TRIM49D2P",'gene_symbol']="TRIM49D2"
    df_wang2017.loc[df_wang2017.gene_symbol=="FAM21A",'gene_symbol']="WASHC2A"
    df_wang2017.loc[df_wang2017.gene_symbol=="SLC35E2",'gene_symbol']="SLC35E2A"
    df_wang2017.loc[df_wang2017.gene_symbol=="APITD1",'gene_symbol']="CENPS"
    df_wang2017.loc[df_wang2017.gene_symbol=="LIMS3L",'gene_symbol']="LIMS4"
    df_wang2017.loc[df_wang2017.gene_symbol=='CSRP2BP','gene_symbol'] = 'KAT14'
    df_wang2017.loc[df_wang2017.gene_symbol=='AZI1','gene_symbol'] = 'CEP131'
    df_wang2017.loc[df_wang2017.gene_symbol=='TCEB3C','gene_symbol'] = 'ELOA3'
    df_wang2017.loc[df_wang2017.gene_symbol=='TCEB3CL','gene_symbol'] = 'ELOA3B'
    df_wang2017.loc[df_wang2017.gene_symbol=='EFTUD1','gene_symbol'] = 'EFL1'
    df_wang2017.loc[df_wang2017.gene_symbol=='CGB','gene_symbol'] = 'CGB3'
    df_wang2017.loc[df_wang2017.gene_symbol=='C10orf2','gene_symbol'] = 'TWNK'

    df_wang2017 = df_wang2017.query(('gene_symbol != '
                                     '["CT45A4","SMCR9","PRAMEF3",'
                                     '"SPANXE","PRAMEF16","C2orf48",'
                                     '"TMEM257","TXNRD3NB","FOXD4L2","FAM25B"]'
                                    )
                                   ).copy()

    symbols_to_ids = df_ncbi.set_index('gene_symbol')['gene_id']
    # symbols_to_ids.index = symbols_to_ids.index.str.upper()
    # df_wang2017['gene_symbol']=df_wang2017['gene_symbol'].str.upper()

    df_wang2017 = (df_wang2017
                   .join(symbols_to_ids, on=['gene_symbol'], how='left')
                   .assign(**{RANK: lambda x: ops.utils.rank_by_order(x, 'gene_symbol')})
                  )

    def LOC_to_ID(s):
      if s.gene_symbol.startswith('LOC') & np.isnan(s.gene_id):
          s.gene_id = s.gene_symbol[3:]
      return s

    df_wang2017 = df_wang2017.apply(lambda x: LOC_to_ID(x),axis=1)
    df_wang2017.loc[df_wang2017.gene_symbol=="CRIPAK",'gene_id'] = 285464
    df_wang2017.loc[df_wang2017.gene_symbol=="FAM231A",'gene_id'] = 729574
    df_wang2017.loc[df_wang2017.gene_symbol=="KIAA1804",'gene_id'] = 84451
    df_wang2017.loc[df_wang2017.gene_symbol=="KIAA1045",'gene_id'] = 23349

    df_wang2017.loc[df_wang2017.gene_symbol == "nontargeting",'gene_id']=-1

    return df_wang2017[['gene_symbol','gene_id','sgRNA','rank']]


def import_hugo_ncbi(filename):
    columns = {'Approved symbol': GENE_SYMBOL,
               'NCBI Gene ID(supplied by NCBI)': GENE_ID}
    return (pd.read_csv(filename, comment='#', sep='\t')
         .rename(columns=columns)
         .dropna()
         .pipe(ops.utils.cast_cols, int_cols=[GENE_ID]))


def import_ncbi_synonyms(filename):
    return (pd.read_csv(filename,index_col=[0])
         .pipe(ops.utils.cast_cols, int_cols=[GENE_ID]))

def import_dialout_primers(filename):
    """Returns an array of (forward, reverse) primer pairs.
    """
    return pd.read_csv('kosuri_dialout_primers.csv').values