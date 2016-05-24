import pyrap.tables as pt
import numpy as np
from collections import Counter, OrderedDict
from time import time
import os
import argparse

def masseuse(ms_name):

    new_ms_name = ms_name + "_sorted"
    os.system("cp -r {0} {1}".format(ms_name, new_ms_name))
        
    ms = pt.table(new_ms_name, readonly=False)

    sorted_data = pt.taql("select from [select from $ms where ANTENNA1!=ANTENNA2] orderby TIME, ANTENNA1, ANTENNA2")

    nant = pt.table(new_ms_name + "::ANTENNA").nrows()
    
    expected_pairings = set(pair_ants(nant))

    times = sorted_data.getcol("TIME") 
    unique_times = np.unique(times)    
    anta = sorted_data.getcol("ANTENNA1")
    antb = sorted_data.getcol("ANTENNA2")
    tdict = OrderedDict(sorted(Counter(times).items()))
        
    tind = [0]
    tind.extend(tdict.values())
    tind = list(np.cumsum(tind))

    missing_count = 0
    missing_anta = []
    missing_antb = []
    missing_time = []

    for i in xrange(len(tind) - 1):

        ftind = tind[i] 
        ltind = tind[i+1]

        t = times[ftind]

        current_pairings = set(zip(anta[ftind:ltind], antb[ftind:ltind]))
        missing_pairings = expected_pairings - current_pairings

        missing_count += len(missing_pairings)

        for a, b in list(missing_pairings):
            missing_anta.append(a)
            missing_antb.append(b)
            missing_time.append(t)

    if missing_count == 0:
        print "All baselines present!"
        return
    else:
        print "{} rows missing!".format(missing_count)

    ms.addrows(missing_count)
    
    special_cols = ["TIME", "ANTENNA1", "ANTENNA2", "FLAG_ROW"]

    t_col = ms.getcol("TIME")
    aa_col = ms.getcol("ANTENNA1")
    ab_col = ms.getcol("ANTENNA2")
    fr_col = ms.getcol("FLAG_ROW")

    t_col[-missing_count:] = missing_time
    aa_col[-missing_count:] = missing_anta
    ab_col[-missing_count:] = missing_antb
    fr_col[-missing_count:] = 1
   
    indices = np.lexsort((ab_col, aa_col, t_col))    

    sorted_t_col = t_col[indices]
    sorted_aa_col = aa_col[indices]
    sorted_ab_col = ab_col[indices]
    sorted_fr_col = fr_col[indices]

    ms.putcol("TIME", sorted_t_col)
    ms.putcol("ANTENNA1", sorted_aa_col)
    ms.putcol("ANTENNA2", sorted_ab_col)
    ms.putcol("FLAG_ROW", sorted_fr_col)

    for i in ms.colnames():
        if i not in special_cols:
            print "Processing {} column...".format(i)         
            ms_col = ms.getcol(i, nrow=missing_count)
            ms_col[:] = 0
            ms.putcol(i, ms_col, startrow=(ms.nrows() - missing_count), nrow=missing_count)
            ms_col = ms.getcol(i)[indices]
            ms.putcol(i, ms_col)
                       


def pair_ants(nant):

    pairs = []

    for i in range(nant - 1):
        for j in range(i + 1, nant):
            pairs.append((i,j))
            
    return pairs

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Fix missing baselines in MS')
    parser.add_argument('msname', type=str, help='MS name/destination')
    args = parser.parse_args()
    
    t0 = time()
    masseuse(args.msname)
    print "MS took {} seconds to process.".format(time() - t0)
