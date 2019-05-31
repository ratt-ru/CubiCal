# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
bold='\033[1m'
nobold='\033[0m'
Separator="================================%s=================================="
silent=0

_color_dict = dict(red=FAIL, green=OKGREEN, yellow=WARNING, blue=OKBLUE,
                   white="")

def disableColors():
    global silent
    silent = 1
    
def Str(strin0,col="red",Bold=True):
    if silent==1: return strin0
    strin=str(strin0)

    ss = _color_dict.get(col)
    if ss is None:
        raise ValueError("unknown color '{}'".format(col))

    ss="%s%s%s"%(ss,strin,ENDC)
    if Bold: ss="%s%s%s"%(bold,ss,nobold)
    return ss

def Sep(strin=None,D=1):
    if D!=1:
        return Str(Separator%("="*len(strin)))
    else:
        return Str(Separator%(strin))

def Title(strin,Big=False):
    print()
    print()
    if Big: print(Sep(strin,D=0))
    print(Sep(strin))
    if Big: print(Sep(strin,D=0))
    print()

def disable():
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''
