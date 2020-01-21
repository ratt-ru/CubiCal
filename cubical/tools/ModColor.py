#   Copyright 2020 Jonathan Simon Kenyon
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

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
