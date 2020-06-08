# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

import sys, os, os.path, pickle, re
from . import NpShared
import numpy as np
import traceback
import collections

SHM_PREFIX = "/dev/shm/"
SHM_PREFIX_LEN = len(SHM_PREFIX)

def _to_shm (path):
    """Helper function, converts /dev/shm/name to shm://name"""
##    return "shm://" + path[SHM_PREFIX_LEN:]
    # it seems that shm_open() does not support subdirectories. open("/dev/shm/a") should have
    # the same effect though (even if it is Linux-specific), so use that instead
    return "file://" + path

_allowed_key_types = dict(int=int, str=str, bool=bool)

def attach(name, load=True, readwrite=True):
    return SharedDict(name, reset=False, load=load, readwrite=readwrite)

def create(name):
    return SharedDict(name, reset=True)

def dict_to_shm(name, D):
    Ds=create(name)
    for key in list(D.keys()):
        Ds[key]=D[key]
    return Ds


class SharedDictRepresentation(object):
    def __init__(self, path, readwrite, load):
        self.path = path
        self.readwrite = readwrite
        self.load = load

    def instantiate(self):
        return SharedDict(self.path, reset=False, readwrite=self.readwrite, load=self.load)


class SharedDict (dict):
    basepath = SHM_PREFIX

    class ItemLoadError(RuntimeError):
        """Exception that captures item loading errors"""
        def __init__(self, path, exc_info):
            RuntimeError.__init__(self, "Error loading SharedDict item %s"%path)
            self.path = path
            self.exc_info = exc_info

    class ItemProxy(object):
        """Base class for helper class used to defer loading of ShareDict items until they are actually requested"""
        def __init__(self, path):
            self.path = path
        def load(self):
            try:
                return self.load_impl()
            except:
                print("Error loading item %s" % self.path)
                traceback.print_exc()
                return SharedDict.ItemLoadError(path, sys.exc_info())

    class SharedArrayProxy (ItemProxy):
        def load_impl(self):
            return NpShared.GiveArray(_to_shm(self.path))

    class SubdictProxy(ItemProxy):
        def load_impl(self):
            return SharedDict(path=self.path, reset=False)

    class PickleProxy(ItemProxy):
        def load_impl(self):
            with open(self.path, 'rb') as pf:
                return pickle.load(pf)

    # this maps "class codes" parsed out of item filenames to appropriate item proxies. See reload() below
    _proxy_class_map = dict(a=SharedArrayProxy, d=SubdictProxy, p=PickleProxy)

    @staticmethod
    def setBaseName(name):
        SharedDict.basepath = os.path.join(SHM_PREFIX, name)
        if not os.path.exists(SharedDict.basepath):
            os.mkdir(SharedDict.basepath)

    def __init__ (self, path, reset=True, load=True, readwrite=True):
        dict.__init__(self)
        self._delete_items = False
        self._readwrite = readwrite
        self._load = load
        if path.startswith(SharedDict.basepath):
            self.path = path
        else:
            self.path = os.path.join(SharedDict.basepath, path)
#        self._path_fd = os.open(self.path, os.O_RDONLY)  # for sync purposes
        if reset or not os.path.exists(self.path):
            self.delete()
        elif load:
            self.reload()

    def __del__(self):
 #       os.close(self._path_fd)
        if self._delete_items:
            self.delete()

    def readwrite(self):
        if not self._load:
            raise RuntimeError("SharedDict %s attached without load permissions" % self.path)
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        return SharedDictRepresentation(self.path, readwrite=True, load=True)

    def readonly(self):
        if not self._load:
            raise RuntimeError("SharedDict %s attached without load permissions" % self.path)
        return SharedDictRepresentation(self.path, readwrite=False, load=True)

    def writeonly(self):
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        return SharedDictRepresentation(self.path, readwrite=True, load=False)

    def delete(self):
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        dict.clear(self)
        if os.path.exists(self.path):
            os.system("rm -fr %s" % self.path)
        os.mkdir(self.path)

    def clear(self):
        if self._delete_items:
            if not self._readwrite:
                raise RuntimeError("SharedDict %s attached as read-only" % self.path)
            self.delete()
        else:
            dict.clear(self)

    def save(self, filename):
        os.system("tar cf %s -C %s ." % (filename, self.path))

    def restore(self, filename):
        self.delete()
        os.system("tar xf %s -C %s" % (filename, self.path))
        self.reload()

    def reload(self):
        """(Re)initializes dict with items from path"""
        if not self._load:
            raise RuntimeError("SharedDict %s attached without load permissions" % self.path)
        dict.clear(self)
        # scan our subdirectory for items
        for name in os.listdir(self.path):
            filepath = os.path.join(self.path, name)
            # each filename is composed as "key_type:name:value_type", e.g. "str:Data:a", where value_type
            # is looked up in _proxy_class_map to determine how to load the file
            match = re.match("^(\w+):(.*):(%s)$" % "|".join(list(SharedDict._proxy_class_map.keys())), name)
            if not match:
                print("Can't parse shared dict entry " + filepath)
                continue
            keytype, key, valuetype = match.groups()
            typefunc = _allowed_key_types.get(keytype)
            if typefunc is None:
                print("Unknown shared dict key type "+keytype)
                continue
            key = typefunc(key)
            try:
                proxyclass = SharedDict._proxy_class_map[valuetype]
                dict.__setitem__(self, key, proxyclass(filepath))
            except:
                print("Error loading item %s"%name)
                traceback.print_exc()
                pass

    def _key_to_name (self, item):
        return "%s:%s:" % (type(item).__name__, str(item))

    def get(self, item, default_value=None):
        value = dict.get(self, item, default_value)
        if isinstance(value, SharedDict.ItemProxy):
            value = value.load()
            dict.__setitem__(self, item, value)
        return value

    def __getitem__(self, item):
        value = dict.__getitem__(self, item)
        if isinstance(value, SharedDict.ItemProxy):
            value = value.load()
            dict.__setitem__(self, item, value)
        return value

    def iteritems(self):
        raise RuntimeError("not implemented")

    def itervalues(self):
        raise RuntimeError("not implemented")

    def items(self):
        raise RuntimeError("not implemented")

    def values(self):
        raise RuntimeError("not implemented")

    def __delitem__(self, item):
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        if self._delete_items:
            return self.delete_item(item)
        else:
            return dict.__delitem__(self, item)

    def delete_item (self, item):
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        dict.__delitem__(self, item)
        name = self._key_to_name(item)
        path = os.path.join(self.path, name)
        for suffix in "ap":
            if os.path.exists(path+suffix):
                os.unlink(path+suffix)
        if os.path.exists(path+"d"):
            os.system("rm -fr "+path+"d")

    def set_item (self, item, value):
        """sets item, and returns it"""
        self.__setitem__(item, value)
        return dict.get(self, item)

    def __setitem__(self, item, value):
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        if type(item).__name__ not in _allowed_key_types:
            raise KeyError("unsupported key of type "+type(item).__name__)
        name = self._key_to_name(item)
        path = os.path.join(self.path, name)
        # remove previous item from SHM, if it's in the local dict
        if dict.__contains__(self,item):
            for suffix in "ap":
                if os.path.exists(path+suffix):
                    os.unlink(path+suffix)
            if os.path.exists(path+"d"):
                os.system("rm -fr "+path+"d")
        # if item is not in local dict but is on disk, this is a multiprocessing logic error
        else:
            for suffix in "apd":
                if os.path.exists(path+suffix):
                    raise RuntimeError("SharedDict entry %s exists, possibly added by another process. This is most likely a bug!" % (path+suffix))

        # for arrays, copy to a shared array
        if isinstance(value, np.ndarray):
            value = NpShared.ToShared(_to_shm(path+'a'), value)
        # for regular dicts, copy across
        elif isinstance(value, (dict, SharedDict, collections.OrderedDict)):
            dict1 = self.addSubdict(item)
            for key1, value1 in value.items():
                dict1[key1] = value1
            value = dict1
        # all other types, just use pickle
        else:
            with open(path+'p', "wb") as pf:
                pickle.dump(value, pf, 2)
        dict.__setitem__(self, item, value)

    def addSubdict (self, item):
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        name = self._key_to_name(item) + 'd'
        filepath = os.path.join(self.path, name)
        subdict = SharedDict(filepath, reset=True)
        dict.__setitem__(self, item, subdict)
        return subdict

    def addSharedArray (self, item, shape, dtype):
        """adds a SharedArray entry of the specified shape and dtype"""
        if not self._readwrite:
            raise RuntimeError("SharedDict %s attached as read-only" % self.path)
        name = self._key_to_name(item) + 'a'
        filepath = os.path.join(self.path, name)
        array = NpShared.CreateShared(_to_shm(filepath), shape, dtype)
        dict.__setitem__(self, item, array)
        return array

SharedDict.setBaseName("cubical."+str(os.getpid()))

def testSharedDict ():
    dic = SharedDict("foo")
    dic['a'] = 'a'
    dic['b'] = (1,2,3)
    dic['c'] = np.array([1,2,3,4])
    subdict = dic.addSubdict('subdict')
    subdict['a'] = 'aa'
    subdict['b'] = ('a', 1, 2, 3)
    subdict['c'] = np.array([1, 2, 3, 4, 5, 6])
    subdict2 = subdict.addSubdict('subdict2')
    subdict2['a'] = 'aaa'

    arr = subdict.addSharedArray("foo",(4, 4), np.float32)
    arr.fill(1)

    print(dic)

    other_view = SharedDict("foo", reset=False)
    print(other_view)




