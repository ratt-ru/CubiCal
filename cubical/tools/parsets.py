# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet

import configparser
from collections import OrderedDict
import re

def test():
    P=Parset()

### dead code?
# def FormatDico (DicoIn):
#     Dico=OrderedDict()
#     for key in DicoIn.keys():
#         Dico[key] = ParseConfigString(DicoIn[key])
#     return Dico


def parse_as_python(string, allow_builtins=False, allow_types=False):
    """Tries to interpret string as a Python object. Returns value, or string itself if unsuccessful.
    Names of built-in functions are _not_ interpreted as functions!
    """
    try:
        value = eval(string, {}, {})
        if type(value) is type(all) and not allow_builtins:  # do not interpret built-in function names
            return string
        if type(value) is type(int) and not allow_types:  # do not interpret built-in function names
            return string
        return value
    except:
        return string

def parse_config_string(string, name='config', extended=True, type=None):
    """
    Parses configuration string, converting it to a Python object (i.e. bool, int, float, etc.).
    Can also create a dict of attributes formed as as described below.

    Config string can contain an optional embedded comment, preceded by a '#' symbol. This
    is converted into a docstring, and returned as attrs['doc']. Docstrings can contain optional
    #ATTRIBUTE:VALUE entries. These are extracted and removed from the docstring, and returned
    in the attrs dict.

    The 'type' attribute has special meaning. If present, it forces the string to be parsed
    as a specific type. This overrides the 'type' argument.

    If extended=False, then it DOESN'T attempt to parse out the docstring and attributes,
    but simply tries to interpret the object as the given type. This mode is suitable to
    parsing command-line arguments.

    Args:
        string:     string value to parse
        name:       name of option (used for error messages)
        extended:   if True, enables parsing of docstring and attributes
        type:       forces string to be interpreted as a specific type

    Returns:
        tuple of value, attribute_dict
    """
    if string is None:
        return None, {}

    attrs = {}
    if extended:
        # parse out docstring
        if "#" in string:
            string, docstring = string.split("#", 1)
            string = string.strip()
            # parse out attributes
            while True:
                docstring = docstring.strip()
                # find instance of #attr:value in docstring
                match = re.match("(.*)#(\w+):([^\s]*)(.*)", docstring, re.DOTALL)
                if not match:
                    break
                # extract attribute
                attrname, value = match.group(2), match.group(3)
                # #options:value, value is always treated as a string. Otherwise, treat as Python expression
                if attrname not in ("options",):
                    value = parse_as_python(value)
                attrs[attrname] = value
                # remove it from docstring
                docstring = match.group(1) + match.group(4)
        else:
            docstring = ""
        attrs["doc"] = docstring

        # if attributes contain a type, parse this out
        if 'type' in attrs:
            if attrs['type'] == 'string':
                attrs['type'] = str
            # type had better be a callable type object
            attrs['type'] = type = parse_as_python(attrs['type'], allow_types=True)
            if not callable(type):
                raise ValueError("%s: invalid '#type:%s' attribute"%(name, attrs['type']))

    # if attributes contain an option list, enforce this
    if 'options' in attrs:
        attrs['options'] = opts = attrs['options'].split("|")
        if string not in opts:
            raise ValueError("%s: value %s not in options list"%(name, string))

    # make sure _Help is interpreted as a string
    if name == "_Help":
        return string, attrs

    # interpret explicit types
    if type:
        # make sure None string is still None
        if type is str and string == "None" or string == "none":
            return None, attrs
        # make sure False/True etc. are interpreted as booleans
        if type is bool:
            return bool(parse_as_python(string)), attrs
        return type(string), attrs

    # Now, some kludges for backward compatibility
    # A,B,C and [A,B,C] are parsed to a list
    as_list = len(string)>1 and string[0] == '[' and string[-1] == ']'
    if as_list:
        string = string[1:-1]
    if as_list or "," in string:
        return [ parse_as_python(x) for x in string.split(",") ], attrs

    # Otherwise just interpret the value as a Python object if possible
    return parse_as_python(string), attrs


class Parset():
    def __init__(self, filename=None):
        """Creates parset, reads from file if specified"""
        self.value_dict = self.DicoPars = OrderedDict()   # call it DicoPars for compatibility with old testing code
        self.attr_dict = OrderedDict()
        if filename:
            self.read(filename)

    def update_values (self, other, other_filename=''):
        """Updates this Parset with keys found in other parset.
        other_filename is only needed for error messages."""
        for secname, secvalues in other.value_dict.items():
            if secname in self.value_dict:
                for name, value in secvalues.items():
                    attrs = self.attr_dict[secname].get(name)
                    if attrs is None:
                        attrs = self.attr_dict[secname][name] = \
                            other.attr_dict[secname].get(name, {})
                    if attrs.get('cmdline_only'):
                        continue
                    # check value for type and options conformance
                    if 'type' in attrs:
                        try:
                            value = attrs['type'](value)
                        except:
                            raise TypeError("invalid [{}] {}={} setting{}".format(
                                            secname, name, value, other_filename))
                    if 'options' in attrs and value not in attrs['options']:
                        if str(value) in attrs['options']:
                            value = str(value)
                        else:
                            raise TypeError("invalid [{}] {}={} setting{}".format(
                                            secname, name, value, other_filename))
                    self.value_dict[secname][name] = value
                    # make sure aliases get copied under both names
                    alias = attrs.get('alias') or attrs.get('alias_of')
                    if alias:
                        self.value_dict[secname][alias] = value
            else:
                self.value_dict[secname] = secvalues
                self.attr_dict[secname] = other.attr_dict[secname]

    def read (self, filename, default_parset=False):
        """Reads parset from filename.
        default_parset: if True, this is treated as the default parset, and things like templated
        section names are expanded.
        """
        self.filename = filename
        self.Config = config = configparser.ConfigParser(dict_type=OrderedDict)
        config.optionxform = str
        success = config.read(self.filename)
        self.success = bool(len(success))
        if self.success:
            self.sections = config.sections()
            for section in self.sections:
                self.value_dict[section], self.attr_dict[section] = self.read_section(config, section)
        # # now migrate from previous versions
        # self.version = self.value_dict.get('Misc', {}).get('ParsetVersion', 0.0)
        # if self.version != 0.1:
        #     self._migrate_ancient_0_1()
        #     self.migrated = self.version
        #     self.version = self.value_dict['Misc']['ParsetVersion'] = 0.1
        # else:
        #     self.migrated = None
        # # if "Mode" not in self.value_dict["Output"] and "Mode" in self.value_dict["Image"]:
        # #     self.value_dict["Output"]["Mode"] = self.value_dict["Image"]["Mode"]
        # #     del self.value_dict["Image"]["Mode"]

    def read_section(self, config, section):
        """Returns two dicts corresponding to the given section: a dict of option:value,
        and a dict of option:attribute_dict"""
        dict_values = OrderedDict()
        dict_attrs = OrderedDict()
        for option in config.options(section):
            strval = config.get(section, option)
            # option names with an "|" in them specify a longhand alias
            if "|" in option:
                option, alias = option.split("|",1)
            else:
                alias = None
            dict_values[option], dict_attrs[option] = parse_config_string(strval, name=option)
            # if option has an alias, mke copy of both in dicts
            if alias:
                dict_attrs[option]['alias'] = alias
                dict_values[alias] = dict_values[option]
                dict_attrs[alias] = { 'alias_of': option }
        return dict_values, dict_attrs

    def set (self, section, option, value):
        self.value_dict.setdefault(section,{})[option] = value

    def write (self, f):
        """Writes the Parset out to a file object"""
        for section, content in self.value_dict.items():
            f.write('[%s]\n'%section)
            for option, value in content.items():
                attrs = self.attr_dict.get(section, {}).get(option, {})
                if option[0] != "_" and not attrs.get('cmdline_only') and not attrs.get('alias_of'):
                    f.write('%s = %s \n'%(option, str(value)))
            f.write('\n')

    def _makeSection (self, section):
        """
        Helper method for migration: makes a new section
        """
        for dd in self.value_dict, self.attr_dict:
            dd.setdefault(section, OrderedDict())
        return section

    def _renameSection (self, oldname, newname):
        """
        Helper method for migration: renames a section. If the new section already exists, merges options into
        it.
        """
        for dd in self.value_dict, self.attr_dict:
            if oldname in dd:
                dd.setdefault(newname, OrderedDict()).update(dd.pop(oldname))
        return newname

    def _del (self, section, option):
        """
        Helper method for migration: removes an option
        """
        for dd in self.value_dict, self.attr_dict:
            if section in dd and option in dd[section]:
                dd[section].pop(option)

    def _rename (self, section, oldname, newname):
        """
        Helper method for migration: renames an option within a section. Optionally remaps option values using
        the supplied dict.
        """
        for dd in self.value_dict, self.attr_dict:
            if section in dd and oldname in dd[section]:
                dd[section][newname] = dd[section].pop(oldname)

    def _remap (self, section, option, remap):
        """
        Helper method for migration: remaps the values of an option
        """
        if section in self.value_dict and option in self.value_dict[section]:
            value = self.value_dict[section][option]
            if value in remap:
                self.value_dict[section][option] = remap[value]

    def _move (self, oldsection, oldname, newsection, newname):
        """
        Helper method for migration: moves an option to a different section
        """
        for dd in self.value_dict, self.attr_dict:
            if oldsection in dd and oldname in dd[oldsection]:
                dd.setdefault(newsection, OrderedDict())[newname] = dd[oldsection].pop(oldname)

