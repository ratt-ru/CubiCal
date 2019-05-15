# CubiCal: a radio interferometric calibration suite
# (c) 2017 Rhodes University & Jonathan S. Kenyon
# http://github.com/ratt-ru/CubiCal
# This code is distributed under the terms of GPLv2, see LICENSE.md for details

# This module has been adapted from the DDFacet package,
# (c) Cyril Tasse et al., see http://github.com/saopicc/DDFacet
from __future__ import print_function
from six import string_types
import sys, re, optparse
from collections import OrderedDict

from . import parsets
from . import ModColor
from . import ClassPrint
from . import logger
log = logger.getLogger("dynoptparse")


#global Parset
#Parset=parsets.Parset("/media/tasse/data/DDFacet/Parset/DefaultParset.cfg")
#D=Parset.DicoPars 


class ForgivingParser(optparse.OptionParser):
    """A ForgivingParser is a version of OptionParser that ignores errors"""
    def exit(self, status=0, msg=None):
        pass

    def error(self, msg):
        self.error_msg = msg

    def print_help(self, file=None):
        pass


class DynamicOptionParser(object):
    def __init__(self,usage='Usage: %prog <options>', version='%prog version 1.0',
                 description="""Questions and suggestions: cyril.tasse@obspm.fr""",
                 defaults=None, attributes=None):
        self._parser_kws = dict(usage=usage, version=version, description=description)
        self._defaults = defaults or OrderedDict()
        self._attributes = attributes or {}
        # create None group for top-level options
        self._current_group_key = None
        self._current_group_opts = []
        self._groups = OrderedDict()
        self._groups[None] = None, self._current_group_opts
        # init
        self._init_from_defaults()
        self._options = self._arguments = None

    def _make_parser(self, parser_class=optparse.OptionParser):
        parser = parser_class(**self._parser_kws)
        for label, (title, option_list) in self._groups.items():
            # create group, unless label is None
            group = optparse.OptionGroup(parser, title) if label is not None else None
            # populate group, or else top level
            for option_names, options_kws in option_list:
                (group or parser).add_option(*option_names, **options_kws)
            # add group to parser, if defined
            if group is not None:
                parser.add_option_group(group)
        return parser

    def start_group(self, title, key=None):
        self._current_group_key = key or title
        self._current_group_opts = []
        self._groups[self._current_group_key] = title, self._current_group_opts
        #
        # if self.CurrentGroup is not None:
        #     self.FinaliseGroup()
        # self.CurrentGroup = OptParse.OptionGroup(self.opt, Name)
        # self.CurrentGroupKey = key or Name
        # self._groups[key] = Name, self.CurrentGroup

    def add_option(self, name, default, callback=None, callback_args=None):
        if name == "_Help":
            return
        # get optional attributes of this item
        attrs = self._attributes.get(self._current_group_key, {}).get(name, {})
        # items that are aliases of other items do not get their own option
        if attrs.get('alias_of'):
            return
        # if default is None:
        #     default = self.DefaultDict[self.CurrentGroupKey][name]
        opttype = attrs.get('type', str)
        choices = attrs.get('options', None)
        metavar = attrs.get('metavar', None) or choices or name.upper()
        action = "callback" if callback is not None else None
        if opttype is bool:
            opttype = str
            metavar = "0|1"
        if choices is not None:
            opttype = "choice"
        # handle doc string
        if 'doc' in attrs:
            help = attrs['doc']
            if '%default' not in help and not action:
                help += " (default: %default)"
        else:
            help = "(default: %default)"

        option_names = [ '--%s-%s' % (self._current_group_key, name) ]

        self._current_group_opts.append((
            option_names, dict(
                help=help, type=opttype, default=default, metavar=metavar,
                action=action, choices=choices, callback=callback, callback_args=callback_args,
                dest=self._form_dest_key(self._current_group_key, name))
        ))

    def _form_dest_key(self, GroupKey, Name):
        return "{}___{}".format(GroupKey or '', Name)

    def _parse_dest_key(self, KeyDest):
        return KeyDest.split("___",1)

    def read_input(self):
        # make a forgiving parser, which ignores all errors
        # this is to allow dynamic options to take effect
        parser = self._make_parser(ForgivingParser)
        self._options, self._arguments = parser.parse_args()

        # now make a proper parser, and parse the command line again
        parser = self._make_parser()
        self._options, self._arguments = parser.parse_args()
        # propagate results back into defaults dict
        for key, value in vars(self._options).items():
            group, name = self._parse_dest_key(key)
            group_dict = self._defaults[group]
            attrs = self._attributes.get(group, {}).get(name, {})
            if isinstance(value, string_types):
                value, _ = parsets.parse_config_string(value, name=name, extended=False, type=attrs.get('type'))
            group_dict[name] = value
            alias = attrs.get('alias') or attrs.get('alias_of')
            if alias:
                group_dict[alias] = value

        return self._defaults

    def get_arguments(self):
        return self._arguments

    def get_config(self):
        return self._defaults

    def write_to_parset(self, parset_filename):
        with open(parset_filename, "w") as f:
            for group, group_dict in self._defaults.items():
                f.write('[{}]\n'.format(group))
                for name, value in group_dict.items():
                    attrs = self._attributes.get(group, {}).get(name, {})
                    if not attrs.get('cmdline_only') and not attrs.get('alias_of'):
                        f.write('{} = {}\n'.format(name, value))
                f.write('\n')

    def print_config(self, skip_groups=[], dest=sys.stdout):
        P = ClassPrint.ClassPrint(HW=50)
        print(ModColor.Str(" Selected Options:"), file=dest)
        for group, group_dict in self._defaults.items():
            if group in skip_groups or '_NameTemplate' in group_dict:
                continue
    
            title = self._groups.get(group, (group, None))[0]
            print(ModColor.Str("[{}] {}".format(group, title), col="green"), file=dest)
    
            for name, value in group_dict.items():
                if name[0] != "_":   # skip "internal" values such as "_Help"
                    attrs = self._attributes.get(group).get(name, {})
                    if not attrs.get('alias_of') and not attrs.get('cmdline_only') and not attrs.get('no_print'): # and V!="":
                        P.Print(name, value, dest=dest)
            print(file=dest)

    def _add_section(self, section, values, attrs):
        # "_Help" value in each section is its documentation string
        help = values.get("_Help", section)
        self.start_group(help, section)
        for name, value in values.items():
            if not name[0] == "_" and not attrs.get(name, {}).get("no_cmdline"):
                section_template = self._templated_sections.get((section, name))
                if section_template:
                    callback = self._instantiate_section_template_callback
                    callback_args = (section_template,)
                    # add to list of initial callbacks
                    self._initial_callbacks.append((value, section_template))
                else:
                    callback = callback_args = None
                self.add_option(name, value, callback, callback_args)

    def _instantiate_section_template_callback(self, option, opt_str, value, parser, section_template):
        # store value in parser
        if parser is not None:
            setattr(parser.values, option.dest, value)
        print("callback invoked for {}".format(value), file=log(2))
        # get template contents
        if isinstance(value, string_types):
            value = value.split(",")
        elif type(value) is not list:
            raise TypeError("list or string expected for {}, got {}".format(opt_str, type(value)))
        for num, label in enumerate(value):
            substitutions = dict(LABEL=label, NUM=num)
            # init values from templated section
            values = self._defaults[section_template].copy()
            # section name is templated
            section = values["_NameTemplate"].format(**substitutions).lower()
            if section in self._instantiated_sections:
                print("section {} already exists".format(section), file=log(2))
                continue
            # if section is already instatiated in the parset, update
            if section in self._defaults:
                values.update(self._defaults[section])
            # now put it into the defaults
            self._defaults[section] = values
            # also add attributes from the templated section
            attrs = self._attributes.setdefault(section, OrderedDict())
            attrs.update(self._attributes[section_template])
            # expand templated values
            for name in values.get("_OtherTemplates", "").split(":"):
                values[name] = values.get(name, "").format(**substitutions)
            # expunge internal variables and mark section as templated
            del values["_NameTemplate"]
            del values["_ExpandedFrom"]
            del values["_OtherTemplates"]
            values["_Templated"] = True
            attrs["_Templated"] = dict(no_cmdline=True, no_print=True)
            # add to parser
            print("adding section {}".format(section), file=log(2))
            self._add_section(section, values, attrs)

    def _init_from_defaults(self):
        """
        Populates internal group dictionaries from the defaults and attributes
        """
        self._templated_sections = {}
        self._initial_callbacks = []
        self._instantiated_sections = set()

        # split defaults into "regular" sections and "templated" sections
        normal_sections = []
        for section, values in self._defaults.items():
            if '_NameTemplate' in values:
                match = re.match("^--(.+)-(.+)$", values["_ExpandedFrom"])
                if not match:
                    raise ValueError("Unrecognized _ExpandedFrom item in [{}]".format(section))
                sec, var = match.groups()
                self._templated_sections[sec, var] = section
            else:
                normal_sections.append((section, values))

        # create options based on contents of parset
        for section, values in normal_sections:
            # templated sections are skipped until they get instantiated from their template option
            if not values.get('_Templated'):
                self._add_section(section, values, self._attributes.get(section, {}))

        # call accumulated callbacks which will create templated sections
        for (value, section_template) in self._initial_callbacks:
            self._instantiate_section_template_callback(None, None, value, None, section_template)




def test():
    OP = MyOptParse()
    
    OP.start_group("* Data","Data")
    OP.add_option('MSName', 'foo.ms')
    OP.add_option('ColName', 'COLUMN')
    config = OP.read_input()

    return config
    

if __name__=="__main__":
    test()

