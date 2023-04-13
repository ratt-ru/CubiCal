#!/usr/bin/env python
import sys
import os.path

from cubical.tools import parsets
from scabha.cargo import Parameter
from omegaconf import OmegaConf, DictConfig
from dataclasses import make_dataclass
from typing import Dict

if __name__ == "__main__":
    filename = sys.argv[1] if len(sys.argv) > 1 else "../DefaultParset.cfg"
    output_name = sys.argv[2] if len(sys.argv) > 2 else "schema.yaml"

    print(f"Reading {filename}")

    default_parset = parsets.Parset(filename)

    base_config = OmegaConf.create()

    for section, content in default_parset.value_dict.items():
        attrs = default_parset.attr_dict[section]

        section = section.replace("-", "_")

        section_config = OmegaConf.create()

        for name, value in content.items():
            if not name.startswith("_"):
                attr = OmegaConf.create(attrs[name])
                param = dict(
                    info=attr.doc.replace("\n", " "), 
                    required=False)
                    # default=value, required=False)
                if hasattr(attr, 'type'):
                    param['dtype'] = attr.type.__name__
                section_config[name] = param

        # insert into main config
        if section not in {"g", "de"}:
            base_config[section] = section_config


    # tweak the schemas
    tweaks = OmegaConf.load("schema_tweaks.yaml")
    base_config = OmegaConf.merge(base_config, tweaks)    

    jones_config = OmegaConf.create({"JONES_TEMPLATE": base_config.JONES_TEMPLATE})
    del base_config["JONES_TEMPLATE"]

    # delete "fake" parameters
    del jones_config.JONES_TEMPLATE["label"]
    del base_config.misc["parset-version"]

    output_base = os.path.splitext(output_name)[0]
    OmegaConf.save(jones_config, f"{output_base}_JONES_TEMPLATE.yaml")
    OmegaConf.save(base_config, output_name)


    print(f"Saved schema to {output_name}, loading back")

    # read config as structured schema
    structured = make_dataclass("CubiCalConfig",
        [(name.replace("-", "_"), Dict[str, Parameter]) for name in default_parset.sections]
    )
    structured = OmegaConf.create(structured)
    new_config = OmegaConf.load("schema.yaml")
    new_config = OmegaConf.merge(structured, new_config)

    OmegaConf.save(new_config, sys.stdout)

