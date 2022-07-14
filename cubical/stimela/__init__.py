import os.path
from typing import Dict, Any
from omegaconf import OmegaConf
from dataclasses import dataclass

from scabha.cargo import Parameter

@dataclass
class JonesTemplateStructure:
    JONES_TEMPLATE: Dict[str, Parameter]

JonesTemplate = None

if JonesTemplate is None:
    _dirname = os.path.dirname(__file__)
    _structure = OmegaConf.create(JonesTemplateStructure)
    _config = OmegaConf.load(os.path.join(
        os.path.dirname(__file__),
        "schema_JONES_TEMPLATE.yaml"))
    JonesTemplate = OmegaConf.merge(_structure, _config).JONES_TEMPLATE

def make_stimela_schema(params: Dict[str, Any], inputs: Dict[str, Parameter], outputs: Dict[str, Parameter]):
    """Augments a schema for stimela based on solver.terms"""
    inputs = inputs.copy()

    terms = params.get('sol.jones', [])

    for jones in terms:
        for key, value in JonesTemplate.items():
            inputs[f"{jones.lower()}.{key}"] = value
        # inputs[f"{jones}.label"].default = jones

    return inputs, outputs
