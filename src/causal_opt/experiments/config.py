from dataclasses import dataclass,field,asdict
from pathlib import Path
import yaml

@dataclass
class ExperimentConfig:
    case:int
    seed:int=1
    n:int=1000
    d:int=10
    s0:int=10
    p:int=0
    k:int=0
    simulation:dict=field(default_factory=dict)
    method:dict=field(default_factory=dict)
    baseline:dict=field(default_factory=dict)

    @classmethod
    def load(cls,path,**overrides):
        raw=yaml.safe_load(Path(path).read_text()) or {}; raw.update(overrides); return cls(**raw)
    def to_dict(self): return asdict(self)
