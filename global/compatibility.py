import sys
import types

pseudo_lilac_embed_module = types.ModuleType("embed")
sys.modules["embed"] = pseudo_lilac_embed_module

class DummyClass:
    pass

pseudo_lilac_embed_module.LILaCDocument = DummyClass  # type: ignore
pseudo_lilac_embed_module.ProcessedComponent = DummyClass  # type: ignore
