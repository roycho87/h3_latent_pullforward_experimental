from .diagnostic import NODE_CLASS_MAPPINGS as _c, NODE_DISPLAY_NAME_MAPPINGS as _cn
from .latent_editing import NODE_CLASS_MAPPINGS as _d, NODE_DISPLAY_NAME_MAPPINGS as _dn
from .context_motion_mask import NODE_CLASS_MAPPINGS as _e, NODE_DISPLAY_NAME_MAPPINGS as _en
from .motion_mask import NODE_CLASS_MAPPINGS as _a, NODE_DISPLAY_NAME_MAPPINGS as _an
from .h3_pullforward import NODE_CLASS_MAPPINGS as _b, NODE_DISPLAY_NAME_MAPPINGS as _bn

NODE_CLASS_MAPPINGS = {**_a, **_b, **_c, **_d, **_e}
NODE_DISPLAY_NAME_MAPPINGS = {**_an, **_bn, **_cn, **_dn, **_en}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
