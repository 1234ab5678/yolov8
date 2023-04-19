from .model import mobile_vit_small
from .model import mobile_vit_x_small
from .model import mobile_vit_xx_small

get_model_from_name = {
    "mobile_vit"               : mobile_vit_small,
    "mobile_vit_x"                  : mobile_vit_x_small,
    "mobile_vit_xx"                  : mobile_vit_xx_small,
}