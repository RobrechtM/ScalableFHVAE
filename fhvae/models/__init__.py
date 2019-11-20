from .fhvae import FHVAE
from .reg_fhvaeL import RegFHVAEL
from .reg_fhvaeLW import RegFHVAELW
from .simple_fhvae import SimpleFHVAE
from .reg_fhvae import RegFHVAE

def load_model(name):
    if name == "fhvae":
        return FHVAE
    elif name == "reg_fhvaeL":
        return RegFHVAEL
    elif name == "reg_fhvaeLW":
        return RegFHVAELW
    elif name == "simple_fhvae":
        return SimpleFHVAE
    elif name == "reg_fhvae":
        return RegFHVAE
    else:
        raise ValueError
