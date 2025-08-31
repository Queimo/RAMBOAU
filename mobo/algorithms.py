from .mobo import MOBO

"""
High-level algorithm specifications by providing config
"""

class qNEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qnehvi",
        "selection": "identity",
    }
    
class qNEHVIdet(MOBO):
    config = {
        "surrogate": "botorchgpmean",
        "acquisition": "identity",
        "solver": "qnehvi",
        "selection": "identity",
    }

class RAqNEIRS(MOBO):
    config = {
        "surrogate": "botorchgprepeat",
        "acquisition": "identity",
        "solver": "raqneirs",
        "selection": "identity",
    }

class RAqNEIRSdet(MOBO):
    config = {
        "surrogate": "botorchgprepeatmean",
        "acquisition": "identity",
        "solver": "raqneirs",
        "selection": "identity",
    }

class RAqNEHVI(MOBO):
    config = {
        "surrogate": "botorchgprepeat",
        "acquisition": "identity",
        "solver": "raqnehvi",
        "selection": "identity",
    }

class RAqLogNEHVI(MOBO):
    config = {
        "surrogate": "botorchgprepeat",
        "acquisition": "identity",
        "solver": "raqlognehvi",
        "selection": "identity",
    }

class RAqLogNEHVIdet(MOBO):
    config = {
        "surrogate": "botorchgprepeatmean",
        "acquisition": "identity",
        "solver": "raqlognehvi",
        "selection": "identity",
    }

class qEHVI(MOBO):
    config = {
        "surrogate": "botorchgp",
        "acquisition": "identity",
        "solver": "qehvi",
        "selection": "identity",
    }


def get_algorithm(name):
    """
    Get class of algorithm by name
    """
    algo = {
        "qnehvi": qNEHVI,
        "qnehvidet": qNEHVIdet,
        "qehvi": qEHVI,
        "raqneirs": RAqNEIRS,
        "raqneirsdet": RAqNEIRSdet,
        "raqnehvi": RAqNEHVI,
        "raqlognehvi": RAqLogNEHVI,
        "raqlognehvidet": RAqLogNEHVIdet,
    }
    return algo[name]
