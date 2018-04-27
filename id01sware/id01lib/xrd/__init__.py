"""
    Thus submodule provides classes and functions for conversion
    of ID01-data to reciprocal space. It is therefore a layer between
    id01lib (id01h5) and xrayutilities.

    Though specific for ID01, it is written in a way to allow easy
    definition of new geometries and detectors in order to be
    useful for other/future setups.

    The goal is that the user does not have to deal with the
    different inputs to xrayutilities as they can be read out from
    scan headers.
"""
from . import detectors
from . import geometries
from . import qconversion