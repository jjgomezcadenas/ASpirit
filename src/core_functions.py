from ic_core_functions import *

from typing      import Tuple
from typing      import Dict
from typing      import Union
from typing      import List
from typing      import TypeVar
from typing      import Optional
from typing      import Iterable

from   numpy      import pi
import numpy as np

Number = TypeVar('Number', None, int, float)
Str   = TypeVar('Str', None, str)
Range = TypeVar('Range', None, Tuple[float, float])
Array = TypeVar('Array', List, np.array)

NN = np.nan



def phirad_to_deg(r : float)-> float:
    return (r + pi) * 180 / pi


def time_delta_from_time(T):
    return np.array([t - T[0] for t in T])


def divide_np_arrays(num : np.array, denom : np.array) -> np.array:
    """Safe division of two arrays"""
    assert len(num) == len(denom)
    ok    = denom > 0
    ratio = np.zeros(len(denom))
    np.divide(num, denom, out=ratio, where=ok)
    return ratio
