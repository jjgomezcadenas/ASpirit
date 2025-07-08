import time
import itertools

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

def progressbar(iterable, *, flushmod=1, nelements=None, index=False):
    if nelements is None:
        temp, iterable = itertools.tee(iterable)
        nelements = sum(1 for _ in temp)

    dt = 0
    t0 = t00 = time.time()
    for i, value in enumerate(iterable):
        if i % flushmod == 0:
            eta = (nelements - i) * dt
            tot = time.time() - t00
            print(f"\rItem {i+1} of {nelements} | {dt:.2f} s/item | ETA {eta/60:.1f} min | Ellapsed {tot/60:.1f} min".ljust(100), end="", flush=True)
        if index:
            yield i, value
        else:
            yield value
        t1  = time.time()
        dt  = (dt*i + t1 - t0) / (i+1)
        t0  = t1
    print(f"\rFinished {nelements} elements in {tot/60:.1f} min | {dt:.2f} s/item".ljust(100), end="", flush=True)
    print()
