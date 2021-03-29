# goes_classes = ["quiet", "A", "B", "C", "M", "X"]
goes_classes = ["quiet", "B", "C", "M", "X"]
import math


def flux_to_class(f: float, only_main=True):
    "maps the peak_flux of a flare to one of the following descriptors: \
    *quiet* = 1e-9, *B* >= 1e-7, *C* >= 1e-6, *M* >= 1e-5, and *X* >= 1e-4\
    See also: https://en.wikipedia.org/wiki/Solar_flare#Classification"
    decade = int(min(math.floor(math.log10(f)), -4))
    sub = round(10 ** -decade * f)
    if decade < -4:  # avoiding class 10
        decade += sub // 10
        sub = max(sub % 10, 1)
    main_class = goes_classes[decade + 8] if decade >= -8 else "quiet"
    sub_class = str(sub) if main_class != "quiet" and only_main != True else ""
    return main_class + sub_class


def flux_to_id(f: float):
    decade = int(min(math.floor(math.log10(f)), -4))
    sub = round(10 ** -decade * f)
    if decade < -4:
        decade += sub // 10

    # return decade + 9 if decade >= -8 else 0
    return max(decade + 8, 0)


def id_to_class(id: int) -> str:
    return goes_classes[id]


def id_to_flux(id: int) -> float:
    fluxes = [1e-9, 5e-7, 5e-6, 5e-5, 5e-4]
    return fluxes[id]


def flux_to_binary_class(f: float):
    # 0: no flare (or A/B)
    # 1: C-, M- or X-class flare

    id = flux_to_id(f)
    return int(id > 1)
