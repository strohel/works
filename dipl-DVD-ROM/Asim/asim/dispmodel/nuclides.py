"""A simple database of radionuclides.

To be used as:

>>> import nuclides
>>> argon = nuclides.db["Ar-41"]
>>> argon.half_life
6560.0
"""

from iface import Nuclide


# TODO: which deposition? In what units?
NOBLE_DEPO_VG = 0.0  #: constant: deposition of noble radionuclides
AERO_DEPO_VG = 0.003  #: constant: deposition of aero radionuclides
ELIOD_DEPO_VG = 0.015  #: constant: deposition of eliod radionuclides

#: structure holding nuclide database
db = {
    "Ar-41": Nuclide(
        dose_mju=0.00668,
        dose_mjue=6.12E-16,
        dose_a=1.040,
        dose_b=0.0338,
        dose_e=1293000.0,
        dose_cb=0.765,
        half_life=6560.0,
        depo_vg=NOBLE_DEPO_VG
    ),
}
