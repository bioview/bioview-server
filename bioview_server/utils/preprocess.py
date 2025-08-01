import numpy as np
from scipy import signal

"""
Filtering is done similar to our LabView implementation, 
using cascaded sections of low-order Butterworth filter. 
"""


def apply_filter(data, filter, zi=None):
    if zi is None:
        zi = (
            signal.sosfilt_zi(filter) * data[0]
            if len(data) > 0
            else signal.sosfilt_zi(filter)
        )

    # Apply filter with state
    filtered_data, zf = signal.sosfilt(filter, data, zi=zi)

    return filtered_data, zf


def get_filter(
    bounds: list[float],
    samp_rate: int,
    ftype: str = "ellip",
    btype: str = "band",
    order: int = 2,
):
    nyquist = samp_rate / 2
    norm_bounds = np.array(bounds) / nyquist

    if ftype == "ellip":
        filt_sos = signal.ellip(order, 0.01, 50, norm_bounds, btype=btype, output="sos")
    else:
        filt_sos = signal.butter(order, norm_bounds, btype=btype, output="sos")

    return filt_sos
