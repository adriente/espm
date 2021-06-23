from snmfem.models.absorption_edxs import det_efficiency, absorption_correction
from snmfem.datasets.generate_EDXS_phases import generate_elts_dict
import numpy as np

x = np.linspace(0.2,19,num = 2000)
det_dict = {
    "detection" : {
        "thickness" : 450e-3,
        "density" : 2.5
    },
    "layer1" : {
        "thickness" : 1e-6,
        "density" : 2.3,
        "atomic_fraction" : True,
        "elements_dict" : {14 : 1, 8 : 2}
    },
    "layer2" : {
        "thickness" : 3e-6,
        "density" : 2.7,
        "elements_dict" : {"Al" : 1.0}

    }
}

def test_efficiency () : 
    det = det_efficiency(x,det_dict) 
    np.testing.assert_array_less(1e-30,det)
    np.testing.assert_array_less(det,1.0)

def test_absorption () : 
    elts = generate_elts_dict(42)
    abs_corr = absorption_correction(x,thickness = 1e-4, toa = 35, density = 5, elements_dict=elts)

    np.testing.assert_array_less(1e-30,abs_corr)
    np.testing.assert_array_less(abs_corr,1.0)