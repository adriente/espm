r"""
The module :mod:`espm.datasets.built_in_EDXS_datasets` implements the functions that generate two built-in datasets:
- A dataset of 2 particles embedded in a matrix.
- A dataset with a linear local accumulation of Sr.
"""

from espm.datasets.base import generate_dataset
from espm.conf import DATASETS_PATH
from pathlib import Path
import hyperspy.api as hs
import os
from espm.models.generate_EDXS_phases import generate_modular_phases
from espm.weights.generate_weights import generate_weights
from espm.models.EDXS_function import elts_list_from_dict_list

particles_phases_dict  = {
    'elts_dicts'  : [
    {'23': 0.04704309583693933,
    '37': 0.914954275584854,
    '74': 0.06834271611694454},
    {'7': 0.4517936299777999,
      '70': 0.39973013314240835,
      '78': 0.08298592142537742},
    {'13': 0.43306626599937914,
      '22': 0.3985896640183708,
      '57': 0.8994030840372912}
    ],
    'brstlg_pars' : [
    {'b0': 7.408513360414626e-05,
    'b1': 6.606903143185911e-03},
    {'b0': 7.317975736931391e-05,
      'b1': 5.2126092148294355e-03},
    {'b0': 2.2664567599307173e-05,
      'b1': 3.1627208027847766e-03}
    ],
    'scales' : [1,1,1],
    'model_params': {'e_offset': 0.2,
    'e_size': 1980,
    'e_scale': 0.01,
    'width_slope': 0.01,
    'width_intercept': 0.065,
    'db_name': '200keV_xrays.json',
    'E0': 200,
    'params_dict': {'Abs': {'thickness': 1e-05,
    'toa': 22,
    'density': None,
    'atomic_fraction': False},
    'Det': 'SDD_efficiency.txt'}}
    }

particles_misc_dict = {
  'N': 500,
  'seed' : 91,
  'data_folder': 'built_in_particules',
  'shape_2d': [80, 80],
  'model': 'EDXS',
  'densities': [0.6030107883539217, 0.9870613994765459, 0.8894990661032164]
}

boundary_phases_dict  = {
    'elts_dicts'  : [
    {"Ca" : 0.54860348,
    "P" : 0.38286879,
    "Sr" : 0.03166235,
    "Cu" : 0.03686538},
    {"Ca" : 0.54860348,
    "P" : 0.38286879,
    "Sr" : 0.12166235,
    "Cu" : 0.03686538}
    ],
    'brstlg_pars' : [
    {"b0" : 5.5367e-4,
    "b1" : 0.0192181},
    {"b0" : 5.5367e-4,
    "b1" : 0.0192181}
    ],
    'scales' : [0.05,0.05],
    'model_params': {'e_offset': 1.27,
    'e_size': 3746,
    'e_scale': 0.005,
    'width_slope': 0.01,
    'width_intercept': 0.065,
    'db_name': '200keV_xrays.json',
    'E0': 200,
    'params_dict': {'Abs': {'thickness': 140e-07,
    'toa': 22,
    'density': 3.124,
    'atomic_fraction': False},
    'Det': 'SDD_efficiency.txt'}}
    }

boundary_misc_dict = {
  'N': 15,
  'seed' : 0,
  'data_folder': 'built_in_grain_boundary',
  'shape_2d': [100, 400],
  'model': 'EDXS',
  'densities': [1, 1]
}


def generate_built_in_datasets (seeds_range = 10) : 
    r"""
    Generate the two built-in datasets if they are not already present in the datasets folder.

    Parameters
    ----------
    seeds_range : int
        The number of seeds to use for the generation of the built-in datasets. The built-in datasets are generated with a base_seed, and then the base_seed + 1, base_seed + 2, etc. up to base_seed + seeds_range -1.

    Returns
    -------
    None
    """
    if not(os.path.isdir(DATASETS_PATH / Path(particles_misc_dict["data_folder"]))) : 
        print("Generating 2 particles + one matrix built-in dataset. This will take a minute.")
        particle_phases = generate_modular_phases(**particles_phases_dict)
        particles_weights = generate_weights("sphere", particles_misc_dict['shape_2d'], n_phases=3, seed=particles_misc_dict['seed'], radius = 15)
        particles_elements = elts_list_from_dict_list(particles_phases_dict['elts_dicts'])
        generate_dataset(base_seed=particles_misc_dict['seed'],
                         sample_number=seeds_range,
                         model_params = particles_phases_dict['model_params'],
                         misc_params = particles_misc_dict,
                         phases = particle_phases,
                         weights = particles_weights,
                         elements = particles_elements)
    if not(os.path.isdir(DATASETS_PATH / Path(boundary_misc_dict["data_folder"]))) :
        print("Generating a grain boundary with Sr segregation. This will take a minute.")
        boundary_phases = generate_modular_phases(**boundary_phases_dict)
        boundary_weights = generate_weights("gaussian_ripple", boundary_misc_dict['shape_2d'], n_phases=2, seed=boundary_misc_dict['seed'], width = 10)
        boundary_elements = elts_list_from_dict_list(boundary_phases_dict['elts_dicts'])
        generate_dataset(base_seed=boundary_misc_dict['seed'],
                         sample_number=seeds_range,
                         model_params = boundary_phases_dict['model_params'],
                         misc_params = boundary_misc_dict,
                         phases = boundary_phases,
                         weights = boundary_weights,
                         elements = boundary_elements)
def load_particules (sample = 0) : 
    r"""
    Load the built-in dataset of particles.

    Parameters
    ----------
    sample : int
        The sample number to load.

    Returns
    -------
    spim : hyperspy.signals.EDS_espm
        The loaded dataset.
    """
    filename = DATASETS_PATH / Path("{}/sample_{}.hspy".format(particles_misc_dict["data_folder"],sample))
    spim = hs.load(filename)
    return spim

def load_grain_boundary (sample = 0) :
    r"""
    Load the built-in dataset of a grain boundary.

    Parameters
    ----------
    sample : int
        The sample number to load.

    Returns
    -------
    spim : hyperspy.signals.EDS_espm
        The loaded dataset.
    """
    filename = DATASETS_PATH / Path("{}/sample_{}.hspy".format(boundary_misc_dict["data_folder"],sample))
    spim = hs.load(filename)
    return spim