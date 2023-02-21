from hyperspy.misc.material import _density_of_mixture
import numpy as np
import espm.utils as u


def test_rescale() :
    np.random.seed(0)
    for _ in range(20):
        # for k small
        W = np.random.rand(27,5)
        H = np.random.rand(5,150)
        W_r, H_r = u.rescaled_DH(W,H)
        assert(np.abs(np.mean(H_r.sum(axis=0)) -1) < np.abs(np.mean(H.sum(axis=0)) -1))
        np.testing.assert_array_almost_equal(W@H,W_r@H_r)

        # for k large
        W = np.random.rand(50,10)
        H = np.random.rand(10,5)
        W_r, H_r = u.rescaled_DH(W,H)
        np.testing.assert_array_almost_equal(W@H,W_r@H_r)
        np.testing.assert_allclose(H_r.sum(axis=0),1, atol=0.2)

def test_number_symbols () : 
    mixed_dict = {"Fe" : 1.0, "Si" : 2.0, 27 : 3.0, "83" : 4.0, 8 : 5.0}
    symbol_dict = {"Fe" : 1.0, "Si" : 2.0, "Co" : 3.0, "Bi" : 4.0, "O" : 5.0}
    number_dict = {26 : 1.0, 14 : 2.0, 27 : 3.0, 83 : 4.0, 8 : 5.0}

    mixed_list = ["Fe","Si",27,"83",8]
    number_list = [26,14,27,83,8]
    symbol_list = ["Fe", "Si", "Co", "Bi", "O"]

    @u.symbol_to_number_dict
    def s_to_n (elements_dict = {}) : 
        return elements_dict

    @u.number_to_symbol_dict
    def n_to_s (elements_dict = {}) : 
        return elements_dict

    @u.symbol_to_number_list
    def s_to_n_list (elements = []) : 
        return elements

    @u.number_to_symbol_list
    def n_to_s_list (elements = []) : 
        return elements

    assert s_to_n(elements_dict = mixed_dict) == number_dict
    assert n_to_s(elements_dict = mixed_dict) == symbol_dict
    assert s_to_n_list(elements=mixed_list) == number_list
    assert n_to_s_list(elements = mixed_list) == symbol_list

    
def test_is_symbol () : 
    symbols = ["Si", "45", 34, "si", "Zbrra"]
    bools = [True, False, False, False, False]

    bools_test = []
    for i in symbols : 
        bools_test.append(u.is_symbol(i))

    assert bools == bools_test

def test_hspy_wrappers () : 
    atomic_fractions = {8 : 0.2, 14 : 0.3, 26 : 0.1, 83 : 0.4}
    weight_fractions = {'O': 0.03174415159035731, 'Si': 0.08358598161408992, 'Fe': 0.055400582070687154, 'Bi': 0.8292692847248655}

    res_wt = u.atomic_to_weight_dict(elements_dict = atomic_fractions)
    assert res_wt == weight_fractions

    res_dens = u.approx_density(atomic_fraction = True, elements_dict=atomic_fractions)
    hspy_dens = _density_of_mixture([0.03174415159035731, 0.08358598161408992, 0.055400582070687154, 0.8292692847248655],['O','Si','Fe','Bi'])

    assert res_dens == hspy_dens

def test_arg_helper () : 
    default_dict = {
        "chou" : "non",
        "carottes" : 3,
        "salades" : {"roquette" : 2, "batavia" : 5},
        "tubercules" : {
            "oignons" : {"jaune" : [1,2,3], "blanc" : [3,2,1]},
             "echalottes" : 5
             }
        }

    input_dict = {
        "chou" : 4,
        "navets" : 3,
        "salades" : {"clous" : 3, "roquette" : 2},
        "tubercules" : {
            "oignons" : {"jaune" : 9},
            "vis" : {"cruciformes" : 3, "hexagonales" : 6}
        }
    }

    result_dict = {
        "chou" : 4,
        "carottes" : 3,
        "navets" : 3,
        "salades" : {"clous" : 3, "roquette" : 2, "batavia" : 5},
        "tubercules" : {
            "oignons" : {"jaune" : 9, "blanc" : [3,2,1]},
            "echalottes" : 5,
            "vis" : {"cruciformes" : 3, "hexagonales" : 6}
        } 
    }

    assert u.arg_helper(input_dict,default_dict) == result_dict

def test_bin_spim () : 
    array = np.random.rand(100,20,30)

    assert u.bin_spim(array,50,10).shape == (50,10,30)
    assert u.bin_spim(array,30,6).shape == (30,6,30)