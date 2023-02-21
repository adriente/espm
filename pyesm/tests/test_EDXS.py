from espm.models.absorption_edxs import det_efficiency, absorption_correction
from espm.datasets.generate_EDXS_phases import generate_elts_dict
import numpy as np
import espm.models.EDXS_function as ef
from espm.models import EDXS
from espm.models.EDXS_function import lifshin_bremsstrahlung, lifshin_bremsstrahlung_b0, lifshin_bremsstrahlung_b1

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

model_parameters = {
        "e_offset" : 0.2,
        "e_size" : 1900,
        "e_scale" : 0.01,
        "width_slope" : 0.01,
        "width_intercept" : 0.065,
        "db_name" : "default_xrays.json",
        "E0" : 200,
        "params_dict" : {
            "Abs" : {
                "thickness" : 200.0e-7,
                "toa" : 35,
                "density" : 3.12
            },
            "Det" : det_dict
        }
    }

def test_lifshin_bremsstrahlung () : 
    E0 = 200
    x = np.linspace(0.2,E0-0.1, num = 2000 )
    for i in range(10) : 
        b0 = np.random.rand()
        b1 = np.random.rand()
        y0 = lifshin_bremsstrahlung_b0(x,b0,E0)
        y1 = lifshin_bremsstrahlung_b1(x,b1,E0)
        y = lifshin_bremsstrahlung (x,b0,b1,E0)

        np.testing.assert_array_less(1e-30, y0)
        np.testing.assert_array_less(1e-30, y1)
        np.testing.assert_array_less(1e-30, y)
        np.testing.assert_allclose(y, y1 + y0)

def test_efficiency () : 
    det = det_efficiency(x,det_dict) 
    np.testing.assert_array_less(1e-30,det)
    np.testing.assert_array_less(det,1.0)

def test_absorption () : 
    elts = generate_elts_dict(42)
    abs_corr = absorption_correction(x,thickness = 1e-4, toa = 35, density = 5, elements_dict=elts)

    np.testing.assert_array_less(1e-30,abs_corr)
    np.testing.assert_array_less(abs_corr,1.0)

def test_elts_dict_from_W () : 
    # elements_mass = [28.085,40.0784,15.999,30.9737619985,12.011]
    part_W = np.random.rand(5,3)
    avg_W = part_W.sum(axis=1)/3
    # weighted_W= [avg_W[i]*elt_mass for i,elt_mass in enumerate(elements_mass)]
    norm_W = avg_W/np.sum(avg_W)
    elements_list = ["Si","Ca", "O", "P", "C"]

    result = ef.elts_dict_from_W(part_W, elements = elements_list)
    for i,key in enumerate(result) : 
        np.testing.assert_allclose(result[key],norm_W[i],rtol = 1e-1)

def test_elts_dict_from_dict_list () : 
    dict_list = [{"chou" : 1, "carottes" : 2, "navet" : 3}, {"chou" : 2, "oignons" : 3, "navet" : 3}, {"orange" : 6, "citron" : 4, "poireau" : 8}]
    unique_dict = ef.elts_dict_from_dict_list(dict_list)
    sum_elts = 32
    assert unique_dict == {"chou" : 3/sum_elts, "carottes" : 2/sum_elts, "navet" : 6/sum_elts, "orange" : 6/sum_elts, "citron" : 4/sum_elts, "poireau" : 8/sum_elts, "oignons" : 3/sum_elts}

def test_continuum_xrays () : 
    E0 = 200
    b0 = np.random.uniform(0.0,50.0)
    b1 = np.random.uniform(0.0, 50.0)
    cont_x = ef.continuum_xrays(x,model_parameters["params_dict"],b0= b0, b1 = b1, E0 = E0,elements_dict = {"Si" : 1.0, "C" : 0.2, "Bi" : 3.2})
    
    np.testing.assert_array_less(1e-30,cont_x)

def test_update_bremsstrahlung () : 
    elts_list = ["Si", "Co", "Cu","O"]
    part_P = np.ones((4,1))/4
    model = EDXS(**model_parameters)
    model.generate_g_matr(elements=elts_list)
    G = model.G
    new_G = ef.update_bremsstrahlung(G, part_P, model_parameters, elts_list)

    assert G.shape == new_G.shape
    np.testing.assert_allclose(G[:,:-2],new_G[:,:-2])

def test_generate_g_matr () : 
    model1 = EDXS(**model_parameters)
    model2 = EDXS(**model_parameters)
    model3 = EDXS(**model_parameters)
    size = model_parameters["e_size"]
    elts_list = ["Na", "Sr", "Ge", "Nb"]
    model1.generate_g_matr(g_type = "bremsstrahlung", elements = elts_list)
    G_brem = model1.G
    model2.generate_g_matr(g_type = "identity", elements = elts_list)
    G_id = model2.G
    model3.generate_g_matr(g_type = "no_brstlg", elements = elts_list)
    G_no_brstlg = model3.G

    assert G_brem.shape == (size, 6)
    assert G_id.shape == (size, size)
    assert G_no_brstlg.shape == (size, 4)

    np.testing.assert_allclose(G_id, np.diag(np.ones((size,))
    ))
    np.testing.assert_allclose(G_brem[:,:-2], G_no_brstlg)

    np.testing.assert_array_less(-1e-30, G_id)
    np.testing.assert_array_less(-1e-30, G_brem)
    np.testing.assert_array_less(-1e-30, G_no_brstlg)

def test_G_bremsstrahlung() : 
    model = EDXS(**model_parameters)
    size = model_parameters["e_size"]
    elts_dict = {"V" : 0.3, "Ti" : 0.1, "Hf" : 0.05, "Pb" : 0.55 }
    B = ef.G_bremsstrahlung(model.x, model.E0, model.params_dict, elements_dict=elts_dict)

    assert B.shape == (size,2)
    np.testing.assert_array_less(-1e-30,B)


