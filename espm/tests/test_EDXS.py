from espm.models.absorption_edxs import det_efficiency, absorption_correction
from espm.models.generate_EDXS_phases import generate_elts_dict
import numpy as np
import espm.models.EDXS_function as ef
from espm.models import EDXS
from espm.models.EDXS_function import lifshin_bremsstrahlung, lifshin_bremsstrahlung_b0, lifshin_bremsstrahlung_b1
import hyperspy.api as hs

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

def create_data()  : 
    a = np.random.rand(50,70,100)
    s = hs.signals.Signal1D(a)
    s.axes_manager[-1].offset = 0.2
    s.axes_manager[-1].scale = 0.1
    s.set_signal_type("EDS_espm")
    s.set_analysis_parameters()
    s.set_elements(elements = ["Si","O","Fe","Ca"])
    return s

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

def test_get_elements () :
    s = create_data()
    s.build_G(elements_dict ={'Fe' : 3.0})
    nelts = []
    for i in s.model.get_elements() :
        nelts.append(i)
    assert nelts == ['14', '8', '26', '20'] 
        
def test_carac_x_span () :
    # TODO : Find a way to implement the test of this function
    pass

def test_NMF_initialize_W () : 
    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "bremsstrahlung", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={})
    D = np.random.rand(1900,2)
    W = model.NMF_initialize_W(D)
    assert W.shape == (6,2)

    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "no_brstlg", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={'Nb' : 3.0})
    D = np.random.rand(1900,4)
    W = model.NMF_initialize_W(D)
    assert W.shape == (5,4)

    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "identity", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={})
    with np.testing.assert_raises(ValueError) : 
        D = np.random.rand(1900,4)
        W = model.NMF_initialize_W(D)

def test_NMF_simplex () :
    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "bremsstrahlung", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={})
    inds = model.NMF_simplex()
    assert inds == [0,1,2,3]

    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "no_brstlg", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={'Nb' : 3.0})
    inds = model.NMF_simplex()
    assert inds == [0,1,2,4]

def test_NMF_update () : 
    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "no_brstlg", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={})
    temp_G = model.G.copy()
    np.testing.assert_array_equal(model.NMF_update(), temp_G)
    np.testing.assert_array_equal(model.NMF_update(np.random.rand(10,10)), temp_G)

    model = EDXS(**model_parameters)
    model.generate_g_matr(g_type = "bremsstrahlung", elements = ["Na", "Sr", "Ge", "Nb"], elements_dict={})
    temp_G = model.G.copy()
    np.testing.assert_array_equal(model.NMF_update(), temp_G)
    assert model.NMF_update(np.random.rand(6,10)).shape == model.G.shape
    with np.testing.assert_raises(AssertionError):
        np.testing.assert_array_equal(model.NMF_update(np.random.rand(6,10))[:,-2:], temp_G[:,-2:])
    
def test_generate_g_matr () : 
    model1 = EDXS(**model_parameters)
    model2 = EDXS(**model_parameters)
    model3 = EDXS(**model_parameters)
    model4 = EDXS(**model_parameters)
    size = model_parameters["e_size"]
    elts_list = ["Na", "Sr", "Ge", "Nb"]
    model1.generate_g_matr(g_type = "bremsstrahlung", elements = elts_list, elements_dict={})
    G_brem = model1.G
    model2.generate_g_matr(g_type = "identity", elements = elts_list, elements_dict={})
    G_id = model2.G
    model3.generate_g_matr(g_type = "no_brstlg", elements = elts_list, elements_dict={})
    G_no_brstlg = model3.G
    model4.generate_g_matr(g_type = "bremsstrahlung", elements = elts_list, elements_dict={'Ge' : 3.0})
    G_brem_ge = model4.G

    assert G_brem.shape == (size, 6)
    assert G_no_brstlg.shape == (size, 4)
    assert G_brem_ge.shape == (size, 7)

    assert G_id is None
    np.testing.assert_allclose(G_brem[:,:-2], G_no_brstlg)
    np.testing.assert_allclose((G_brem_ge[:,3] + G_brem_ge[:,2])*model4.norm[0,0], G_brem[:,2]*model1.norm[0,0])

    # np.testing.assert_array_less(-1e-30, G_id)
    np.testing.assert_array_less(-1e-30, G_brem)
    np.testing.assert_array_less(-1e-30, G_no_brstlg)
    assert(model1.model_elts == ['11', '38', '32', '41'])
    assert(model2.model_elts == [])
    assert(model3.model_elts == ['11', '38', '32', '41'])
    assert(model4.model_elts == ['11', '38', '32_lo', '32_hi', '41'])

def test_G_bremsstrahlung() : 
    model = EDXS(**model_parameters)
    size = model_parameters["e_size"]
    elts_dict = {"V" : 0.3, "Ti" : 0.1, "Hf" : 0.05, "Pb" : 0.55 }
    B = ef.G_bremsstrahlung(model.x, model.E0, model.params_dict, elements_dict=elts_dict)

    assert B.shape == (size,2)
    np.testing.assert_array_less(-1e-30,B)


