import json
import re
from espm.conf import DB_PATH, SYMBOLS_PERIODIC_TABLE, SIEGBAHN_TO_IUPAC

def modify_cross_sections(energy, input_type = None, lines_new_values = None, lines_scaling_factors = None, k_factors = None, reference_line = None, output_filename = None):
    r"""
    Function allowing the user to modify the X-ray emission cross-sections of the input database. The function can be with three diffrent input types:
    new values, scaling factors and k-factors with a reference line.
    
    Parameters
    ----------
    energy : int
        The electron energy in keV for which the cross-sections are to be modified.
    input_type : str
        The type of input to be used. Can be one of the following: 'new_values', 'scaling_factors', 'k_factors'.
    lines_new_values : dict
        [For the 'new_values' input type] A dictionary containing the new values for the cross-sections. 
        The keys of the dictionary should be the element symbol and the X-ray line in Siegbahn notation, separated by an underscore.
        The correspoding values should be the new cross-section values. 
        For example: {'Cu_Ka' : 6.3e-23}.
    lines_scaling_factors : dict
        [For the 'scaling_factors' input type] A dictionary containing the scaling factors for the cross-sections. 
        The keys of the dictionary should be the element symbol and the X-ray line in Siegbahn notation, separated by an underscore.
        The corresponding values should be the scaling factors.
        For example: {'Cu_Ka' : 1.2}.
    k_factors : dict
        [For the 'k_factors' input type] A dictionary containing the k-factors for the cross-sections. 
        The keys of the dictionary should be the element symbol and the X-ray line in Siegbahn notation, separated by an underscore.
        The corresponding values should be the Cliff-Lorimer k-factors.
        For example: {'Cu_Ka' : 1.75}.
    reference_line : str
        [For the 'k_factors' input type] The reference line for the k-factors. 
        Should be the element symbol and the X-ray line in Siegbahn notation, separated by an underscore.
        For example: 'Si_Ka'.
    output_filename : str
        The name of the output file. If None, the default name is the same as the input file with '_modified' appended.
    
    Returns
    -------
    None
    """
    
    if energy == 100:
        cross_sections_json = DB_PATH / '100keV_xrays.json'
    elif energy == 200:
        cross_sections_json = DB_PATH / '200keV_xrays.json'
    elif energy == 300:
        cross_sections_json = DB_PATH / '300keV_xrays.json'

    with open(cross_sections_json, 'r') as cross_sections_file:
        cross_sections_data = json.load(cross_sections_file)

    with open(SYMBOLS_PERIODIC_TABLE, 'r') as atomic_symbols_file:
        atomic_symbols_data = json.load(atomic_symbols_file)

    with open(SIEGBAHN_TO_IUPAC, 'r') as siegbahn_to_iupac_file:
        siegbahn_to_iupac_data = json.load(siegbahn_to_iupac_file)

    if input_type == 'new_values':
        for el_line, value in lines_new_values.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line = line_match.group(2)
            
            atomic_number = str(atomic_symbols_data['table'][element]['number'])
            line_iupac = siegbahn_to_iupac_data.get(line, [])[0]

            if atomic_number in cross_sections_data['table'] and line_iupac in cross_sections_data['table'][atomic_number]:
                cross_sections_data['table'][atomic_number][line_iupac]['cs'] = value
                print(f"Set new cross-section for Element {element}, Line {line} to {value}")
            else:
                raise ValueError(f"Element {element} or Line {line} not found in the JSON data.")

    elif input_type == 'scaling_factors':
        for el_line, value in lines_scaling_factors.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line = line_match.group(2)
            
            atomic_number = str(atomic_symbols_data['table'][element]['number'])
            line_iupac = siegbahn_to_iupac_data.get(line, [])[0]

            if atomic_number in cross_sections_data['table'] and line_iupac in cross_sections_data['table'][atomic_number]:
                current_cs = cross_sections_data['table'][atomic_number][line_iupac]['cs']
                cross_sections_data['table'][atomic_number][line_iupac]['cs'] = current_cs * value
                print(f"Applied scaling factor {value} to cross-section for Element {element}, Line {line}")
            else:
                raise ValueError(f"Element {element} or Line {line} not found in the JSON data.")

    elif input_type == 'k_factors' and reference_line is not None:
        ref_match = re.match(r"([A-Z][a-z]?)_(.*)", reference_line)
        ref_element = ref_match.group(1)
        ref_line = ref_match.group(2)
        
        ref_at_num = str(atomic_symbols_data["table"][ref_element]['number'])
        ref_line_iupac = siegbahn_to_iupac_data.get(ref_line, [])[0]
            
        if ref_line_iupac in cross_sections_data['table'][ref_at_num]:
            ref_cs = cross_sections_data['table'][ref_at_num][ref_line_iupac]['cs']
        else:
            raise ValueError(f"Reference element {ref_element} or line {ref_line} not found in the JSON data.")
        
        for el_line, factor in k_factors.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line = line_match.group(2)
            
            atomic_number = str(atomic_symbols_data['table'][element]['number'])
            line_iupac = siegbahn_to_iupac_data.get(line, [])[0]
                
            if line_iupac in cross_sections_data['table'][atomic_number]:
                new_value = ref_cs * factor
                cross_sections_data['table'][atomic_number][line_iupac]['cs'] = new_value
                print(f"Set new cross-section for Element {element}, Line {line} based on a k-factor of {factor} having element {ref_element}, line {ref_line} as reference line.")
            else:
                raise ValueError(f"Element {element} or Line {line} not found in the JSON data.")
            
    if output_filename is None:
        output_filename = f"{energy}keV_xrays_modified.json"

    with open(DB_PATH / output_filename, 'w') as file:
        json.dump(cross_sections_data, file, indent = 4)