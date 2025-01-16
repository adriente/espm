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
        For example: {'Cu_Ka1' : 6.3e-23}.
    lines_scaling_factors : dict
        [For the 'scaling_factors' input type] A dictionary containing the scaling factors for the cross-sections. 
        The keys of the dictionary should be the element symbol and the X-ray line family in Siegbahn notation, separated by an underscore.
        The corresponding values should be the scaling factors.
        For example: {'Cu_Ka' : 1.2}.
    k_factors : dict
        [For the 'k_factors' input type] A dictionary containing the k-factors for the cross-sections. 
        The keys of the dictionary should be the element symbol and the X-ray line family in Siegbahn notation, separated by an underscore.
        The corresponding values should be the Cliff-Lorimer k-factors.
        For example: {'Cu_Ka' : 1.75}.
    reference_line : str
        [For the 'k_factors' input type] The reference line for the k-factors. 
        Should be the element symbol and the X-ray line family in Siegbahn notation, separated by an underscore.
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
        
    iupac_to_siegbahn = {
        iupac: siegbahn
        for siegbahn, iupac_list in siegbahn_to_iupac_data.items()
        for iupac in iupac_list
    }
    
    if input_type == 'new_values':
        for el_line, value in lines_new_values.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line_name = line_match.group(2)
            
            atomic_number = str(atomic_symbols_data['table'][element]['number'])
            line_iupac = siegbahn_to_iupac_data.get(line_name, [line_name])[0]
            
            if atomic_number in cross_sections_data['table'] and line_iupac in cross_sections_data['table'][atomic_number]:
                cross_sections_data['table'][atomic_number][line_iupac]['cs'] = value

                line_siegbahn = iupac_to_siegbahn.get(line_iupac, line_iupac)

                print(f"Set new cross-section for Element {element}, Line {line_siegbahn} to {value:.2e}")
            else:
                print(f"Warning: Element {element} or line {line_iupac} not found in the JSON data.")

    elif input_type == 'scaling_factors':
        for el_line, factor in lines_scaling_factors.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line_group = line_match.group(2)

            atomic_number = str(atomic_symbols_data['table'][element]['number'])
            line_iupac_list = siegbahn_to_iupac_data.get(line_group, [])

            for line_iupac in line_iupac_list:
                if atomic_number in cross_sections_data['table'] and line_iupac in cross_sections_data['table'][atomic_number]:
                    current_cs = cross_sections_data['table'][atomic_number][line_iupac]['cs']
                    cross_sections_data['table'][atomic_number][line_iupac]['cs'] = current_cs * factor

                    line_siegbahn = iupac_to_siegbahn.get(line_iupac, line_iupac)

                    print(f"Applied scaling factor {factor} to cross-section for element {element}, line {line_siegbahn}")
                    
                else:
                    print(f"Skipped element {element}, line {line_iupac}: No matching data in database.")

    elif input_type == 'k_factors' and reference_line is not None:
        ref_match = re.match(r"([A-Z][a-z]?)_(.*)", reference_line)
        ref_element = ref_match.group(1)
        ref_line_family = ref_match.group(2)
        
        ref_at_num = str(atomic_symbols_data["table"][ref_element]['number'])
        
        ref_lines_iupac = siegbahn_to_iupac_data.get(ref_line_family, [])
        if not ref_lines_iupac:
            raise ValueError(f"Reference line family {ref_line_family} not found in Siegbahn to IUPAC mapping.")
        
       
        valid_ref_lines = {
            line: cross_sections_data['table'][ref_at_num][line]['cs']
            for line in ref_lines_iupac if line in cross_sections_data['table'][ref_at_num]
        }
        
        if not valid_ref_lines:
            raise ValueError(f"No valid reference lines found for {ref_element}_{ref_line_family} in the cross-section data.")
        
        for el_line, factor in k_factors.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line_family = line_match.group(2)
            
            atomic_number = str(atomic_symbols_data['table'][element]['number'])

            target_lines_iupac = siegbahn_to_iupac_data.get(line_family, [])
            
            if not target_lines_iupac:
                raise ValueError(f"Target line family {line_family} not found in Siegbahn to IUPAC mapping.")

            for ref_line, target_line in zip(ref_lines_iupac, target_lines_iupac):
                if ref_line in valid_ref_lines and target_line in cross_sections_data['table'][atomic_number]:
                    ref_cs = valid_ref_lines[ref_line]
                    new_value = ref_cs * factor
                    cross_sections_data['table'][atomic_number][target_line]['cs'] = new_value
                    
                    ref_line_siegbahn = iupac_to_siegbahn.get(ref_line, ref_line)
                    target_line_siegbahn = iupac_to_siegbahn.get(target_line, target_line)
                    
                    print(f"Set new cross-section for element {element}, line {target_line_siegbahn} "
                        f"based on k-factor {factor} using {ref_element}, line {ref_line_siegbahn} as reference.")
                else:
                    print(f"Skipped element {element}, line {target_line}: No matching data in reference or target.")
        
    original_files_list = [
        'SDD_efficiency.txt',
        '200keV_xrays.json',
        '__init__.py',
        'default_xrays.json',
        'periodic_table_symbols.json',
        'siegbahn_to_iupac.json',
        '300keV_xrays.json',
        'periodic_table_number.json',
        '100keV_xrays.json'
        ]
    
    if output_filename is None:
        input_filename = cross_sections_json.name
        output_filename = input_filename.replace('.json', '_modified.json')
    
    if output_filename in original_files_list:
        raise ValueError("The output filename cannot be the same as one of the original files.")

    with open(DB_PATH / output_filename, 'w') as file:
        json.dump(cross_sections_data, file, indent = 4)