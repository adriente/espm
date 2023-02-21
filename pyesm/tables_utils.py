import json
from pathlib import Path
from espm.conf import DB_PATH, SYMBOLS_PERIODIC_TABLE, SIEGBAHN_TO_IUPAC
import re
import numpy as np
import espm.utils as u

def load_table (db_name) :
    """
    Takes a filename in tables folder of espm and returns a tuple of dicts.
    1st dict : table of the cross sections
    2nd dict : metadata of the table
    
    Call espm.conf.DB_PATH to get the folder of the tables.
    """
    db_path = DB_PATH / Path(db_name)
    with open(db_path,"r") as f :
        json_dict = json.load(f)
    return json_dict["table"], json_dict["metadata"]

def import_k_factors(table,mdata,k_factors_names,k_factors_values,ref_name) : 

    with open(SYMBOLS_PERIODIC_TABLE,"r") as f : 
        SPT = json.load(f)["table"]

    with open(SIEGBAHN_TO_IUPAC,"r") as f : 
        STI = json.load(f)

    for i,name in enumerate(k_factors_names) : 
        if name == ref_name : 
            mr = re.match(r"([A-Z][a-z]?)_(.*)",name)
            ref_at_num = SPT[mr.group(1)]["number"]
            ref_lines =  STI[mr.group(2)]
            ref_sig_vals = []
            for l in ref_lines :
                if l in table[str(ref_at_num)] : 
                    ref_sig_vals.append(table[str(ref_at_num)][l]["cs"])
            ref_sig_val = np.mean(ref_sig_vals)
            ref_k_val = k_factors_values[i]

    for i,name in enumerate(k_factors_names) : 
        m0 = re.match(r"([A-Z][a-z]?)_(.*)",name)
        if m0 : 
            at_num = SPT[m0.group(1)]["number"]
            lines =  STI[m0.group(2)]
            for line in lines : 
                new_k = k_factors_values[i]/ref_k_val
                if line in table[str(at_num)] : 
                    sig_val = table[str(at_num)][line]["cs"]
                    new_value = ref_sig_val*new_k/sig_val
                    new_table, new_mdata = modify_table_lines(table,mdata,[at_num],line,new_value)
    return new_table,new_mdata
            

def modify_table_lines (table, mdata, elements, line, coeff) :
    """
    Takes a table, its metadata, a list of elements (atomic number), a line regex and a coefficient.
    Returns the table and metadata, with the cross section of all the lines of the selected elements modified by the coefficient.
    line regex examples :  input "L" will modify all the L lines, input "L3" will modifiy all the L3 lines,
    input "L3M2" will modify the "L3M2" line.  
    """ 
    if mdata["lines"] :
        for elt in elements : 
            for key in table[str(elt)].keys() :
                if re.match(r"^{}".format(line),key) : 
                    table[str(elt)][key]["cs"] *=coeff
                    if "modifications" in mdata : 
                        mdata["modifications"][str(elt) + "_" + key] = coeff
                    else : 
                        mdata["modifications"] = {}
                        mdata["modifications"][str(elt) + "_" + key] = coeff
                        
    else :
        print("You need to enable line notation")
    return table, mdata

def save_table (filename, table, mdata) : 
    """
    Saves a table and its metadata in a json file.
    The structure of the json file is compliant with espm.
    """
    d = {}
    d["table"] = table
    d["metadata"] = mdata
    with open(filename,"w") as f :
        json.dump(d,f,indent = 4)
        
def get_k_factor (table_name, element, line, range = 0.5, ref_elt = "14", ref_line = "KL3", ref_range = 0.5) : 
    """
    Takes a table name, an element (atomic number), a line name and an energy range.
    Returns the k-factor of this group of lines.
    
    The default reference line is Si KL3 with an integration range of 0.5. 

    To group lines together, the cross section of all lines within the specified energy range are summed up.
    With correct energy range you can get K-alpha, K-beta, etc ...
    """
    table, mdata = load_table(table_name)
    ref_cs = 0.0
    cs = 0.0
    if mdata["lines"] :
        ref_en = table[str(ref_elt)][ref_line]["energy"]
        for key in table[str(ref_elt)].keys() :
            en = table[str(ref_elt)][key]["energy"]
            if (en < ref_en + ref_range) and (en > ref_en - ref_range) : 
                ref_cs += table[str(ref_elt)][key]["cs"]

        elt_en = table[str(element)][line]["energy"]
        for key in table[str(element)].keys() :
            en = table[str(element)][key]["energy"]
            if (en < elt_en + range) and (en > elt_en - range) : 
                cs += table[str(element)][key]["cs"]
        
    else :
        print("You need to enable line notation")

    return cs/ref_cs