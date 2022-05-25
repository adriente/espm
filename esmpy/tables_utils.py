import json
from pathlib import Path
from esmpy.conf import DB_PATH
import re

def load_table (db_name) :
    """
    Takes a filename in tables folder of esmpy and returns a tuple of dicts.
    1st dict : table of the cross sections
    2nd dict : metadata of the table
    
    Call esmpy.conf.DB_PATH to get the folder of the tables.
    """
    db_path = DB_PATH / Path(db_name)
    with open(db_path,"r") as f :
        json_dict = json.load(f)
    return json_dict["table"], json_dict["metadata"]

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
                    mdata["modifications"][str(elt)][key] = coeff
    else :
        print("You need to enable line notation")
    return table, mdata

def save_table (filename, table, mdata) : 
    """
    Saves a table and its metadata in a json file.
    The structure of the json file is compliant with esmpy.
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