import json
from pathlib import Path
from esmpy.conf import DB_PATH
import re

def load_table (db_name) :
    db_path = DB_PATH / Path(db_name)
    with open(db_path,"r") as f :
        json_dict = json.load(f)
    return json_dict["table"], json_dict["metadata"]

def modify_table_lines (table_name, elements, line, coeff) : 
    table, mdata = load_table(table_name)
    if mdata["lines"] :
        for elt in elements : 
            for key in table[str(elt)].keys() :
                if re.match(r"^{}".format(line),key) : 
                    table[str(elt)][key]["cs"] *=coeff
    else :
        print("You need to enable line notation")

def get_k_factor (table_name, element, line, range = 0.5, ref_elt = "14", ref_line = "KL3", ref_range = 0.5) : 
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