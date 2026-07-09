import json
import re
from pathlib import Path

import numpy as np

from espm.conf import DB_PATH, SIEGBAHN_TO_IUPAC, SYMBOLS_PERIODIC_TABLE


def modify_cross_sections(
    energy: int | str,
    input_type: str = "new_values",
    lines_new_values: dict[str, float] = None,
    lines_scaling_factors: dict[str, float] = None,
    k_factors: dict[str, float] = None,
    reference_line: str = None,
    output_filename: str = None,
) -> None:
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

    assert not (
        lines_new_values is None and lines_scaling_factors is None and k_factors is None
    ), f"You need to input a dict of lines to modify with type : {input_type}."

    if energy == 100:
        cross_sections_json = DB_PATH / "100keV_xrays.json"
    elif energy == 200:
        cross_sections_json = DB_PATH / "200keV_xrays.json"
    elif energy == 300:
        cross_sections_json = DB_PATH / "300keV_xrays.json"

    with open(cross_sections_json, "r") as cross_sections_file:
        cross_sections_data = json.load(cross_sections_file)

    with open(SYMBOLS_PERIODIC_TABLE, "r") as atomic_symbols_file:
        atomic_symbols_data = json.load(atomic_symbols_file)

    with open(SIEGBAHN_TO_IUPAC, "r") as siegbahn_to_iupac_file:
        siegbahn_to_iupac_data = json.load(siegbahn_to_iupac_file)

    iupac_to_siegbahn = {
        iupac: siegbahn
        for siegbahn, iupac_list in siegbahn_to_iupac_data.items()
        for iupac in iupac_list
    }

    if input_type == "new_values":
        for el_line, value in lines_new_values.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line_name = line_match.group(2)

            atomic_number = str(atomic_symbols_data["table"][element]["number"])
            line_iupac = siegbahn_to_iupac_data.get(line_name, [line_name])[0]

            if (
                atomic_number in cross_sections_data["table"]
                and line_iupac in cross_sections_data["table"][atomic_number]
            ):
                cross_sections_data["table"][atomic_number][line_iupac]["cs"] = value

                line_siegbahn = iupac_to_siegbahn.get(line_iupac, line_iupac)

                print(
                    f"Set new cross-section for Element {element}, Line {line_siegbahn} to {value:.2e}"
                )
            else:
                print(
                    f"Warning: Element {element} or line {line_iupac} not found in the JSON data."
                )

    elif input_type == "scaling_factors":
        for el_line, factor in lines_scaling_factors.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line_group = line_match.group(2)

            atomic_number = str(atomic_symbols_data["table"][element]["number"])
            line_iupac_list = siegbahn_to_iupac_data.get(line_group, [])

            for line_iupac in line_iupac_list:
                if (
                    atomic_number in cross_sections_data["table"]
                    and line_iupac in cross_sections_data["table"][atomic_number]
                ):
                    current_cs = cross_sections_data["table"][atomic_number][
                        line_iupac
                    ]["cs"]
                    cross_sections_data["table"][atomic_number][line_iupac]["cs"] = (
                        current_cs * factor
                    )

                    line_siegbahn = iupac_to_siegbahn.get(line_iupac, line_iupac)

                    print(
                        f"Applied scaling factor {factor} to cross-section for element {element}, line {line_siegbahn}"
                    )

                else:
                    print(
                        f"Skipped element {element}, line {line_iupac}: No matching data in database."
                    )

    elif input_type == "k_factors" and reference_line is not None:
        ref_match = re.match(r"([A-Z][a-z]?)_(.*)", reference_line)
        ref_element = ref_match.group(1)
        ref_line_family = ref_match.group(2)

        ref_at_num = str(atomic_symbols_data["table"][ref_element]["number"])

        ref_lines_iupac = siegbahn_to_iupac_data.get(ref_line_family, [])
        if not ref_lines_iupac:
            raise ValueError(
                f"Reference line family {ref_line_family} not found in Siegbahn to IUPAC mapping."
            )

        valid_ref_lines = {
            line: cross_sections_data["table"][ref_at_num][line]["cs"]
            for line in ref_lines_iupac
            if line in cross_sections_data["table"][ref_at_num]
        }

        if not valid_ref_lines:
            raise ValueError(
                f"No valid reference lines found for {ref_element}_{ref_line_family} in the cross-section data."
            )

        for el_line, factor in k_factors.items():
            line_match = re.match(r"([A-Z][a-z]?)_(.*)", el_line)
            element = line_match.group(1)
            line_family = line_match.group(2)

            atomic_number = str(atomic_symbols_data["table"][element]["number"])

            target_lines_iupac = siegbahn_to_iupac_data.get(line_family, [])

            if not target_lines_iupac:
                raise ValueError(
                    f"Target line family {line_family} not found in Siegbahn to IUPAC mapping."
                )

            for ref_line, target_line in zip(ref_lines_iupac, target_lines_iupac):
                if (
                    ref_line in valid_ref_lines
                    and target_line in cross_sections_data["table"][atomic_number]
                ):
                    ref_cs = valid_ref_lines[ref_line]
                    new_value = ref_cs * factor
                    cross_sections_data["table"][atomic_number][target_line]["cs"] = (
                        new_value
                    )

                    ref_line_siegbahn = iupac_to_siegbahn.get(ref_line, ref_line)
                    target_line_siegbahn = iupac_to_siegbahn.get(
                        target_line, target_line
                    )

                    print(
                        f"Set new cross-section for element {element}, line {target_line_siegbahn} "
                        f"based on k-factor {factor} using {ref_element}, line {ref_line_siegbahn} as reference."
                    )
                else:
                    print(
                        f"Skipped element {element}, line {target_line}: No matching data in reference or target."
                    )

    original_files_list = [
        "SDD_efficiency.txt",
        "200keV_xrays.json",
        "__init__.py",
        "default_xrays.json",
        "periodic_table_symbols.json",
        "siegbahn_to_iupac.json",
        "300keV_xrays.json",
        "periodic_table_number.json",
        "100keV_xrays.json",
    ]

    if output_filename is None:
        input_filename = cross_sections_json.name
        output_filename = input_filename.replace(".json", "_modified.json")

    if output_filename in original_files_list:
        raise ValueError(
            "The output filename cannot be the same as one of the original files."
        )

    with open(DB_PATH / output_filename, "w") as file:
        json.dump(cross_sections_data, file, indent=4)


def load_table(db_name: str) -> tuple[dict, dict]:
    r"""
    Load the table and metadata of a json table generated by emtables.

    Parameters
    ----------
    db_name : str
        The file name of the table to load.

    Returns
    -------
    table : dict
        The table of the cross sections.
    metadata : dict
        The metadata of the table.

    Notes
    -----
    Call espm.conf.DB_PATH to get the folder of the tables.
    """
    db_path = DB_PATH / Path(db_name)
    with open(db_path, "r") as f:
        json_dict = json.load(f)
    return json_dict["table"], json_dict["metadata"]


def import_k_factors(
    table: dict,
    mdata: dict,
    k_factors_names: list[str],
    k_factors_values: list[int],
    ref_name: str,
) -> tuple[dict, dict]:
    r"""
    Modify the X-ray emission cross-sections of the input table using the k-factors input, i.e. imposing cross-sections ratios to correspond to the k-factors.
    The metadata are modified too to keep track of the modifications.

    Parameters
    ----------
    table : dict
        The table of the X-ray emission cross sections.
    mdata : dict
        The metadata of the table.
    k_factors_names : list
        The list of the names of the k-factors to import. It has to correspond to the nomenclature of the hyperspy X-ray lines.
    k_factors_values : list
        The list of the values of the k-factors to import. It has to have the same length and ordering as k_factors_names.
    ref_name : str
        The name of the X-ray line to use as a reference for the k-factors. It has to correspond to the nomenclature of the hyperspy X-ray lines.

    Returns
    -------
    new_table : dict
        The modified table of the X-ray emission cross sections.
    new_mdata : dict
        The modified metadata of the table.
    """

    with open(SYMBOLS_PERIODIC_TABLE, "r") as f:
        SPT = json.load(f)["table"]

    with open(SIEGBAHN_TO_IUPAC, "r") as f:
        STI = json.load(f)

    for i, name in enumerate(k_factors_names):
        if name == ref_name:
            mr = re.match(r"([A-Z][a-z]?)_(.*)", name)
            ref_at_num = SPT[mr.group(1)]["number"]
            ref_lines = STI[mr.group(2)]
            ref_sig_vals = []
            for l in ref_lines:
                if l in table[str(ref_at_num)]:
                    ref_sig_vals.append(table[str(ref_at_num)][l]["cs"])
            ref_sig_val = np.mean(ref_sig_vals)
            ref_k_val = k_factors_values[i]

    for i, name in enumerate(k_factors_names):
        m0 = re.match(r"([A-Z][a-z]?)_(.*)", name)
        if m0:
            at_num = SPT[m0.group(1)]["number"]
            lines = STI[m0.group(2)]
            for line in lines:
                new_k = k_factors_values[i] / ref_k_val
                if line in table[str(at_num)]:
                    sig_val = table[str(at_num)][line]["cs"]
                    new_value = ref_sig_val * new_k / sig_val
                    new_table, new_mdata = modify_table_lines(
                        table, mdata, [at_num], line, new_value
                    )
    return new_table, new_mdata


def modify_table_lines(
    table: dict, mdata: dict, elements: list[str], line: str, coeff: float
) -> tuple[dict, dict]:
    r"""
    Modify the cross section of the lines of the selected elements in the input table.

    Parameters
    ----------
    table : dict
        The table of the X-ray emission cross sections.
    mdata : dict
        The metadata of the table.
    elements : list
        The list of the atomic numbers of the elements to modify.
    line : str
        The regex of the line to modify. It has to correspond to IUPAC notation.
    coeff : float
        The coefficient to multiply the cross section of the selected lines.

    Returns
    -------
    new_table : dict
        The modified table of the X-ray emission cross sections.
    new_mdata : dict
        The modified metadata of the table.

    Notes
    -----
    X-ray line regex examples :  input "L" will modify all the L lines, input "L3" will modifiy all the L3 lines,
    input "L3M2" will modify the "L3M2" line.
    """
    if mdata["lines"]:
        for elt in elements:
            for key in table[str(elt)].keys():
                if re.match(r"^{}".format(line), key):
                    table[str(elt)][key]["cs"] *= coeff
                    if "modifications" in mdata:
                        mdata["modifications"][str(elt) + "_" + key] = coeff
                    else:
                        mdata["modifications"] = {}
                        mdata["modifications"][str(elt) + "_" + key] = coeff

    else:
        print("You need to enable line notation")
    return table, mdata


def save_table(filename: str, table: dict, mdata: dict) -> None:
    r"""
    Saves a table and its metadata in a json file.
    The structure of the json file is compliant with espm.
    """
    d = {}
    d["table"] = table
    d["metadata"] = mdata
    with open(filename, "w") as f:
        json.dump(d, f, indent=4)


def get_k_factor(
    table: dict,
    mdata: dict,
    element: int,
    line: str,
    range: float = 0.5,
    ref_elt: str = "14",
    ref_line: str = "KL3",
    ref_range: float = 0.5,
):
    r"""
    Obtain the k-factor of a line from an emtables, X-ray emission cross section table.

    Parameters
    ----------
    table : dict
        The table of the X-ray emission cross sections.
    mdata : dict
        The metadata of the table.
    element : int
        The atomic number of the element to use.
    line : str
        The regex of the line to use. It has to correspond to IUPAC notation.
    range : float
        The energy range to use for the integration of the cross section of the line. For example, if range = 0.5, the integration will be done between the energy of the line - 0.5 and the energy of the line + 0.5. We do so that when you select the "KL3" line, it integrates around it and make it correspond to the K-alpha bunch of lines.
    ref_elt : str
        The atomic number of the element to use as a reference for the k-factor. The default reference line is Si "KL3" with an integration range of 0.5.
    ref_line : str
        The regex of the line to use as a reference for the k-factor. It has to correspond to IUPAC notation.
    ref_range : float
        The energy range to use for the integration of the cross section of the reference line.

    Returns
    -------
    k_factor : float
        The k-factor of the line. It does not take into account the absorption correction.
    """
    ref_cs = 0.0
    cs = 0.0
    if mdata["lines"]:
        ref_en = table[str(ref_elt)][ref_line]["energy"]
        for key in table[str(ref_elt)].keys():
            en = table[str(ref_elt)][key]["energy"]
            if (en < ref_en + ref_range) and (en > ref_en - ref_range):
                ref_cs += table[str(ref_elt)][key]["cs"]

        elt_en = table[str(element)][line]["energy"]
        for key in table[str(element)].keys():
            en = table[str(element)][key]["energy"]
            if (en < elt_en + range) and (en > elt_en - range):
                cs += table[str(element)][key]["cs"]

    else:
        print("You need to enable line notation")

    return cs / ref_cs
