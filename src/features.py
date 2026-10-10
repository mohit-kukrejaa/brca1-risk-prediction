import pandas as pd

# Kyte-Doolittle hydropathy scale
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2
}

# Zamyatnin amino-acid residue volumes
AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0
}

# Grantham: composition c polarity p, volume v
GRANTHAM_PROPERTIES = {
    "A": (0.00, 8.1, 31.0), "R": (0.65, 10.5, 124.0),
    "N": (1.33, 11.6, 56.0), "D": (1.38, 13.0, 54.0),
    "C": (2.75, 5.5, 55.0), "Q": (0.89, 10.5, 85.0),
    "E": (0.92, 12.3, 83.0), "G": (0.74, 9.0, 3.0),
    "H": (0.58, 10.4, 96.0), "I": (0.00, 5.2, 111.0),
    "L": (0.00, 4.9, 111.0), "K": (0.33, 11.3, 119.0),
    "M": (0.00, 5.7, 105.0), "F": (0.00, 5.2, 132.0),
    "P": (0.39, 8.0, 32.5), "S": (1.42, 9.2, 32.0),
    "T": (0.71, 8.6, 61.0), "W": (0.13, 5.4, 170.0),
    "Y": (0.20, 6.2, 136.0), "V": (0.00, 5.9, 84.0)
}

# BLOSUM62
AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

BLOSUM62_VALUES = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4]
]

BLOSUM62 = {
    (ref, alt): BLOSUM62_VALUES[i][j]
    for i, ref in enumerate(AMINO_ACIDS)
    for j, alt in enumerate(AMINO_ACIDS)
}

def calculate_grantham_distance(ref_aa, alt_aa):
    ref_c, ref_p, ref_v = GRANTHAM_PROPERTIES[ref_aa]
    alt_c, alt_p, alt_v = GRANTHAM_PROPERTIES[alt_aa]

    composition_diff = ref_c - alt_c
    polarity_diff = ref_p - alt_p
    volume_diff = ref_v - alt_v

    distance = 50.723 * (1.833 * (composition_diff**2) + 0.1018 * (polarity_diff**2) + 0.000399 * (volume_diff**2)) ** 0.5

    return round(distance)


def add_engineered_features(df):
    df = df.copy()

    # Structural features
    df["plddt_delta"] = (df["plddt_at_pos"] - df["plddt_window11"])

    # Assuming window11 = current residue + 10 neighboring residues
    neighbor_mean = ((df["plddt_window11"] * 11 - df["plddt_at_pos"]) / 10)
    df["plddt_neighbor_delta"] = (df["plddt_at_pos"] - neighbor_mean)

    # Evolutionary substitution score
    df["blosum62_score"] = [BLOSUM62[(ref, alt)] for ref, alt in zip(df["ref_aa"], df["alt_aa"])]

    # Grantham substitution severity
    df["grantham_distance"] = [calculate_grantham_distance(ref, alt) for ref, alt in zip(df["ref_aa"], df["alt_aa"])]

    # Signed physicochemical changes
    df["hydro_signed_delta"] = [HYDROPHOBICITY[alt] - HYDROPHOBICITY[ref] for ref, alt in zip(df["ref_aa"], df["alt_aa"])]
    df["volume_signed_delta"] = [AA_VOLUME[alt] - AA_VOLUME[ref] for ref, alt in zip(df["ref_aa"], df["alt_aa"])]

    # Functional-domain indicator
    df["in_functional_domain"] = (df["domain"] != "OTHER").astype(int)

    # Interaction features
    df["grantham_plddt"] = (df["grantham_distance"] * df["plddt_at_pos"])
    df["blosum_plddt"] = (df["blosum62_score"] * df["plddt_at_pos"])

    negative_blosum = (-df["blosum62_score"].clip(upper=0))
    df["blosum_unfavorable_plddt"] = (negative_blosum * df["plddt_at_pos"])

    df["grantham_domain"] = (df["grantham_distance"] * df["in_functional_domain"])

    return df