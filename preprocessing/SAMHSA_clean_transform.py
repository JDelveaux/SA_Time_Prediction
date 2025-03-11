# -*- coding: utf-8 -*-
"""
Cleans and transforms the SAMHSA TEDS-D data to ML input

@author: jaked
"""

import pandas as pd
import numpy as np

PATH = "../data/"
FILE_NAME = "tedsd_puf_2022.csv"
#Choose if output is continuous (0) or categorical (1)
output_option = 1

with open(PATH + FILE_NAME) as file:
    data = pd.read_csv(file)

#FILTER SERVICES == SERVICES_D
services_differ_index = data[data["SERVICES"] != data["SERVICES_D"]].index
data.drop(services_differ_index, inplace = True)

#REMOVE CATEGORIES
remove_discharge = [
    "ARRESTS_D",
    "DETNLF_D",
    "EMPLOY_D",
    "FREQ1_D",
    "FREQ2_D",
    "FREQ3_D",
    "FREQ_ATND_SELF_HELP_D",
    "LIVARAG_D",
    "SERVICES_D",
    "SUB1_D",
    "SUB2_D",
    "SUB3_D"
    ]
remove_redundant = [
    "CASEID",
    "DISYR",
    "DIVISION"
    ]

remove_inflated_variables = [
    "CBSA2020",
    "STFIPS"
    ]

#OPTIONAL
"""
remove_missing_data = [
    "DSMCRIT",
    "HLTHINS",
    "PRIMPAY",
    ]
"""

for column in remove_discharge + remove_redundant + remove_inflated_variables:
    del data[column]

#MERGE DETAILED CATEGORIES "DETCRIM"->"PSOURCE" and "DETNLF"->"EMPLOY"
#Set missing data to 0, defaults "7" for undetailed criminal referral
data['DETCRIM'] = np.where(data['DETCRIM'] == -9, 0, data["DETCRIM"])
data['PSOURCE'] = np.where(data['PSOURCE'] == 7, data['PSOURCE'] + data['DETCRIM'], data["PSOURCE"])
#Do the same for EMPLOY/DETNLF
data['DETNLF'] = np.where(data['DETNLF'] == -9, 0, data["DETNLF"])
data['EMPLOY'] = np.where(data['EMPLOY'] == 4, data['EMPLOY'] + data['DETNLF'], data["EMPLOY"])

#IMPUTE DATA FOR FREQUENCY, PRIMINC, DAYWAIT, PREG, IDU, and ROUTE
#assume frequency of use for missing data for secondary/tertiary substances is "none"
data["FREQ2"] = np.where(data["FREQ2"] == -9, 1, data["FREQ2"])
data["FREQ3"] = np.where(data["FREQ3"] == -9, 1, data["FREQ3"])
#assume part-time/full-time job --> primary income is wages
data["PRIMINC"] = np.where((data["PRIMINC"] == -9) & (data["EMPLOY"] == 1), 1, data["PRIMINC"])
data["PRIMINC"] = np.where((data["PRIMINC"] == -9) & (data["EMPLOY"] == 2), 1, data["PRIMINC"])
#assume unrecorded daywait --> zero days wait
data["DAYWAIT"] = np.where(data["DAYWAIT"] == -9, 0, data["DAYWAIT"])
#set male/missing data to "not pregnant"
data["PREG"] = np.where(data["PREG"] == -9, 2, data["PREG"])
#assume no IDU if missing data
data["IDU"] = np.where(data["IDU"] == -9, 0, data["IDU"])
#make educated guess of substance use route
sub_route_pairs = [(2,1), (6,1), (17,3), (18,1)]
for pair in sub_route_pairs:
    i,j = pair
    data["ROUTE1"] = np.where((data["ROUTE1"] == -9) & (data["SUB1"] == i), j, data["ROUTE1"])
    data["ROUTE2"] = np.where((data["ROUTE2"] == -9) & (data["SUB2"] == i), j, data["ROUTE2"])
    data["ROUTE3"] = np.where((data["ROUTE3"] == -9) & (data["SUB3"] == i), j, data["ROUTE3"])
#remaining values assumed "no substance", set to category 0
data["ROUTE1"] = np.where(data["ROUTE1"] == -9, 0, data["ROUTE1"])
data["ROUTE2"] = np.where(data["ROUTE2"] == -9, 0, data["ROUTE2"])
data["ROUTE3"] = np.where(data["ROUTE3"] == -9, 0, data["ROUTE3"])


#REMOVE MISSING DATA
def check_row(row):
    drop_cat = [
        "ARRESTS",
        "EDUC",
        "EMPLOY",
        "ETHNIC",
        "FREQ1",
        "FREQ_ATND_SELF_HELP",
        "LIVARAG",
        "MARSTAT",
        "METHUSE",
        "NOPRIOR",
        "PRIMINC",
        "PSOURCE",
        "PSYPROB",
        "RACE",
        "VET"
        ]
    return any(row[col] in [-9] for col in drop_cat)
#data = data[~data.apply(check_row, axis=1)]

#OPTION 1: AVERAGE LENGTH OF STAY Cats 31-37
if output_option == 0:
    data["LOS"] = np.where(data["LOS"] == 31, (31+45)/2.0, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 32, (46+60)/2.0, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 33, (61+90)/2.0, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 34, (91+120)/2.0, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 35, (121+180)/2.0, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 36, (181+365)/2.0, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 37, 365, data["LOS"])

#OPTION 2: CONVERT LOS TO CATEGORIES (1 day, 2-30 days, 30-60 days,...)
if output_option == 1:
    data["LOS"] = np.where((data["LOS"] < 16) & (data["LOS"] != 1), 2, data["LOS"])
    data["LOS"] = np.where((data["LOS"] < 31) & (data["LOS"] >= 16), 3, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 31, 4, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 32, 5, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 33, 6, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 34, 7, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 35, 8, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 36, 9, data["LOS"])
    data["LOS"] = np.where(data["LOS"] == 37, 10, data["LOS"])

#ONE-HOT ENCODING
#OHE all categories excluding SERVICES and LOS
ohe_exclude_cats = ["SERVICES", "LOS"]
ohe_include_cats = data.columns.tolist()
data = pd.get_dummies(data, columns=[cat for cat in ohe_include_cats if cat not in ohe_exclude_cats])

#SEPARATE SERVICES and SAVE DATA
S2 = data[data["SERVICES"] == 2].drop(["SERVICES"], axis=1).reset_index(drop=True)
np.save(PATH + "S2_data_" + str(output_option), S2)
del S2
S4 = data[data["SERVICES"] == 4].drop(["SERVICES"], axis=1).reset_index(drop=True)
np.save(PATH + "S4_data_" + str(output_option), S4)
del S4
S5 = data[data["SERVICES"] == 5].drop(["SERVICES"], axis=1).reset_index(drop=True)
np.save(PATH + "S5_data_" + str(output_option), S5)
del S5
S6 = data[data["SERVICES"] == 6].drop(["SERVICES"], axis=1).reset_index(drop=True)
np.save(PATH + "S6_data_" + str(output_option), S6)
del S6
S7 = data[data["SERVICES"] == 7].drop(["SERVICES"], axis=1).reset_index(drop=True)
np.save(PATH + "S7_data_" + str(output_option), S7)
del S7

#OHE SERVICES AND SAVE DATA
data = pd.get_dummies(data, columns=["SERVICES"])
np.save(PATH + "all_data_" + str(output_option) + ".npy", data)