# -*- coding: utf-8 -*-
"""
Script for cleaning and transforming the SAMHSA TEDS-D data for ML input
"""

import pandas as pd
import numpy as np
import os

# Configuration
PATH = "../data/"
FILE_NAME = "tedsd_puf_2022.csv"
OUTPUT_OPTION = 1  # 0 = continuous LOS, 1 = categorical LOS

# Column groups to remove
REMOVE_DISCHARGE = [
    "ARRESTS_D", "DETNLF_D", "EMPLOY_D", "FREQ1_D", "FREQ2_D", "FREQ3_D",
    "FREQ_ATND_SELF_HELP_D", "LIVARAG_D", "SERVICES_D", "SUB1_D",
    "SUB2_D", "SUB3_D"
]
REMOVE_REDUNDANT = ["CASEID", "DISYR", "DIVISION"]
REMOVE_INFLATED = ["CBSA2020", "STFIPS"]

# Columns to check for missing values
DROP_IF_MISSING = [
    "ARRESTS", "EDUC", "EMPLOY", "ETHNIC", "FREQ1", "FREQ_ATND_SELF_HELP",
    "LIVARAG", "MARSTAT", "METHUSE", "NOPRIOR", "PRIMINC", "PSOURCE",
    "PSYPROB", "RACE", "VET"
]


def load_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def filter_services_match(df):
    return df[df["SERVICES"] == df["SERVICES_D"]].copy()


def drop_unwanted_columns(df):
    for column in REMOVE_DISCHARGE + REMOVE_REDUNDANT + REMOVE_INFLATED:
        df.drop(column, axis=1, inplace=True, errors='ignore')
    return df


def merge_detailed_categories(df):
    df["DETCRIM"] = np.where(df["DETCRIM"] == -9, 0, df["DETCRIM"])
    df["PSOURCE"] = np.where(df["PSOURCE"] == 7, df["PSOURCE"] + df["DETCRIM"], df["PSOURCE"])
    df["DETNLF"] = np.where(df["DETNLF"] == -9, 0, df["DETNLF"])
    df["EMPLOY"] = np.where(df["EMPLOY"] == 4, df["EMPLOY"] + df["DETNLF"], df["EMPLOY"])
    return df


def impute_data(df):
    df["FREQ2"] = np.where(df["FREQ2"] == -9, 1, df["FREQ2"])
    df["FREQ3"] = np.where(df["FREQ3"] == -9, 1, df["FREQ3"])
    df["PRIMINC"] = np.where((df["PRIMINC"] == -9) & (df["EMPLOY"].isin([1, 2])), 1, df["PRIMINC"])
    df["DAYWAIT"] = np.where(df["DAYWAIT"] == -9, 0, df["DAYWAIT"])
    df["PREG"] = np.where(df["PREG"] == -9, 2, df["PREG"])
    df["IDU"] = np.where(df["IDU"] == -9, 0, df["IDU"])

    sub_route_pairs = [(2, 1), (6, 1), (17, 3), (18, 1)]
    for sub, route in sub_route_pairs:
        for i in ["ROUTE1", "ROUTE2", "ROUTE3"]:
            df[i] = np.where((df[i] == -9) & (df[f"SUB{i[-1]}"] == sub), route, df[i])
    for i in ["ROUTE1", "ROUTE2", "ROUTE3"]:
        df[i] = np.where(df[i] == -9, 0, df[i])

    return df


def transform_los(df, output_option):
    if output_option == 0:
        los_map = {
            31: (31+45)/2, 32: (46+60)/2, 33: (61+90)/2, 34: (91+120)/2,
            35: (121+180)/2, 36: (181+365)/2, 37: 365
        }
    else:
        los_map = {
            31: 4, 32: 5, 33: 6, 34: 7, 35: 8, 36: 9, 37: 10
        }
        df["LOS"] = np.where((df["LOS"] < 16) & (df["LOS"] != 1), 2, df["LOS"])
        df["LOS"] = np.where((df["LOS"] >= 16) & (df["LOS"] < 31), 3, df["LOS"])

    df["LOS"] = df["LOS"].replace(los_map)
    return df


def one_hot_encode(df, exclude=[]):
    return pd.get_dummies(df, columns=[col for col in df.columns if col not in exclude])


def save_by_service_type(df, path, output_option):
    for service in [2, 4, 5, 6, 7]:
        subset = df[df["SERVICES"] == service].drop("SERVICES", axis=1)
        np.save(os.path.join(path, f"S{service}_data_{output_option}"), subset.reset_index(drop=True))


def main():
    data = load_data(PATH, FILE_NAME)
    data = filter_services_match(data)
    data = drop_unwanted_columns(data)
    data = merge_detailed_categories(data)
    data = impute_data(data)
    data = transform_los(data, OUTPUT_OPTION)
    data = one_hot_encode(data, exclude=["LOS", "SERVICES"])

    save_by_service_type(data, PATH, OUTPUT_OPTION)

    data = pd.get_dummies(data, columns=["SERVICES"])
    np.save(os.path.join(PATH, f"all_data_{OUTPUT_OPTION}.npy"), data)


if __name__ == "__main__":
    main()
