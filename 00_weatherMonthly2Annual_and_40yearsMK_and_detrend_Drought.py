# ---------------------
# August 12, 2025. Added drought to this notebook that was previousy called
# ```00_weather_monthly2Annual_and_40yearsMK_and_detrend.ipynb```.
# Now we have to add drought indices to output.


#
# This was called ``MK_weather``. We are adding min and max of weather stuff and numbering notebooks
#
# Jul. 11, 2025
#
# ------------------

# %%
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, os.path, pickle, sys
import pymannkendall as mk
from scipy.stats import variation
from scipy import stats

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import geopandas

import matplotlib.colors as colors
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm

from datetime import datetime
from datetime import date
import time
from functools import reduce
from sklearn.linear_model import LinearRegression

sys.path.append("/home/h.noorazar/rangeland/")
import rangeland_core as rc

start_time = time.time()

# %%
dpi_, map_dpi_ = 300, 500
custom_cmap_coral = ListedColormap(["lightcoral", "black"])
custom_cmap_BW = ListedColormap(["white", "black"])
cmap_G = cm.get_cmap("Greens")  # 'PRGn', 'YlGn'
cmap_R = cm.get_cmap("Reds")

#####################################################################################
#####################################################################################
#####################################################################################
research_data_ = "/data/project/agaid/h.noorazar/"
rangeland_bio_base = research_data_ + "rangeland_bio/"
rangeland_bio_data = rangeland_bio_base + "Data/"
bio_reOrganized = rangeland_bio_data + "reOrganized/"
common_data = research_data_ + "common_data/"
#####################################################################################
#####################################################################################
#####################################################################################
"""
   ANPP is from 1984. So, Let us filter things from 1984.
   This problem came up when I included the drought indices that 
   Min gave us was from 1984. So, here when I merged them with weather data
   we have a few years of NAs. Hence, some error

"""
#####################################################################################
#####################################################################################
#####################################################################################

county_fips_dict = pd.read_pickle(common_data + "county_fips.sav")
county_fips = county_fips_dict["county_fips"]
full_2_abb = county_fips_dict["full_2_abb"]
abb_2_full_dict = county_fips_dict["abb_2_full_dict"]
abb_full_df = county_fips_dict["abb_full_df"]
filtered_counties_29States = county_fips_dict["filtered_counties_29States"]
SoI = county_fips_dict["SoI"]
state_fips = county_fips_dict["state_fips"]

state_fips = state_fips[state_fips.state != "VI"].copy()
state_fips.head(2)

### Read the shapefile
# And keep the vegtype in subsequent dataframes
# %%time
Albers_SF_name = bio_reOrganized + "Albers_BioRangeland_Min_Ehsan"
Albers_SF = geopandas.read_file(Albers_SF_name)
Albers_SF.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)
Albers_SF.rename(
    columns={"minstatsid": "fid", "satae_max": "state_majority_area"}, inplace=True
)
Albers_SF.head(2)

len(Albers_SF["fid"].unique())

## Focus only on West Meridian
print((Albers_SF["state_majority_area"] == Albers_SF["state_1"]).sum())
print((Albers_SF["state_majority_area"] == Albers_SF["state_2"]).sum())
print(Albers_SF.shape)
print(len(Albers_SF) - (Albers_SF["state_1"] == Albers_SF["state_2"]).sum())
print((Albers_SF["state_1"] == Albers_SF["state_2"]).sum())

# %%
Albers_SF = pd.merge(
    Albers_SF,
    state_fips[["EW_meridian", "state_full"]],
    how="left",
    left_on="state_majority_area",
    right_on="state_full",
)

Albers_SF.drop(columns=["state_full"], inplace=True)

print(Albers_SF.shape)
Albers_SF.head(2)

# %%
Albers_SF = Albers_SF[Albers_SF["EW_meridian"] == "W"].copy()
Albers_SF.shape

# %%
print(len(Albers_SF["fid"].unique()))
print(len(Albers_SF["value"].unique()))
print(len(Albers_SF["hucsgree_4"].unique()))

print((Albers_SF["hucsgree_4"] - Albers_SF["value"]).unique())
print((list(Albers_SF.index) == Albers_SF.fid).sum())

Albers_SF.drop(columns=["value"], inplace=True)
Albers_SF.head(2)

# ## Read weather Data

# %%
filename = bio_reOrganized + "bps_weather.sav"
bps_weather = pd.read_pickle(filename)
bps_weather = bps_weather["bps_weather"]
bps_weather["fid"].unique()[-8::]
############
############   Filter weather data from 1984. Aug. 14. 2025 this filter was added.
############   Do we want to do this? or just use less data for drought indices.
############
bps_weather = bps_weather[bps_weather["year"] >= 1984].copy()

# %%
west_FIDs = list(Albers_SF["fid"])
bps_weather = bps_weather[bps_weather["fid"].isin(west_FIDs)]
bps_weather.reset_index(drop=True, inplace=True)
bps_weather.head(2)

# %%
sorted(bps_weather.columns)

# %%
annual_weather = (
    bps_weather.groupby(["fid", "year"])
    .agg(
        {
            "avg_of_dailyAvg_rel_hum": "mean",
            "avg_of_dailyAvgTemp_C": "mean",
            "thi_avg": "mean",
            "avg_of_dailyMaxTemp_C": "max",
            "avg_of_dailyMinTemp_C": "min",
            "max_of_dailyMaxTemp_C": "mean",
            "min_of_dailyMinTemp_C": "mean",
            "precip_mm_month": "sum",
        }
    )
    .reset_index()
)

annual_weather.rename(
    columns={
        "precip_mm_month": "precip_mm",
        "avg_of_dailyMaxTemp_C": "max_of_monthlyAvg_of_dailyMaxTemp_C",
        "avg_of_dailyMinTemp_C": "min_of_monthlyAvg_of_dailyMinTemp_C",
        "max_of_dailyMaxTemp_C": "avg_of_monthlymax_of_dailyMaxTemp_C",
        "min_of_dailyMinTemp_C": "avg_of_monthlymin_of_dailyMinTemp_C",
    },
    inplace=True,
)
annual_weather.head(3)

# %% [markdown]
#### Check if all locations have all years in it
len(annual_weather[annual_weather.fid == 1])
annual_weather.head(2)

# %%
num_locs = len(annual_weather["fid"].unique())
num_locs

# %%
cols_ = ["fid", "state_majority_area", "state_1", "state_2", "EW_meridian"]
if not ("EW_meridian" in annual_weather.columns):
    annual_weather = pd.merge(annual_weather, Albers_SF[cols_], how="left", on="fid")

annual_weather.drop(columns=["state_1", "state_2"], inplace=True)
annual_weather.head(2)

#####################################################################################
#####################################################################################
#####################################################################################
### Read Drought


drought_wide = pd.read_pickle(bio_reOrganized + "drought_wide.sav")
drought_wide = drought_wide["drought_wide"]
drought_wide.head(2)

annual_weather.head(2)

# # MK test and Spearman's rank for Weather
sorted(annual_weather.columns)

print(annual_weather.shape)
annual_weather = pd.merge(annual_weather, drought_wide, how="left", on=["fid", "year"])
print(annual_weather.shape)

non_ys = ["EW_meridian", "fid", "year", "state_majority_area"]
y_vars = [x for x in annual_weather.columns if (not (x in non_ys))]
len_y_vars = len(y_vars)
#####################################################################################
#####################################################################################
#####################################################################################
############
############ MK on weather variables
############
#################################################################
################################################################# Beginning of Done Part
#################################################################
count = 1
all_treds_dict = {}
MK_test_cols = [
    "sens_slope",
    "sens_intercept",
    "Tau",
    "MK_score",
    "trend",
    "p",
    "var_s",
]

for y_var in y_vars:
    MK_df = annual_weather[["fid"]].copy()
    MK_df.drop_duplicates(inplace=True)
    MK_df.reset_index(drop=True, inplace=True)

    ##### z: normalized test statistics
    ##### Tau: Kendall Tau
    MK_df = pd.concat([MK_df, pd.DataFrame(columns=MK_test_cols)])
    MK_df[MK_test_cols] = ["-666"] + [-666] * (len(MK_test_cols) - 1)

    # Why data type changed?!
    MK_df["fid"] = MK_df["fid"].astype(np.int64)
    ###############################################################
    # populate the dataframe with MK test result now
    for a_FID in MK_df["fid"].unique():
        precip_TS = annual_weather.loc[annual_weather.fid == a_FID, y_var].values
        year_TS = annual_weather.loc[annual_weather.fid == a_FID, "year"].values

        # MK test original
        trend, _, p, z, Tau, MK_score, var_s, slope, intercept = mk.original_test(
            precip_TS
        )
        # Spearman, p_Spearman = stats.spearmanr(year_TS, precip_TS) # Spearman's rank

        # Update dataframe by MK result
        L_ = [slope, intercept, Tau, MK_score, trend, p, var_s]

        MK_df.loc[MK_df["fid"] == a_FID, MK_test_cols] = L_

        del (slope, intercept, Tau, MK_score, trend, p, var_s)
        del (L_, precip_TS, year_TS)

    # Round the columns to 4-decimals
    for a_col in ["sens_slope", "sens_intercept", "Tau", "MK_score", "p", "var_s"]:
        MK_df[a_col] = MK_df[a_col].astype(float)
        MK_df[a_col] = round(MK_df[a_col], 4)

    MK_df.rename(
        columns={
            col: col + "_" + y_var if col != "fid" else col for col in MK_df.columns
        },
        inplace=True,
    )
    key_ = "MK_" + y_var
    all_treds_dict[key_] = MK_df
    print(f"{count} out of {len_y_vars}")
    count += 1
    print("================================================================")

# temp_ACF_trends_MK_dict[list(temp_ACF_dict.keys())[0]].head(3)
# Convert dict values to a list of DataFrames
df_list = list(all_treds_dict.values())

# Perform left merges iteratively
weather_MK_df = reduce(
    lambda left, right: pd.merge(left, right, on="fid", how="left"), df_list
)

# filename = bio_reOrganized + "weather_MK_Spearman.sav"
filename = bio_reOrganized + "weather_drought_MK_Spearman.sav"  # Aug 12, 2025

export_ = {
    "weather_drought_MK_df": weather_MK_df,
    # "source_code" : "00_weather_monthly2Annual_and_40yearsMK_and_detrend",
    "source_code": "00_weatherMonthly2Annual_and_40yearsMK_and_detrend_Drought",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

print("line 317")
print(filename)
pickle.dump(export_, open(filename, "wb"))
print("line 320")

################################################################# End of Done Part
print(Albers_SF.shape)
print(weather_MK_df.shape)

## Detrend
#### Add detrending to this notebook from ```deTrend_weather.ipynb```
weather_MK_df.head(2)
annual_weather.head(2)

sens_cols = ["fid"] + [
    x for x in weather_MK_df.columns if ("slope" in x) or ("intercept" in x)
]
sens_cols
annual_weather_detrend = annual_weather.copy()
annual_weather_detrend = pd.merge(
    annual_weather_detrend, weather_MK_df[sens_cols], how="left", on="fid"
)
annual_weather_detrend.head(2)

# ### Sens prediction
# must not be based on year since that test only lookst at y values.
annual_weather_detrend["row_number_perfid"] = annual_weather_detrend.groupby(
    "fid"
).cumcount()
annual_weather_detrend.head(2)
sorted(annual_weather.columns)

print("line 349")
for y_var in y_vars:
    annual_weather_detrend[f"{y_var}_senPred"] = (
        annual_weather_detrend["row_number_perfid"]
        * annual_weather_detrend[f"sens_slope_{y_var}"]
        + annual_weather_detrend[f"sens_intercept_{y_var}"]
    )
    annual_weather_detrend[f"{y_var}_detrendSens"] = (
        annual_weather_detrend[y_var] - annual_weather_detrend[f"{y_var}_senPred"]
    )

annual_weather_detrend.head(2)
############################################################
############################################################
############################################################
######
###### detrend using Simple Linear regression
######
print("line 367")
unique_fids = annual_weather_detrend["fid"].unique()
len(unique_fids)
# regression_df is optional to save slopes and intercepts
regression_df = pd.DataFrame({"fid": unique_fids})
for y_var in y_vars:
    regression_df[f"{y_var}_linReg_slope"] = np.nan
    # Prepare a column to store detrended values
    annual_weather_detrend[f"{y_var}_detrendLinReg"] = np.nan

regression_df = regression_df.set_index("fid")
regression_df.head(2)
annual_weather_detrend.head(2)
print("line 380")
# Loop over each fid group
for fid, group in annual_weather_detrend.groupby("fid"):
    for y_var in y_vars:
        # Reshape year for sklearn
        X = group["year"].values.reshape(-1, 1)
        y = group[y_var].values

        """
        This commented block is for when we have mismatched years for different
        y variables. 

        """
        mask = ~np.isnan(y)
        X_masked = X[mask]
        y_masked = y[mask]
        if len(y_masked) == 0:
            continue  # nothing to fit
        # Fit your model
        model.fit(X_masked, y_masked)
        yhat_masked = model.predict(X_masked)
        # Prepare full-size residuals array
        residuals = np.full_like(y, np.nan, dtype=float)
        residuals[mask] = y_masked - yhat_masked
        # Assign back aligned with group.index
        annual_weather_detrend.loc[group.index, f"{y_var}_detrendLinReg"] = residuals

        # Fit linear regression Commented out on Aug. 14. 2025 and replaced it with what is above.
        # model = LinearRegression()
        # model.fit(X, y)
        # yhat = model.predict(X)
        # annual_weather_detrend.loc[group.index, f"{y_var}_detrendLinReg"] = y - yhat
        # Optionally store slope/intercept
        regression_df.loc[fid, f"{y_var}_linReg_slope"] = model.coef_[0]
        regression_df.loc[fid, f"{y_var}_linReg_intercept"] = model.intercept_

regression_df.reset_index(drop=False, inplace=True)
regression_df.head(2)

##############################################################################################
sens_pred_cols = [x for x in annual_weather_detrend.columns if "Pred" in x]
annual_weather_detrend.drop(columns=sens_pred_cols, inplace=True)

# %%
annual_weather_detrend.drop(columns=["row_number_perfid"], inplace=True)

# %%
sensSlopes_interc_cols = [x for x in annual_weather_detrend.columns if "sens" in x]
sensSlopes_interc_cols

# %%
sensSlopes_interc_df = annual_weather_detrend[["fid"] + sensSlopes_interc_cols].copy()
print(sensSlopes_interc_df.shape)
sensSlopes_interc_df.drop_duplicates(inplace=True)
print(sensSlopes_interc_df.shape)
sensSlopes_interc_df.head(2)

# %%
regression_df.head(2)

# %%
annual_weather_detrend.drop(columns=sensSlopes_interc_cols, inplace=True)
print(annual_weather_detrend.shape)
annual_weather_detrend.head(2)


# regressions data
regression_df = pd.merge(regression_df, sensSlopes_interc_df, how="left", on="fid")


# out_name = bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.csv"
out_name = bio_reOrganized + "bpszone_annualWeatherDroughtByHN_and_deTrended.csv"
annual_weather_detrend.to_csv(out_name, index=False)

# filename = bio_reOrganized + "bpszone_annualWeatherByHN_and_deTrended.sav"
filename = bio_reOrganized + "bpszone_annualWeatherDroughtByHN_and_deTrended.sav"
print("line 456")
print(filename)
export_ = {
    "bpszone_annual_weather_byHN": annual_weather_detrend,
    "slopes_interceps": regression_df,
    "source_code": "00_weatherMonthly2Annual_and_40yearsMK_and_detrend_Drought",
    "Author": "HN",
    "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
print("line 465")
pickle.dump(export_, open(filename, "wb"))
end_time = time.time()
print("it took {:.0f} minutes to run this code.".format((end_time - start_time) / 60))
