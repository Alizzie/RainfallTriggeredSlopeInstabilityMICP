"""
Calculate the soil moisture leading up to the Bondo landslide event in 2017.
This script loads the BAFU soil moisture data, filters for the relevant region and time period, and plots the soil moisture leading up to the event.
It also prints a snapshot of the soil moisture values in the days immediately before the landslide.
"""

import pandas as pd
import matplotlib.pyplot as plt

# =====================================================================
# 1. LOAD THE BAFU HISTORIC CSV DATA
# =====================================================================
# Update this to the exact name of your downloaded BAFU CSV file
file_path = "data/soil_moisture_history/weekly_historic_regions.csv"

print("Loading BAFU Soil Moisture Data...")
# The Swiss CSV uses semicolons ';' instead of commas
df = pd.read_csv(
    file_path,
    sep=";",
    skiprows=3,
    parse_dates=["measured_at"],
    dayfirst=True,
)

# =====================================================================
# 2. FILTER FOR THE BONDO REGION
# =====================================================================
# The BAFU data is divided into 'drought_region_id's.
# We need to see which regions exist to pick the one for Graubünden/Bondo.
print(f"Available Region IDs in the dataset: {df['drought_region_id'].unique()}")

# FOR NOW: Let's assume Region 1 (You will need to check the BAFU map to confirm
# which ID corresponds to the Southern Graubünden / Bergell valley).
REGION_ID = 66
bondo_df = df[df["drought_region_id"] == REGION_ID].copy()

# =====================================================================
# 3. FILTER FOR SUMMER 2017 (The Bondo Event)
# =====================================================================
# The landslide happened on August 23, 2017. Let's look at July through September.
start_date = pd.Timestamp("2017-07-01")
end_date = pd.Timestamp("2017-09-30")

mask = (bondo_df["measured_at"] >= start_date) & (bondo_df["measured_at"] <= end_date)

bondo_2017 = bondo_df.loc[mask]

# =====================================================================
# 4. PLOT THE REAL SOIL MOISTURE
# =====================================================================
# We need to look at 'soil_moisture_ufc' to see what scale the government uses
# (e.g., is it a percentage 0-100? A decimal 0.0-1.0? Or absolute millimeters?)

plt.figure(figsize=(10, 5))
plt.plot(
    bondo_2017["measured_at"],
    bondo_2017["soil_moisture_ufc"],
    color="black",
    linewidth=2,
    label="BAFU Soil Moisture (UFC)",
)

# Draw a red line on the exact day of the Bondo landslide
plt.axvline(
    pd.to_datetime("2017-08-23"),
    color="red",
    linestyle="--",
    label="Bondo Landslide Triggered",
)

plt.title(f"Measured Soil Moisture Leading Up to Bondo Event (Region {REGION_ID})")
plt.ylabel("Soil Moisture UFC")
plt.xlabel("Date")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print a snapshot of the data just before the landslide
print("\nData snapshot for the days before the landslide:")
print(
    bondo_2017[["measured_at", "soil_moisture_ufc"]].loc[
        (bondo_2017["measured_at"] >= "2017-08-18")
        & (bondo_2017["measured_at"] <= "2017-08-23")
    ]
)
