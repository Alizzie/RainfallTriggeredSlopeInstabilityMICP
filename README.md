# Rainfall-Triggered Slope Instability & MICP

Code for the dry-lab models of the UZurich iGEM 2026 project. Methods, parameters and results are on the wiki:

- [Slope Stability Model](https://2026.igem.wiki/uzurich/stability-model)
- [MICP Feasibility Model](https://2026.igem.wiki/uzurich/injection-model)
- [Economic Model](https://2026.igem.wiki/uzurich/economic-model)

## Structure

| Model | Code |
|---|---|
| **Slope Stability Model** | `core/`, `pipeline/temporal/`, `pipeline/spatial-temporal/`, `pipeline/impacts/`, `pipeline/utils/`, `data_analysis/`, `diagnostics/`, `bondo_case_study/` |
| **MICP Feasibility Model** | `data_analysis/utility_feasibility_map/` (utility scripts only) |
| **Economic Model** | `pipeline/exposure_estimation/` (run with `automate.sh`) |

`archive/` holds superseded scripts.

## Setup

- Install the Python packages:

  ```bash
  pip install -r requirements.txt
  ```

- Input data is not included because of its size. See [Data](#data) below.
- The Economic Model also needs [Flow-Py](https://github.com/avaframe/FlowPy). Clone it into the repo root as `FlowPy/`, since `automate.sh` calls `FlowPy/main.py`. The script activates a conda environment named `flowpy`.

Run all scripts from the repo root. Results are written to `output/`, which is created on the first run and is not in the repo.

## Data

Create a `data/` folder at the repo root. The input data is not included because of its size. The paths the scripts expect are defined in `core/data_loader.py` and `pipeline/utils/constants.py`. If you have trouble getting a dataset, request it from the authors.

All downloaded maps must be rasters on the same 25 m grid (for example rasterize vectors in QGIS). In binary maps, the true value is always `1`. Use these folders and file names:

| Folder | File(s) | Source | How to prepare |
|---|---|---|---|
| `data/rhiresD/` | RhiresD yearly files | [MeteoSwiss RhiresD archive](https://data.geo.admin.ch/browser/index.html#/collections/ch.meteoschweiz.ogd-surface-derived-grid/items/archive-ch) | Download all yearly NetCDF files from 1991 to 2025 and keep the original file names. |
| `data/wsl_inventory/` | `wsl_landslide.csv`, `wsl_usable_events.csv`, `hangmuren_storme.csv` | [WSL Flood and Landslide Damage Database](https://www.wsl.ch/en/natural-hazards/understanding-and-forecasting-floods/flood-and-landslide-damage-database/); StorMe: [hangmuren.wsl.ch](https://hangmuren.wsl.ch/) | Request permission from WSL, then place the exports here. |
| `data/soil_moisture_history/` | `weekly_historic_regions.csv` | [BAFU historic drought soil moisture (nFK)](https://data.geo.admin.ch/browser/#/collections/ch.bafu.trockenheitsdaten-numerisch/items/trockenheitsdaten-numerisch_historic?.asset=asset-trockenheitsdaten-numerisch_historic-csv-zip) | Download and unzip the CSV. |
| `data/trockenheitsdaten-numerisch_reference/` | reference files | [BAFU Trockenheitsdaten, numerisch (reference)](https://data.geo.admin.ch/browser/#/collections/ch.bafu.trockenheitsdaten-numerisch/items/trockenheitsdaten-numerisch_reference) | Supplementary context about the drought moisture data. |
| `data/` (no subfolder) | `trockenheitsindex_2_2056.json` | [BAFU drought index JSON](https://data.geo.admin.ch/browser/index.html#/collections/ch.bafu.trockenheitsindex/items/trockenheitsindex_2?.asset=asset-trockenheitsindex_2_2056-json) | Download the JSON, which summarizes the weekly data. `core/data_loader.py` extracts the geometry of each drought region from it automatically. |
| `data/hangmuren_silverprotect/` | `hangmuren_processed.tif` | [BAFU SilvaProtect Hangmuren](https://www.bafu.admin.ch/de/silvaprotect) | Convert to a binary map, with `1` marking the unstable areas. |
| `data/injection/` | `binary_injection.tif`, `injection_map.tif` | Output of the MICP Feasibility Model ([wiki page](https://2026.igem.wiki/uzurich/injection-model)) | Place the binary maps from the feasibility model. The steps are on the wiki page. |
| `data/swissalti_slope/` | `dem_25m_ch.tif`, `slope_deg_25m_ch.tif` | [swisstopo SwissALTI^Regio](https://www.swisstopo.admin.ch/en/height-model-swissaltiregio) | Rasterize to the 25 m grid. Derive the slope with the QGIS slope analysis function. |
| `data/swissboundaries/` | `canton_id_25m.tif`, `canton_names.csv` | [swisstopo swissBOUNDARIES3D](https://www.swisstopo.admin.ch/en/landscape-model-swissboundaries3d) | Rasterize with one ID per canton. Build the ID-to-name lookup `canton_names.csv` from the downloaded data. |
| `data/swissTLM3D/` | `tlm_forest_25m_ch.tif`, `tlm_nonsoil_25m_ch.tif` | [swisstopo SwissTLM3D](https://www.swisstopo.admin.ch/de/landschaftsmodell-swisstlm3d) | Rasterize to 25 m using the forest, non-soil and soil categories defined on the [wiki](https://2026.igem.wiki/uzurich/stability-model). |
| `data/infra/` | `buildings_id_25m.tif` | [swisstopo SwissTLM3D](https://www.swisstopo.admin.ch/de/landschaftsmodell-swisstlm3d) (building footprints) | Each building gets an ID. Buildings falling in the same 25 m cell are aggregated into one. |
| | `nutzungsgebiete.tif` | [Landwirtschaftliche Nutzungsfläche Schweiz (opendata.swiss)](https://opendata.swiss/de/dataset/landwirtschaftliche-nutzungsflachen-schweiz) | Rasterized agricultural land. |
| | `protected_area_25m.tif` | [swisstopo TLMRegio](https://www.swisstopo.admin.ch/de/landschaftsmodell-swisstlmregio) | Binary map of the protected areas. |
| | `railway_binary_25m.tif` | [swisstopo SwissTLM3D](https://www.swisstopo.admin.ch/de/landschaftsmodell-swisstlm3d) | Binary map of existing railways, interpolated to the 25 m grid. |
| | `roads_binary_25m.tif` | [swisstopo official street directory](https://www.swisstopo.admin.ch/de/amtliches-verzeichnis-der-strassen) | Binary map of existing roads. |
| | `schutzwald_rutschungen_25m.tif` | [Canton Glarus Schutzwald layer](https://map.geo.gl.ch/) | Binary map of the protection forest (Schutzwald) that protects against landslides. Keep only features whose `prozesse` field contains "Rutschungen". |

## Reproducing the results

### Slope Stability Model

Run the scripts in numeric order, folder by folder, following the order of the wiki sections:

1. `pipeline/temporal/`
2. `pipeline/spatial-temporal/`
3. `pipeline/impacts/` (application analysis, built on the outputs above)

Later steps read outputs of earlier ones. For example, the calibration in `pipeline/temporal/02_calibrate_model.py` writes `output/temporal/02_calibration/calibration_results.csv`, which later scripts load.

### Understanding the data

`data_analysis/` contains scripts to explore the inputs and the model behaviour. They are not required to reproduce the results. They cover:

- **FoS theory:** how the Factor of Safety depends on slope, cohesion and saturation (`01_fos_saturation_theory.py`, `slope_angle_min_max.py`).
- **Soil moisture:** BAFU field capacity and the bucket-model moisture driven by rainfall, aggregated by year and month (`02_`, `03_`, `10_`, `yearly_soil_moisture_fos_analysis.py`).
- **Rainfall and slopes:** rainfall coverage of the WSL events, slope and rainfall statistics, and differences between slope maps from interpolation (`04_`, `05_`, `06_`).
- **WSL inventory and soils:** slope distribution of the events, their spread over regions, and StorMe soil types per drought region (`07_`, `08_`, `09_`).

### MICP Feasibility Model

The scripts in `data_analysis/utility_feasibility_map/` are helpers for working with the feasibility and hazard maps. They compare rasters and compute class shares.

### Economic Model

```bash
bash pipeline/exposure_estimation/automate.sh
```

This runs the chain for the bare, root, MICP and root + MICP scenarios. It needs the unstable-cell FoS maps from the Slope Stability Model (`pipeline/impacts/01_fos_maps.py`).