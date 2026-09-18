#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="flowpy"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

PYTHON=python

M_PP=1.0

TYPES=(
    "bare"
    "root"
    "micp"
    "root_micp"
)


echo "=================================================="
echo " GLARUS LANDSLIDE RISK + MICP PIPELINE"
echo "m_pp = ${M_PP}"
echo "=================================================="

for TYPE in "${TYPES[@]}"; do
    FLOWPY_NAME="${TYPE}_${M_PP}"
    MAP_NAME="m_pp_${M_PP}/fos_${TYPE}_${M_PP}.tif"

    FLOWPY_DIR="output/flowpy/glarus/${FLOWPY_NAME}"
    FLOWPY_INPUT="${FLOWPY_DIR}/flowpy_input/baseline"
    FLOWPY_MARKER="${FLOWPY_DIR}/.flowpy_complete"

    echo "=================================================="
    echo " SCENARIO: ${FLOWPY_NAME}"
    echo "=================================================="

    echo "[1] Exporting FoS / release raster..."
    $PYTHON pipeline/exposure_estimation/01_export_release_raster.py \
        --fos_map "$MAP_NAME" \
        --flowpy_name "$FLOWPY_NAME"

    if [[ -f "$FLOWPY_MARKER" ]]; then
        echo "[2] Flow-Py already completed -> skipping"
    else
        echo "[2] Running Flow-Py..."

        $PYTHON FlowPy/main.py \
            25 \
            75 \
            "$FLOWPY_DIR" \
            "${FLOWPY_INPUT}/dem.tif" \
            "${FLOWPY_INPUT}/release.tif" \
            flux=0.0003 \
            max_z=22
        
        RES_DIR=$(find "$(dirname "$FLOWPY_DIR")" -maxdepth 1 -type d -name "$(basename "$FLOWPY_DIR")res_*" | sort | tail -n 1)
        if [[ -z "$RES_DIR" ]]; then
            echo "ERROR: Flow-Py completed but no res_* directory was found"
            exit 1
        fi

        echo "Moving Flow-Py output into ${FLOWPY_DIR}/"
        cp -a "${RES_DIR}/." "${FLOWPY_DIR}/"
        rm -rf "$RES_DIR"

        touch "$FLOWPY_MARKER"
        echo "Flow-Py completed -> ${FLOWPY_MARKER}"
    fi

    echo "[3] Running Flow-Py exposure..."
    $PYTHON pipeline/exposure_estimation/02_flowpy_exposure.py \
        --flowpy_name "$FLOWPY_NAME"

    echo "[4] Visualizing exposure..."
    $PYTHON pipeline/exposure_estimation/03_exposure_visuals.py \
        --flowpy_name "$FLOWPY_NAME"

    echo "[5] Validating against WSL..."
    $PYTHON pipeline/exposure_estimation/04_validate_against_wsl.py \
        --flowpy_name "$FLOWPY_NAME"

    echo "[6] Running Schwanden case study..."
    $PYTHON pipeline/exposure_estimation/05_schwanden_case.py \
        --flowpy_name "$FLOWPY_NAME"

done

echo "=================================================="
echo " COMPUTING EAD FOR ALL TREATMENTS"
echo "=================================================="

$PYTHON pipeline/exposure_estimation/08_compute_ead.py \
    --bare_name "bare_${M_PP}" \
    --root_name "root_${M_PP}" \
    --micp_name "micp_${M_PP}" \
    --root_micp_name "root_micp_${M_PP}"


echo "=================================================="
echo " COMPUTING Risk Ranking"
echo "=================================================="

for TYPE in "${TYPES[@]}"; do
    FLOWPY_NAME="${TYPE}_${M_PP}"

    $PYTHON pipeline/exposure_estimation/09_risk_ranking.py \
        --flowpy_name "$FLOWPY_NAME"

done


echo "=================================================="
echo " PIPELINE COMPLETE"
echo " m_pp = ${M_PP}"
echo "=================================================="