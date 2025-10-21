# Clinical Patient Data

## Source

This clinical data originates from:

**Repository:** https://github.com/DGermano8/JSFGermano2024/tree/main/TEIVR_Results/particle-filter-example-tiv_covid

**Original Location:** `TEIVR_Results/particle-filter-example-tiv_covid/data/`

## Contents

This directory contains RT-PCR viral load time series data for COVID-19 patients:

- `432192.ssv`
- `443108.ssv`
- `444332.ssv`
- `444391.ssv`
- `445602.ssv`
- `451152.ssv`

Each file contains time series measurements of viral load for a single patient.

## Data Format

Files are space-separated values (`.ssv`) with viral load measurements over time.

## Usage

These files are used for clinical inference in the TEIRV NPE workflow. The inference step loads these patient time series and generates posterior distributions over TEIRV model parameters for each patient.

## Date Added

October 17, 2025
