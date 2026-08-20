# Brenton Workflow Handoff

The enhanced earthquake catalog workflow developed by Brenton Hirao. The
pipeline builds a merged, deduplicated regional catalog from ML-derived
phase picks (PNW-EQTransformer in the PNSN region; PhaseNet picks from
the Ni et al., 2025 global database in Northern California and Northwest
Nevada), the CRESCENT Community Velocity Model (CVM; He et al., 2026),
and permanent-network station metadata across a 2002–2024 study window.
This page captures the pipeline in execution order so the team can pick
up the work after Brenton's departure.

The driver scripts referenced below live in
[`scripts/brenton/`](https://github.com/cascadiaquakes/earthquake_catalog_book/tree/main/scripts/brenton).

:::{admonition} Scientific context
:class: note
The scientific rationale, equations, and results that motivate each
stage below are documented in Chapter 4 of Brenton Hirao's PhD thesis,
*Enhanced Multi-decade Long Earthquake Catalogs at Multiple Scales:
Insights into Volcanic and Tectonic Environments in the Pacific
Northwest* (University of Oregon, 2026), which is also being prepared
as a standalone journal paper. Section references in each stage
(e.g. *Methods §2.3*) point to Chapter 4 of that thesis.
:::

## Step-by-step workflow

The pipeline runs in five stages. Each stage below lists the driver
script, its purpose, and any per-stage notes.

### 1. Experimental setup — subregion grid boundaries

**Script:** `Set_grid_boundaries.py`

Defines the six subregions (W1–W3 along the west, E1–E3 along the east)
that every downstream stage is chunked over. Subregion extents and the
110–130 km inter-subregion overlap live here.
(*Methods §2.2*)

### 2. Process the velocity model, topography, and stations

Four scripts run in sequence (all corresponding to *Methods §2.2*):

**a. Trim velocity grids** — `trim_vs_grids_V2_aeqd_topocorr.py`  
Trims the CVM Vs grids to each subregion, applies the AEQD projection,
and corrects for topography.

**b. Travel-time grids** — `Make_tt_grids_v2.py`  
Computes per-station P-wave travel-time grids from the trimmed velocity
model using the Fast-Marching Method (Pykonal; White et al., 2020).

**c. Prepare topography** — `Prep_topo.py`  
Prepares the topographic datum used by the location step.

**d. Regional Vp/Vs ratios** — `Make_regional_wadati.py`  
Derives subregion-specific Vp/Vs ratios via Wadati (1933) analysis with
stratified sampling over 10×10×2 km³ bins.

### 3. Association

**Script:** `Association_v1_global.py`

Associates individual phase picks into candidate events with the PyOcto
associator (Münchmeyer et al., 2024). Events with at least 3 P- and 3
S-wave picks recorded by at least 6 stations are retained. Output feeds
directly into the NonLinLoc stage below.
(*Methods §2.3*)

### 4. NonLinLoc — initial locations and SSST refinement

Two scripts:

**a. Prepare NonLinLoc runfiles and obs files** — `Make_nlloc_obs_runfiles.py`  
Consumes the association output and emits NonLinLoc obs files and
per-subregion runfiles.

**b. Run NonLinLoc on a cluster** — `Nll_driver_v2.py`  
Driver that runs NonLinLoc (Lomax et al., 2014) across subregions on
the cluster with OCT-TREE nested gridsearch and EDT weighting for
initial locations, followed by the source-specific station correction
term (SSST; Lomax and Savvaidis, 2020) refinement pass. Produces
per-event hypocenters with 3-D uncertainty ellipsoids.
(*Methods §2.3–2.4*)

### 5. Post-processing

**Script:** `Postprocessing_combined.py`

Runs three sub-steps end-to-end (*Methods §2.5*):

- **a.** Adjust the topographic datum.
- **b.** Merge the six subregional catalogs into a single catalog.
- **c.** Remove duplicate events in overlapping subregions using a
  Ball-Tree neighbor search (Scikit-learn), matching pairs within 2 s
  and 50 km that share phase arrivals, retaining the event with the
  smallest azimuthal gap.

Output: the merged, deduplicated earthquake catalog.

:::{admonition} Not yet covered by driver scripts on this page
:class: important
Two paper sections describe post-catalog steps whose driver scripts are
not yet included above and will be added when Brenton hands them over:

- **Magnitude scale derivation** (*Methods §2.6*) — regional
  vertical-component local-magnitude inversion with per-station
  correction terms.
- **Benchmarking / merge with ANSS network catalogs** (*Methods §2.7*) —
  relocation of PNSN/NCEDC/SCEDC/NEIC events with LibComCat and
  reconciliation with the ML catalog.
:::

## Updates — associating and locating new events after 2024

:::{admonition} To document with Brenton
:class: important
Which stages need to be rerun for an incremental update, and which can
reuse cached inputs from the initial catalog build?
:::

## Running the pipeline

Environment, dependencies, and cluster setup will be documented here
once Brenton hands over the codebase and environment specification.

- **Codebase:** _repo / archive location TBD_
- **Python environment:** _requirements.txt or conda env yml TBD_
- **External tools:** NonLinLoc (cluster install), any GMT / topo utilities TBD
- **Cluster:** name, queue, and module load commands TBD
- **Data locations:** phase picks, station metadata, source Vs grids,
  topography rasters — paths TBD

## Contacts

- **Brenton Hirao** — original author of this workflow
- **Amanda M. Thomas** — <amthom@ucdavis.edu>
- **William Marfo** — <wmarfo@ucdavis.edu>
