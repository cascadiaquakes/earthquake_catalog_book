# Brenton Workflow Handoff

The earthquake catalog workflow developed by Brenton Hirao during his
postdoc. The pipeline builds a merged, deduplicated regional earthquake
catalog from raw phase picks, a Vs velocity model, and station
metadata. This page captures the pipeline in execution order so the
team can pick up the work after Brenton's departure.

## Step-by-step workflow

The pipeline runs in five stages. Each stage below lists the driver
script, its purpose, and any per-stage notes.

### 1. Experimental setup — geographic grid boundaries

**Script:** `Set_grid_boundaries.py`

Defines the six regional bounding boxes that every downstream stage is
chunked over. Region extents and inter-region overlap live here.

### 2. Process the velocity model, topography, and stations

Four scripts run in sequence:

**a. Trim velocity grids** — `trim_vs_grids_V2_aeqd_topocorr.py`  
Trims the source Vs grids to each region, applies the AEQD projection,
and corrects for topography.

**b. Travel-time grids** — `Make_tt_grids_v2.py`  
Computes per-station travel-time grids from the trimmed velocity model.

**c. Prepare topography** — `Prep_topo.py`  
Prepares the topographic datum used by the location step.

**d. Regional Vp/Vs ratios** — `Make_regional_wadati.py`  
Derives regional Vp/Vs ratios via Wadati analysis.

### 3. Association

**Script:** `Association_v1_global.py`

Associates individual phase picks into candidate events. The output
feeds directly into the NonLinLoc stage below.

### 4. NonLinLoc — initial locations

Two scripts:

**a. Prepare NonLinLoc runfiles and obs files** — `Make_nlloc_obs_runfiles.py`  
Consumes the association output and emits NonLinLoc obs files and
per-region runfiles.

**b. Run NonLinLoc on a cluster** — `Nll_driver_v2.py`  
Driver that runs NonLinLoc across regions on the cluster. Produces
per-event hypocenters with associated uncertainty ellipsoids.

### 5. Post-processing

**Script:** `Postprocessing_combined.py`

Runs three sub-steps end-to-end:

- **a.** Adjust the topographic datum.
- **b.** Merge the six regional catalogs into a single catalog.
- **c.** Remove duplicate events in the overlapping (intersecting)
  regions.

Output: the final merged, deduplicated earthquake catalog.

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
