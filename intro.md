# CRESCENT earthquake catalog

## Catalog overview

This jupyterbook describes the creation of the CRESCENT earthquake catalog.

```{mermaid}
flowchart TB
  A[(Seismic data)] --> B
  B(ML Phase detection) --> C
  C[(Picks)] --> D
  D(Association) --> E
  E(EQ localization NLL) --> F[(Catalog)]
  G[(Velocity model)] --> E
  H[(Selected stations)] --> E
  click A "https://pnsn.org/"
  click B "./notebooks/phase_detection.html"
  click G "./notebooks/velocity_model.html"
  click H "./notebooks/choosing_stations.html"
  click E "./notebooks/absolute_earthquake_locations.html"
```

```{tableofcontents}
```
