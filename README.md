# CRESCENT Earthquake Catalog Workflow Book

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

Notebooks and documentation for the workflow used to build the CRESCENT earthquake catalog, from station selection and velocity model gridding to machine-learning phase picking and earthquake location.

**Read the book:** https://cascadiaquakes.github.io/earthquake_catalog_book/

**Catalog viewer:** https://eqcat.cascadiaquakes.org

---

## Contents

- **Earthquakes in Cascadia**: cataloged seismicity in the region
- **Velocity Model**: preparing and gridding the velocity model for NonLinLoc
- **Stations**: choosing stations
- **Augmenting with Machine Learning**: phase detection and combining picks
- **Earthquake Location**: absolute earthquake locations

---

## Building Locally

The book is built with [Jupyter Book](https://jupyterbook.org) (v1). From the repository root:

```bash
pip install -r requirements.txt
jupyter-book build .
```

Then open `_build/html/index.html`.

---

## Deployment

GitHub Actions builds the book on every push to `main` and publishes it to GitHub Pages.

---

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).

---

## Contacts

- **Amanda M. Thomas**, University of California, Davis: [amthom@ucdavis.edu](mailto:amthom@ucdavis.edu)
- **William Marfo**, University of California, Davis: [wmarfo@ucdavis.edu](mailto:wmarfo@ucdavis.edu)
- **Loïc Bachelot**, University of Oregon: [lbachelo@uoregon.edu](mailto:lbachelo@uoregon.edu)

For corrections or suggestions, please [open an issue](https://github.com/cascadiaquakes/earthquake_catalog_book/issues).

---

## Acknowledgments

<table>
  <tr>
    <td align="center" valign="middle">
      <img src="https://raw.githubusercontent.com/cascadiaquakes/.github/main/assets/crescent-circle.png" alt="CRESCENT Logo" width="150" />
    </td>
    <td align="center" valign="middle">
      <img src="https://raw.githubusercontent.com/cascadiaquakes/.github/main/assets/nsf-logo.png" alt="NSF Logo" width="450" />
    </td>
  </tr>
</table>

This project is supported by the National Science Foundation through the Cascadia Region Earthquake Science Center (CRESCENT), Cooperative Agreement EAR-2225286.
