# BIOM 200: Microbiome analysis with QIIME 2

Two notebooks for analyzing 16S rRNA amplicon data from Qiita:

| Notebook | Data |
|---|---|
| `biom200_microbiome.ipynb` | Tutorial: composition, rarefaction, alpha and beta diversity, PCoA and PERMANOVA, differential abundance, using the bundled Moving Pictures example (`data/mp/`) |
| `class_nasal_microbiome.ipynb` | The class's own nasal swabs (`data/class/`): left vs. right nostril, self vs. others, students vs. instructors |

Both use the functions in `scripts/biom200.py`.
## Start

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/lucaspatel/biom200?quickstart=1)

1. Click the button above (sign in to GitHub if asked). The first start takes a few minutes.
2. `biom200_microbiome.ipynb` opens automatically; `class_nasal_microbiome.ipynb` is next to it. When asked for a kernel, choose **QIIME 2 2026.7**.
3. Run the notebook from the top. The example and class data are already in `data/`; to analyze other data, download it from [Qiita](https://qiita.ucsd.edu) and drag it onto the `data` folder.

When you finish, stop the codespace (**Codespaces** menu → **Stop Current Codespace**) to save your free monthly hours. Your files are kept until you delete the codespace.

## Run locally instead

Install QIIME 2 2026.7 following the [installation guide](https://library.qiime2.org/quickstart/amplicon), then:

```bash
conda activate rachis-qiime2-2026.7
conda install -c conda-forge notebook   # QIIME 2 2026.7 no longer bundles Jupyter
jupyter notebook biom200_microbiome.ipynb
```

## Data

Class data are never committed to this repository; `data/` is ignored by git except for its README.
