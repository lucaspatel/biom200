# BIOM 200: Microbiome analysis with QIIME 2

A notebook for analyzing the class 16S rRNA amplicon data from Qiita: taxonomic composition, rarefaction, alpha and beta diversity, PCoA and PERMANOVA, and differential abundance.

## Start

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/lucaspatel/biom200?quickstart=1)

1. Click the button above (sign in to GitHub if asked). The first start takes a few minutes.
2. `biom200_microbiome.ipynb` opens automatically. When asked for a kernel, choose **QIIME 2 2026.7**.
3. Download your feature table, sample information, and taxonomy from [Qiita](https://qiita.ucsd.edu), then drag them onto the `data` folder in the file explorer.
4. Run the notebook from the top.

When you finish, stop the codespace (**Codespaces** menu → **Stop Current Codespace**) to save your free monthly hours. Your files are kept until you delete the codespace.

## Run locally instead

Install QIIME 2 2026.7 following the [installation guide](https://library.qiime2.org/quickstart/amplicon), then:

```bash
conda activate rachis-qiime2-2026.7
conda install -c conda-forge jupyterlab
jupyter lab biom200_microbiome.ipynb
```

## Data

Class data are never committed to this repository; `data/` is ignored by git except for its README.
