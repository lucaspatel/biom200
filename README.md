# BIOM 200: Microbiome analysis

Repository for BIOM 200 (2026). Notebook `01_introduction.ipynb` covers an introduction to microbiome analysis, using the bundled Moving Pictures example. Notebook `02_nasal-analyses.ipynb` explores the nasal microbiomes of class participants (Study 1), and `03_eating-analyses.ipynb` tests the class's Study 2 hypotheses on eating, pets, and the oral and nasal microbiomes. All three use the functions in `scripts/biom200.py`.

## Start

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/lucaspatel/biom200?quickstart=1)

1. Click the button above (sign in to GitHub if asked). The first start takes a few minutes.
2. `01_introduction.ipynb` opens automatically; `02_nasal-analyses.ipynb` and `03_eating-analyses.ipynb` are next to it. When asked for a kernel, choose **QIIME 2 2026.7**.
3. Run the notebook from the top. The example and class data are already in `data/`; to analyze other data, download it from [Qiita](https://qiita.ucsd.edu) and drag it onto the `data` folder.

When you finish, stop the codespace (**Codespaces** menu → **Stop Current Codespace**) to save your free monthly hours. Your files are kept until you delete the codespace.

## Run locally instead

Install QIIME 2 2026.7 following the [installation guide](https://library.qiime2.org/quickstart/amplicon), then:

```bash
conda activate rachis-qiime2-2026.7
conda install -c conda-forge notebook   # QIIME 2 2026.7 no longer bundles Jupyter
jupyter notebook 01_introduction.ipynb
```

## Data

Class data are never committed to this repository; `data/` is ignored by git except for its README.
