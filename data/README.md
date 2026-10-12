# Data

## Example: Moving Pictures of the Human Microbiome (Qiita study 550)

Included so the notebook runs end to end before you have your own data.

| File | Contents |
|---|---|
| `550_feature-table.biom` | Deblur feature table from Qiita (16S V4, 150 nt ASVs; 1,966 samples) |
| `550_sample_information.tsv` | Sample information from Qiita |
| `550_palms_sample_information.tsv` | Left and right palm samples only; the stand-in "own samples" for the last section |
| `550_taxonomy.tsv` | Greengenes2 2024.09 taxonomy for these ASVs |
| `550_tree.nwk` | Greengenes2 2024.09 phylogeny pruned to these ASVs |

Taxonomy and tree were cut from the full Greengenes2 release with `tools/prune_gg2.py`.
Reference: Caporaso et al. 2011, *Genome Biol*, https://doi.org/10.1186/gb-2011-12-5-r50

## Your own data

Download from Qiita and upload into this folder:

- the feature table (`.biom` or `.qza`)
- the sample information file (`.tsv`)

**In a Codespace:** drag the files from your computer onto this `data` folder in the file explorer on the left, or right-click `data` → **Upload...**

Files you add here are never committed to the repository.
