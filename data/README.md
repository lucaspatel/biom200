# Data

## `mp/`: Moving Pictures of the Human Microbiome (Qiita study 550)

Included so the notebook runs end to end before you have your own data.

| File | Contents |
|---|---|
| `mp/550_feature-table.biom` | Deblur feature table from Qiita (16S V4, 150 nt ASVs; 1,966 samples) |
| `mp/550_sample_information.tsv` | Sample information from Qiita |
| `mp/550_palms_sample_information.tsv` | Left and right palm samples only; the stand-in "own samples" for the last section |
| `mp/550_taxonomy.tsv` | Greengenes2 2024.09 taxonomy for these ASVs |
| `mp/550_tree.nwk` | Greengenes2 2024.09 phylogeny pruned to these ASVs |

Taxonomy and tree were cut from the full Greengenes2 release with `tools/prune_gg2.py`.
Reference: Caporaso et al. 2011, *Genome Biol*, https://doi.org/10.1186/gb-2011-12-5-r50

## `class/`: the class's own samples (Qiita study 16596)

Not committed to the repository. Download from Qiita and upload into `data/class/`:

| File | Contents |
|---|---|
| `16596_feature-table.biom` | Deblur feature table (16S V4, 150 nt ASVs), swabs and blanks |
| `16596_sample_information.tsv` | Sample information |
| `16596_taxonomy.tsv` | Greengenes2 taxonomy for these ASVs |
| `16596_tree.nwk` | Greengenes2 phylogeny pruned to these ASVs |

**In a Codespace:** drag the files from your computer onto the `data/class` folder in the file explorer on the left, or right-click `data/class` → **Upload...**
