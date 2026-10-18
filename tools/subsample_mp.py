"""Make the balanced Moving Pictures subset used by the tutorial notebook.

The full study has 1,966 samples from only two people, so every test is heavily pseudoreplicated and the plots are
crowded. This draws 25 samples per person per body site (200 in total) from samples with at least 300 reads that are
not flagged as mislabeled. The seed below fixes which samples are drawn; nothing else is seeded.

    python tools/subsample_mp.py
"""
import biom
import pandas as pd

SEED = 2026
PER_GROUP, MIN_READS = 25, 300
SRC, OUT = "data/mp/550", "data/mp/550_subset"

table = biom.load_table(f"{SRC}_feature-table.biom")
meta = pd.read_csv(f"{SRC}_sample_information.tsv", sep="\t", dtype=str, keep_default_na=False)
reads = pd.Series(table.sum("sample"), index=table.ids())
meta["reads"] = meta["sample_name"].map(reads)

eligible = meta[(meta["mislabeled"] != "TRUE") & (meta["reads"] >= MIN_READS)]
subset = (eligible.groupby(["host_subject_id", "common_sample_site"], group_keys=False)
          .apply(lambda g: g.sample(n=min(PER_GROUP, len(g)), random_state=SEED)))
keep = set(subset["sample_name"])

sub_table = table.filter(keep, axis="sample", inplace=False).remove_empty(axis="observation", inplace=False)
with biom.util.biom_open(f"{OUT}_feature-table.biom", "w") as fh:
    sub_table.to_hdf5(fh, "Moving Pictures subset")
subset.drop(columns="reads").to_csv(f"{OUT}_sample_information.tsv", sep="\t", index=False)
print(f"{len(keep)} samples, {sub_table.shape[0]} features")
print(subset.groupby(["host_subject_id", "common_sample_site"]).size().unstack())
