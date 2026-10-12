"""Prune the Greengenes2 phylogeny and taxonomy to the features in a feature table.

The full Greengenes2 tree has ~23 million tips and needs ~9 GB of memory to load, too much for a
Codespace. Run this once per feature table on a machine with enough memory, inside the QIIME 2
environment, and commit the small outputs next to the table.

    python tools/prune_gg2.py --table example/550_table.biom \
        --tree 2024.09.phylogeny.asv.nwk.qza --taxonomy 2024.09.taxonomy.asv.tsv.qza \
        --out-prefix example/550

Writes <out-prefix>_tree.nwk and <out-prefix>_taxonomy.tsv. Greengenes2 files:
https://ftp.microbio.me/greengenes_release/current/
"""
import argparse
import gzip
import io
import time
import zipfile

import biom
import bp
import pandas as pd


def feature_ids(path):
    if path.endswith(".qza"):
        from qiime2 import Artifact
        return set(Artifact.load(path).view(pd.DataFrame).columns)
    return set(biom.load_table(path).ids(axis="observation"))


def open_member(path, suffix):
    # Text stream for a plain/gzipped file, or the matching file inside a .qza archive.
    if path.endswith(".qza"):
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.endswith(suffix))
        return io.TextIOWrapper(z.open(name), encoding="utf-8")
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="feature table (.biom or .qza)")
    ap.add_argument("--tree", required=True, help="Greengenes2 phylogeny (.nwk.qza or .nwk)")
    ap.add_argument("--taxonomy", required=True, help="Greengenes2 taxonomy (.tsv.qza, .tsv.gz, or .tsv)")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    t0 = time.time()

    ids = feature_ids(a.table)
    print(f"{len(ids)} features in {a.table}")

    with open_member(a.taxonomy, ".tsv") as fh:
        hits = []
        for chunk in pd.read_csv(fh, sep="\t", chunksize=1_000_000, dtype=str, comment=None):
            hits.append(chunk[chunk.iloc[:, 0].isin(ids)])
    tax = pd.concat(hits)
    tax.columns = ["Feature ID", "Taxon"] + list(tax.columns[2:])
    tax.to_csv(f"{a.out_prefix}_taxonomy.tsv", sep="\t", index=False)
    print(f"taxonomy: {len(tax)} of {len(ids)} features found ({time.time() - t0:.0f}s)")

    with open_member(a.tree, ".nwk") as fh:
        tree = bp.parse_newick(fh.read())
    print(f"tree: {tree.ntips():,} tips loaded ({time.time() - t0:.0f}s)")
    tips = {tree.name(i) for i in range(tree.B.size) if tree.B[i] and tree.isleaf(i)}
    keep = ids & tips
    print(f"tree: {len(keep)} of {len(ids)} features are tips")
    pruned = tree.shear(keep).collapse()
    with open(f"{a.out_prefix}_tree.nwk", "w") as fh:
        fh.write(bp.to_skbio_treenode(pruned).__str__())
    missing = len(ids) - len(keep)
    print(f"wrote {a.out_prefix}_tree.nwk and {a.out_prefix}_taxonomy.tsv ({time.time() - t0:.0f}s)")
    if missing:
        print(f"note: {missing} features are not in Greengenes2; they are dropped from UniFrac and Faith's PD only "
              "if the table is filtered to the tree first")


if __name__ == "__main__":
    main()
