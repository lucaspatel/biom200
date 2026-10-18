"""Shared analysis and plotting functions for the BIOM 200 notebooks.

Every step takes a *dataset* (returned by `load_dataset`) and stores its results on it, so the same functions
analyse the Moving Pictures example, the class study, or any other Qiita feature table. Plots are interactive
(ipywidgets) and every plot is also saved as a PDF under figures/<dataset name>/.
"""
import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import biom
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skbio
from qiime2 import Artifact, Metadata
from qiime2.plugins import diversity_lib, emperor, feature_table, phylogeny, taxa
from scipy import stats
from skbio.stats.composition import ancombc
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa

warnings.filterwarnings("ignore", category=FutureWarning)
RESULTS, FIGURES = Path("results"), Path("figures")
LEVELS = {"Phylum": 2, "Class": 3, "Order": 4, "Family": 5, "Genus": 6, "Species": 7}

# one style for every figure
sns.set_theme(context="notebook", style="ticks", font_scale=0.95,
              rc={"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                  "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
                  "pdf.fonttype": 42})
SEQ_CMAP = "mako_r"
UP, DOWN, NS = "#4C72B0", "#C44E52", "#BBBBBB"


# ---------------------------------------------------------------------------------------------------- helpers
def save_and_show(fig, ds, name, **params):
    """Save the figure as figures/<dataset>/<name>_<params>.pdf, then display it."""
    slug = "_".join(re.sub(r"[^A-Za-z0-9.-]+", "-", f"{k}-{v}") for k, v in params.items())
    out = FIGURES / ds.name
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}{'_' + slug if slug else ''}.pdf", bbox_inches="tight")
    plt.show()


def colors_for(ds, col):
    """Fixed colour for each level of a metadata column, identical in every plot."""
    levels = sorted(ds.meta[col].astype(str).unique())
    pal = sns.color_palette("deep" if len(levels) <= 10 else "husl", len(levels))
    return dict(zip(levels, pal))


def dropdown(options, value, label):
    return widgets.Dropdown(options=options, value=value if value in options else options[0], description=label,
                            style={"description_width": "initial"})


def slider(lo, hi, value, label, step=1):
    return widgets.IntSlider(min=lo, max=hi, value=value, step=step, description=label, continuous_update=False,
                             style={"description_width": "initial"}, layout={"width": "420px"})


def as_metadata(df):
    df = df.copy()
    df.index.name = "sample-id"
    return Metadata(df)


def short_label(taxon):
    """Last named rank, e.g. 'Streptococcus', or 'Neisseriaceae (f.)' when the genus is unresolved."""
    ranks = [r.strip() for r in str(taxon).split(";")]
    named = [r for r in ranks if len(r) > 3 and not r.endswith("__")]
    if not named:
        return "Unassigned"
    if "__" not in named[-1]:
        return named[-1]
    rank, name = named[-1].split("__", 1)
    name = re.sub(r"_\d+\b", "", name)   # drop Greengenes2 numeric suffixes, e.g. Klebsiella_724518
    if rank == "s" and " " not in name:    # older references give the epithet only, e.g. s__uniformis
        genus = next((re.sub(r"_\d+\b", "", r.split("__", 1)[1]) for r in named if r.startswith("g__")), "")
        return f"{genus} {name}".strip()
    if rank in ("g", "s"):                 # Greengenes2 species names already start with their genus
        return name
    return f"{name} ({rank}.)"


def categorical_columns(meta, group, subject, max_levels=12):
    cols = [c for c in meta.columns if meta[c].dtype == object and 2 <= meta[c].nunique() <= max_levels]
    return sorted(cols, key=lambda c: (c not in (group, subject), c))


# ---------------------------------------------------------------------------------------------------- loading
def load_table(path):
    if str(path).endswith(".qza"):
        return Artifact.load(path), None
    bt = biom.load_table(path)
    tax = bt.metadata_to_dataframe("observation") if bt.metadata(axis="observation") else None
    return Artifact.import_data("FeatureTable[Frequency]", path, view_type="BIOMV210Format"), tax


def load_taxonomy(path, from_table):
    if path is None:
        if from_table is None:
            raise ValueError("No taxonomy file given and the table carries no taxonomy.")
        cols = [c for c in from_table.columns if c.startswith("taxonomy")]
        df = pd.DataFrame({"Taxon": from_table[cols].astype(str).agg("; ".join, axis=1)})
        df.index.name = "Feature ID"
        return Artifact.import_data("FeatureData[Taxonomy]", df)
    if str(path).endswith(".qza"):
        return Artifact.load(path)
    return Artifact.import_data("FeatureData[Taxonomy]", path, view_type="TSVTaxonomyFormat")


def load_tree(path):
    if str(path).endswith(".qza"):
        return Artifact.load(path)
    return Artifact.import_data("Phylogeny[Rooted]", path)   # Newick (.nwk)


def load_dataset(name, table_path, metadata_path, taxonomy_path=None, tree_path=None, *, group, subject,
                 control_col="sample_type", control_values=("control blank", "control mock")):
    """Load a feature table and its metadata, report the controls, and return a dataset without them.

    The controls are kept on the dataset (`ds.controls`, `ds.control_counts`) for `explore_controls`.
    """
    table, tax_from_table = load_table(table_path)
    taxonomy = load_taxonomy(taxonomy_path, tax_from_table)
    meta = Metadata.load(metadata_path).to_dataframe()
    counts = table.view(pd.DataFrame)
    shared = counts.index.intersection(meta.index)
    print(f"{name}: {counts.shape[0]} samples and {counts.shape[1]} features; {len(shared)} samples have metadata")
    missing_samples = meta.index.difference(counts.index)
    if len(missing_samples):
        print(f"  {len(missing_samples)} samples in the metadata have no reads in the table")
    counts, meta = counts.loc[shared], meta.loc[shared]
    depth = counts.sum(axis=1)

    # features without a taxonomy entry (e.g. ASVs not in Greengenes2) are labelled Unassigned
    tax = taxonomy.view(pd.DataFrame)
    missing = counts.columns.difference(tax.index)
    if len(missing):
        print(f"  {len(missing)} features ({counts[missing].values.sum() / counts.values.sum():.1%} of reads) "
              "have no taxonomy and are labelled Unassigned")
        extra = pd.DataFrame({"Taxon": "Unassigned"}, index=pd.Index(missing, name=tax.index.name))
        tax = pd.concat([tax[["Taxon"]], extra])
        tax.index.name = "Feature ID"
        taxonomy = Artifact.import_data("FeatureData[Taxonomy]", tax)
    is_control = meta[control_col].isin(control_values) if control_col in meta else pd.Series(False, index=meta.index)
    if is_control.any():
        print(f"  {is_control.sum()} controls ({', '.join(meta.loc[is_control, control_col].unique())}): "
              f"median {depth[is_control].median():,.0f} reads vs {depth[~is_control].median():,.0f} in samples")

    keep = meta.index[~is_control]
    controls = meta.loc[is_control]
    meta = meta.loc[keep]
    meta = meta.loc[:, meta.notna().any()]
    md = as_metadata(meta)
    filtered = feature_table.methods.filter_samples(table=table, metadata=md).filtered_table
    tree = load_tree(tree_path) if tree_path else None
    categorical = categorical_columns(meta, group, subject)
    print(f"  {len(keep)} biological samples retained; fields available for grouping: {categorical}")
    return SimpleNamespace(name=name, table=filtered, taxonomy=taxonomy, tree=tree, meta=meta, md=md,
                           group=group, subject=subject, categorical=categorical, control_col=control_col,
                           controls=controls, control_counts=counts.loc[controls.index],
                           sample_counts=counts.loc[keep])


def add_metadata_column(ds, name, values):
    """Add or replace a metadata column (a Series indexed by sample, or a function of ds.meta)."""
    values = values(ds.meta) if callable(values) else values
    ds.meta[name] = pd.Series(values, index=ds.meta.index).astype(str)
    ds.md = as_metadata(ds.meta)
    ds.categorical = categorical_columns(ds.meta, ds.group, ds.subject)
    if hasattr(ds, "permanova_cache"):
        ds.permanova_cache = {}
    print(ds.meta[name].value_counts().to_string())


# ---------------------------------------------------------------------------------------------------- controls
def plot_controls(ds, level="Genus", top_n=10):
    """Read depth of controls vs samples, and what the controls contain compared with the samples."""
    depth = pd.concat([ds.sample_counts.sum(axis=1).rename("reads").to_frame().assign(type="sample"),
                       ds.control_counts.sum(axis=1).rename("reads").to_frame().assign(type="control")])
    tax = ds.taxonomy.view(pd.Series)

    def collapse(counts):
        ranks = tax.reindex(counts.columns).fillna("Unassigned").str.split(";").apply(lambda r: [x.strip() for x in r])
        g = counts.T.groupby(ranks.str[:LEVELS[level]].str.join(";").values).sum().T
        g.columns = [short_label(c) for c in g.columns]
        g = g.T.groupby(level=0).sum().T
        return g.div(g.sum(axis=1).replace(0, np.nan), axis=0).mean()

    both = pd.concat({"samples": collapse(ds.sample_counts), "controls": collapse(ds.control_counts)}, axis=1).fillna(0)
    top = both.max(axis=1).sort_values(ascending=False).index[:top_n]
    both = pd.concat([both.loc[top], (1 - both.loc[top].sum()).to_frame("Other").T])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 3]})
    pal = {"sample": "#4C72B0", "control": "#C44E52"}
    sns.stripplot(data=depth, x="type", y="reads", hue="type", palette=pal, size=4, alpha=0.7, jitter=0.2, ax=ax,
                  legend=False)
    ax.set_yscale("log"); ax.set_xlabel(""); ax.set_ylabel("Reads per sample (log scale)")
    ax.set_title(f"{len(ds.control_counts)} controls vs {len(ds.sample_counts)} samples")
    colors = list(sns.color_palette("tab20", 20)[:len(top)]) + [(0.85, 0.85, 0.85)]
    both.T.plot.bar(stacked=True, width=0.7, color=colors, ax=ax2, legend=False, linewidth=0, rot=0)
    ax2.set_ylim(0, 1); ax2.set_ylabel("Mean relative abundance")
    ax2.set_title(f"{level}-level composition: samples vs controls")
    ax2.legend(*ax2.get_legend_handles_labels(), bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title=level)
    plt.tight_layout()
    save_and_show(fig, ds, "controls", level=level, top=top_n)


def explore_controls(ds):
    if ds.control_counts.empty:
        print("This dataset has no controls.")
        return
    widgets.interact(plot_controls, ds=widgets.fixed(ds), level=dropdown(list(LEVELS), "Genus", "Level"),
                     top_n=slider(3, 20, 10, "Taxa shown"))


def control_report(ds, top_n=5):
    """Reads and most abundant taxa in each control, deepest first."""
    tax = ds.taxonomy.view(pd.Series)
    rows = []
    for sid, row in ds.control_counts.loc[ds.control_counts.sum(axis=1).sort_values(ascending=False).index].iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_n)
        rows.append({"control": sid, "reads": int(row.sum()),
                     "top taxa": ", ".join(f"{short_label(tax.get(f, f))} ({n / row.sum():.0%})" for f, n in top.items())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------- composition
def add_composition(ds, save_qzv=False):
    """Sum ASV counts within each taxon at every level (phylum ... genus)."""
    counts = ds.table.view(pd.DataFrame)
    ranks = ds.taxonomy.view(pd.Series).reindex(counts.columns).fillna("Unassigned").str.split(";")
    ranks = ranks.apply(lambda r: [x.strip() for x in r])
    ds.collapsed, ds.rel_at = {}, {}
    for name, lvl in LEVELS.items():
        g = counts.T.groupby(ranks.str[:lvl].str.join(";").values).sum().T
        ds.collapsed[name] = g
        rel = g.div(g.sum(axis=1), axis=0)
        rel.columns = [short_label(c) for c in rel.columns]
        ds.rel_at[name] = rel.T.groupby(level=0).sum().T
    if save_qzv:  # interactive QIIME 2 bar plot for view.qiime2.org (slow for large studies)
        (RESULTS / ds.name).mkdir(parents=True, exist_ok=True)
        taxa.visualizers.barplot(table=ds.table, taxonomy=ds.taxonomy, metadata=ds.md).visualization.save(
            str(RESULTS / ds.name / "taxa-barplot.qzv"))


def top_taxa(ds, level="Genus", n=15, subject=None):
    """The most abundant taxa: mean relative abundance, how many samples and subjects carry each one,
    and in how many subjects it is the most abundant taxon (averaging each subject's samples)."""
    subject = subject or ds.subject
    rel = ds.rel_at[level]
    per_subject = rel.groupby(ds.meta.loc[rel.index, subject]).mean()
    dominant = per_subject.idxmax(axis=1).value_counts()
    out = pd.DataFrame({
        "mean relative abundance": rel.mean(),
        "samples carrying it": (rel > 0).mean(),
        f"{subject}s carrying it": (per_subject > 0).mean(),
        f"dominant in (of {len(per_subject)} {subject}s)": dominant,
    }).fillna({f"dominant in (of {len(per_subject)} {subject}s)": 0})
    out = out.sort_values("mean relative abundance", ascending=False).head(n)
    out[out.columns[-1]] = out[out.columns[-1]].astype(int)
    out.index.name = level
    return out.style.format({c: "{:.1%}" for c in out.columns[:3]})


def plot_composition(ds, level="Genus", group=None, top_n=10, group_means=False):
    group = group or ds.group
    rel = ds.rel_at[level]
    top = rel.mean().sort_values(ascending=False).index[:top_n]
    df = rel[top].assign(Other=1 - rel[top].sum(axis=1))
    grp = ds.meta[group].astype(str)
    if group_means:
        df = df.groupby(grp).mean()
    else:
        order = df.assign(_g=grp, _d=df[top[0]]).sort_values(["_g", "_d"], ascending=[True, False]).index
        df = df.loc[order]
    colors = list(sns.color_palette("tab20", 20)[:top_n]) + [(0.85, 0.85, 0.85)]
    fig, ax = plt.subplots(figsize=(8 if group_means else min(16, max(8, 0.16 * len(df) + 3)), 4.2))
    df.plot.bar(stacked=True, width=0.9 if group_means else 1.0, color=colors, ax=ax, legend=False, linewidth=0)
    ax.set_ylim(0, 1); ax.set_ylabel("Relative abundance"); ax.set_xlabel("")
    if group_means:
        ax.set_xticklabels(df.index, rotation=0)
    else:
        g = grp.loc[df.index].values
        edges = [0] + [i for i in range(1, len(g)) if g[i] != g[i - 1]] + [len(g)]
        ax.set_xticks([(a + b - 1) / 2 for a, b in zip(edges[:-1], edges[1:])])
        ax.set_xticklabels([g[a] for a in edges[:-1]], rotation=0)
        for e in edges[1:-1]:
            ax.axvline(e - 0.5, color="white", lw=2)
    ax.set_title(f"{level}-level composition by {group}")
    ax.legend(*ax.get_legend_handles_labels(), bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title=level)
    save_and_show(fig, ds, "composition", level=level, by=group, top=top_n, means=group_means)


def explore_composition(ds):
    widgets.interact(plot_composition, ds=widgets.fixed(ds),
                     level=dropdown(list(LEVELS), "Genus", "Level"),
                     group=dropdown(ds.categorical, ds.group, "Group by"),
                     top_n=slider(3, 20, 10, "Taxa shown"),
                     group_means=widgets.Checkbox(value=False, description="Group means"))


# ---------------------------------------------------------------------------------------------------- rarefaction
def add_curves(ds, n_steps=20, iterations=3, seed=0):
    """Rarefaction curves. One shuffle of a sample's reads gives a subsample without replacement at every depth
    at once: the first d reads of the shuffle. Observed features at depth d = features first seen before position d."""
    ds.counts = ds.table.view(pd.DataFrame)
    ds.depth = ds.counts.sum(axis=1)
    hi = np.percentile(ds.depth, 95)
    # very uneven depths: space the steps on a log scale so the low range, where curves flatten, is well sampled
    uneven = hi / max(ds.depth.min(), 1) > 100
    steps = np.geomspace(50, hi, n_steps) if uneven else np.linspace(50, hi, n_steps)
    steps = np.unique(steps.astype(int))
    rng = np.random.default_rng(seed)
    curves = {}
    for sid, row in ds.counts.iterrows():
        v = row.values.astype(int)
        reads = np.repeat(np.arange(v.size), v)
        observed = np.zeros(len(steps))
        for _ in range(iterations):
            _, first_seen = np.unique(rng.permutation(reads), return_index=True)
            observed += np.searchsorted(np.sort(first_seen), steps)
        curves[sid] = np.where(steps <= v.sum(), observed / iterations, np.nan)
    ds.curves = pd.DataFrame(curves, index=steps)
    # x-axis range: twice the depth by which most curves (75th percentile) reach 80% of their own maximum
    reach80 = [ds.curves.index[np.argmax(c.values >= 0.8 * np.nanmax(c.values))] for _, c in ds.curves.items()
               if np.isfinite(c.values).any()]
    ds.curves_xmax = int(min(steps.max(), 2 * np.percentile(reach80, 75)))


def plot_rarefaction(ds, color_by=None, depth_cutoff=1000, group_means=False):
    color_by = color_by or ds.group
    pal = colors_for(ds, color_by)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    levels = ds.meta.loc[ds.curves.columns, color_by].astype(str)
    for lvl, c in pal.items():
        ids = levels.index[levels == lvl]
        if not len(ids):
            continue
        if group_means:
            sub_ = ds.curves[ids]
            sub_ = sub_[sub_.notna().mean(axis=1) >= 0.5]   # only depths that at least half the group reaches
            ax.plot(sub_.index, sub_.mean(axis=1), color=c, lw=2, label=f"{lvl} (n={len(ids)})")
            ax.fill_between(sub_.index, sub_.quantile(0.25, axis=1), sub_.quantile(0.75, axis=1), color=c, alpha=0.2, lw=0)
        else:
            ax.plot(ds.curves.index, ds.curves[ids].values, color=c, alpha=0.35, lw=0.8)
            ax.plot([], [], color=c, label=lvl)
    ax.set_xlim(0, max(ds.curves_xmax, 1.2 * depth_cutoff))
    ax.axvline(depth_cutoff, color="black", ls="--", lw=1)
    ax.set_xlabel("Reads sampled"); ax.set_ylabel("Observed features"); ax.legend(title=color_by, fontsize=8)
    ax.set_title("Rarefaction curves" + (" (mean, 25th-75th percentile)" if group_means else ""))
    lost = ds.depth < depth_cutoff
    sns.histplot(x=ds.depth, hue=ds.meta[color_by].astype(str), palette=pal, multiple="stack", bins=25, ax=ax2,
                 legend=False, log_scale=bool(ds.depth.max() / max(ds.depth.min(), 1) > 100))
    ax2.axvline(depth_cutoff, color="black", ls="--", lw=1)
    ax2.set_xlabel("Reads per sample")
    ax2.set_title(f"{lost.sum()} of {len(ds.depth)} samples below {depth_cutoff:,} reads")
    plt.tight_layout()
    save_and_show(fig, ds, "rarefaction", by=color_by, depth=depth_cutoff, means=group_means)


def explore_rarefaction(ds):
    default = int(max(100, round(ds.depth.quantile(0.1), -2)))
    top = int(min(ds.depth.max(), max(2 * ds.curves_xmax, ds.depth.median())))
    ds.depth_slider = slider(100, top, default, "Rarefaction depth", step=100 if top <= 50_000 else 500)
    widgets.interact(plot_rarefaction, ds=widgets.fixed(ds),
                     color_by=dropdown(ds.categorical, ds.group, "Colour by"), depth_cutoff=ds.depth_slider,
                     group_means=widgets.Checkbox(value=False, description="Average by group"))


# ---------------------------------------------------------------------------------------------------- diversity
def add_diversity(ds, depth=None, save_qzv=False):
    """Rarefy once, then compute alpha and beta diversity with QIIME 2 and a 3-axis PCoA for each metric."""
    ds.rarefaction_depth = depth or ds.depth_slider.value
    lost = ds.depth[ds.depth < ds.rarefaction_depth].index
    print(f"Rarefying to {ds.rarefaction_depth:,} reads; {len(lost)} of {len(ds.depth)} samples dropped")
    table = ds.table
    if ds.tree is not None:
        # phylogenetic metrics need every feature on the tree; drop any that are missing
        table = phylogeny.methods.filter_table(table=ds.table, tree=ds.tree).filtered_table
        kept = table.view(pd.DataFrame)
        n_lost = ds.counts.shape[1] - kept.shape[1]
        if n_lost:
            share = 1 - kept.values.sum() / ds.counts.values.sum()
            print(f"{n_lost} features ({share:.1%} of reads) are not in the tree and are excluded from diversity analyses")
    rare = feature_table.methods.rarefy(table=table, sampling_depth=ds.rarefaction_depth).rarefied_table

    alpha = {"observed_features": diversity_lib.methods.observed_features(table=rare).vector,
             "shannon": diversity_lib.methods.shannon_entropy(table=rare).vector}
    beta = {"bray_curtis": diversity_lib.methods.bray_curtis(table=rare).distance_matrix,
            "jaccard": diversity_lib.methods.jaccard(table=rare).distance_matrix}
    if ds.tree is not None:
        alpha["faith_pd"] = diversity_lib.methods.faith_pd(table=rare, phylogeny=ds.tree).vector
        beta["unweighted_unifrac"] = diversity_lib.methods.unweighted_unifrac(table=rare, phylogeny=ds.tree).distance_matrix
        beta["weighted_unifrac"] = diversity_lib.methods.weighted_unifrac(table=rare, phylogeny=ds.tree).distance_matrix
    ds.alpha_metrics, ds.beta_metrics = list(alpha), list(beta)
    ds.alpha = pd.concat({m: a.view(pd.Series) for m, a in alpha.items()}, axis=1)
    ds.dms = {m: d.view(skbio.DistanceMatrix) for m, d in beta.items()}
    ds.pcoas = {m: pcoa(d, method="fsvd", number_of_dimensions=3) for m, d in ds.dms.items()}   # first 3 axes only
    ds.permanova_cache = {}
    if save_qzv:  # interactive 3D EMPeror plots for view.qiime2.org
        (RESULTS / ds.name).mkdir(parents=True, exist_ok=True)
        for m, pc in ds.pcoas.items():
            emperor.visualizers.plot(pcoa=Artifact.import_data("PCoAResults", pc), metadata=ds.md).visualization.save(
                str(RESULTS / ds.name / f"{m}-emperor.qzv"))
    return ds.alpha.join(ds.meta[ds.group]).groupby(ds.group).agg(["mean", "std"]).round(2)


def plot_alpha(ds, metric="shannon", group=None):
    """Box plot of an alpha metric by group, treating every sample as independent."""
    group = group or ds.group
    df = ds.alpha[[metric]].join(ds.meta[group].astype(str))
    pal = colors_for(ds, group)
    order = sorted(df[group].unique())
    fig, ax = plt.subplots(figsize=(1.3 * len(order) + 2.5, 4))
    sns.boxplot(data=df, x=group, y=metric, order=order, hue=group, palette=pal, fill=False, width=0.55,
                showfliers=False, linewidth=1.4, ax=ax, legend=False)
    sns.stripplot(data=df, x=group, y=metric, order=order, hue=group, palette=pal, size=5, alpha=0.75,
                  jitter=0.15, ax=ax, legend=False)
    groups = [g[metric].values for _, g in df.groupby(group)]
    test = stats.mannwhitneyu(*groups) if len(groups) == 2 else stats.kruskal(*groups)
    name = "Mann–Whitney" if len(groups) == 2 else "Kruskal–Wallis"
    ax.set_title(f"{metric.replace('_', ' ')} by {group}\n{name} p = {test.pvalue:.3g}")
    ax.set_xlabel(""); ax.set_ylabel(metric.replace("_", " "))
    save_and_show(fig, ds, "alpha", metric=metric, by=group)


def explore_alpha(ds):
    widgets.interact(plot_alpha, ds=widgets.fixed(ds), metric=dropdown(ds.alpha_metrics, "shannon", "Metric"),
                     group=dropdown(ds.categorical, ds.group, "Group by"))


def plot_paired_alpha(ds, metric="shannon", group=None, pair_by=None):
    """Paired comparison: one line per subject joining its two samples, with a Wilcoxon signed-rank test."""
    group, pair_by = group or ds.group, pair_by or ds.subject
    df = ds.alpha[[metric]].join(ds.meta[[group, pair_by]].astype(str))
    wide = df.pivot_table(index=pair_by, columns=group, values=metric, aggfunc="mean")
    levels = sorted(df[group].unique())
    if len(levels) != 2:
        print(f"A paired test needs exactly two levels of {group}; found {levels}.")
        return
    wide = wide[levels].dropna()
    pal = colors_for(ds, group)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for _, (a, b) in wide.iterrows():
        ax.plot([0, 1], [a, b], color="#999999", lw=1, alpha=0.7, zorder=1)
    for x, lvl in enumerate(levels):
        ax.scatter(np.full(len(wide), x), wide[lvl], color=pal[lvl], s=36, zorder=2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(levels); ax.set_xlim(-0.4, 1.4)
    ax.set_ylabel(metric.replace("_", " "))
    if len(wide) >= 2:
        test = stats.wilcoxon(wide[levels[0]], wide[levels[1]])
        sub = f"Wilcoxon signed-rank p = {test.pvalue:.3g}, {len(wide)} pairs"
    else:
        sub = f"only {len(wide)} complete pair(s)"
    ax.set_title(f"{metric.replace('_', ' ')}: {levels[0]} vs {levels[1]} within {pair_by}\n{sub}")
    save_and_show(fig, ds, "paired-alpha", metric=metric, by=group, pairs=pair_by)


def explore_paired_alpha(ds, group=None, pair_by=None):
    widgets.interact(plot_paired_alpha, ds=widgets.fixed(ds), metric=dropdown(ds.alpha_metrics, "shannon", "Metric"),
                     group=widgets.fixed(group or ds.group), pair_by=widgets.fixed(pair_by or ds.subject))


# ---------------------------------------------------------------------------------------------------- beta diversity
def plot_distance_matrix(ds, metric="bray_curtis", group=None):
    group = group or ds.group
    dm = ds.dms[metric]
    ids = list(dm.ids)
    grp = ds.meta.loc[ids, group].astype(str)
    order = grp.sort_values(kind="stable").index
    df = dm.to_data_frame().loc[order, order]
    pal = colors_for(ds, group)
    strip = list(grp.loc[order].map(pal))
    same = grp.values[:, None] == grp.values[None, :]
    off = ~np.eye(len(ids), dtype=bool)
    within, between = dm.data[same & off].mean(), dm.data[~same].mean()
    g = sns.clustermap(df, row_cluster=False, col_cluster=False, row_colors=strip, col_colors=strip, cmap=SEQ_CMAP,
                       xticklabels=False, yticklabels=False, figsize=(6.5, 6.5), dendrogram_ratio=0.02,
                       colors_ratio=0.025, rasterized=True, cbar_pos=(0.3, -0.02, 0.45, 0.022),
                       cbar_kws={"orientation": "horizontal", "label": f"{metric.replace('_', ' ')} distance"})
    g.ax_heatmap.set_xlabel(""); g.ax_heatmap.set_ylabel("")
    if len(pal) <= 12:
        for lvl, c in pal.items():
            g.ax_heatmap.bar(0, 0, color=c, label=lvl)
        g.ax_heatmap.legend(title=group, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    g.fig.suptitle(f"{metric.replace('_', ' ')}: mean within-group {within:.2f}, between-group {between:.2f}",
                   y=1.02, fontweight="bold", fontsize=11)
    save_and_show(g.fig, ds, "distance-matrix", metric=metric, by=group)


def explore_distance_matrix(ds):
    widgets.interact(plot_distance_matrix, ds=widgets.fixed(ds),
                     metric=dropdown(ds.beta_metrics, "bray_curtis", "Metric"),
                     group=dropdown(ds.categorical, ds.group, "Group by"))


def plot_within_between(ds, metric="bray_curtis", by=None):
    """Distances between samples from the same subject vs from different subjects."""
    by = by or ds.subject
    dm = ds.dms[metric]
    lab = ds.meta.loc[list(dm.ids), by].astype(str).values
    iu = np.triu_indices(len(lab), k=1)
    d = dm.data[iu]
    same = lab[iu[0]] == lab[iu[1]]
    df = pd.DataFrame({"distance": d, "pair": np.where(same, f"same {by}", f"different {by}")})
    order = [f"same {by}", f"different {by}"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.boxplot(data=df, x="pair", y="distance", order=order, color="white", width=0.5, showfliers=False, ax=ax)
    sns.stripplot(data=df[df.pair == order[0]], x="pair", y="distance", order=order, color="#4C72B0", size=5, ax=ax)
    sns.stripplot(data=df[df.pair == order[1]], x="pair", y="distance", order=order, color="#999999", size=2.5,
                  alpha=0.5, jitter=0.25, ax=ax)
    sub = ""
    if same.sum() and (~same).sum():
        test = stats.mannwhitneyu(d[same], d[~same], alternative="less")
        sub = f"\n{same.sum()} same vs {(~same).sum()} different pairs; one-sided Mann–Whitney p = {test.pvalue:.3g}"
    ax.set_title(f"{metric.replace('_', ' ')} distance{sub}", fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel(f"{metric.replace('_', ' ')} distance")
    save_and_show(fig, ds, "within-between", metric=metric, by=by)


def explore_within_between(ds, by=None):
    widgets.interact(plot_within_between, ds=widgets.fixed(ds),
                     metric=dropdown(ds.beta_metrics, "bray_curtis", "Metric"), by=widgets.fixed(by or ds.subject))


def run_permanova(ds, metric, column):
    key = (metric, column)
    if key not in ds.permanova_cache:
        dm = ds.dms[metric]
        grouping = ds.meta.loc[list(dm.ids), column].astype(str)
        if grouping.nunique() < 2 or grouping.nunique() == len(grouping):
            ds.permanova_cache[key] = None
        else:
            r = permanova(dm, grouping, permutations=999)
            ds.permanova_cache[key] = (r["test statistic"], r["p-value"])
    return ds.permanova_cache[key]


def permanova_table(ds, columns=None):
    columns = columns or [ds.group, ds.subject]
    rows = [{"metric": m, "grouping": c, "pseudo-F": round(r[0], 2), "p": r[1]}
            for m in ds.beta_metrics for c in columns if c in ds.meta and (r := run_permanova(ds, m, c))]
    return pd.DataFrame(rows)


def plot_pcoa(ds, metric="bray_curtis", color_by=None, axes="PCo1 vs PCo2", connect=None, label_points=False):
    """PCoA scatter (equal axis scaling) with PERMANOVA for the colour field. Optionally join samples sharing a value
    of `connect`, and label each point with its subject."""
    color_by = color_by or ds.group
    pc = ds.pcoas[metric]
    i, j = [int(a[3:]) - 1 for a in axes.split(" vs ")]
    coords = pc.samples.iloc[:, [i, j]].join(ds.meta[color_by].astype(str))
    xc, yc = coords.columns[0], coords.columns[1]
    expl = pc.proportion_explained * 100
    pal = colors_for(ds, color_by)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if connect:
        for _, grp_ in coords.join(ds.meta[connect].astype(str).rename("_c")).groupby("_c"):
            if len(grp_) > 1:
                ax.plot(grp_[xc], grp_[yc], color="#BBBBBB", lw=0.8, zorder=1)
    sns.scatterplot(data=coords, x=xc, y=yc, hue=color_by, palette=pal, s=55,
                    alpha=0.85, edgecolor="white", linewidth=0.5, ax=ax, zorder=2)
    if label_points:
        for sid, row in coords.iterrows():
            ax.annotate(str(ds.meta.loc[sid, ds.subject]), (row[xc], row[yc]), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color="#444444")
    ax.set_aspect("equal", adjustable="datalim")   # distances on the plot match distances between samples
    ax.set_xlabel(f"PCo{i + 1} ({expl.iloc[i]:.0f}%)"); ax.set_ylabel(f"PCo{j + 1} ({expl.iloc[j]:.0f}%)")
    ax.set_xticks([]); ax.set_yticks([])
    res = run_permanova(ds, metric, color_by)
    sub = f"PERMANOVA by {color_by}: pseudo-F = {res[0]:.2f}, p = {res[1]:.3g}" if res else ""
    ax.set_title(f"{metric.replace('_', ' ')} PCoA\n{sub}")
    if len(pal) <= 12:
        ax.legend(title=color_by, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    else:
        ax.legend().remove()
    save_and_show(fig, ds, "pcoa", metric=metric, by=color_by, axes=axes.replace(" vs ", "-"),
                  **({"connect": connect} if connect else {}), **({"labels": True} if label_points else {}))


def explore_pcoa(ds, connect=None, labels=None):
    """Interactive PCoA. Point labels default to on when samples are connected (e.g. two nostrils per person)."""
    labels = bool(connect) if labels is None else labels
    widgets.interact(plot_pcoa, ds=widgets.fixed(ds), metric=dropdown(ds.beta_metrics, "bray_curtis", "Metric"),
                     color_by=dropdown(ds.categorical, ds.group, "Colour by"),
                     axes=dropdown(["PCo1 vs PCo2", "PCo1 vs PCo3", "PCo2 vs PCo3"], "PCo1 vs PCo2", "Axes"),
                     connect=widgets.fixed(connect),
                     label_points=widgets.Checkbox(value=labels, description=f"Label {ds.subject}"))


# ---------------------------------------------------------------------------------------------------- differential abundance
def add_differential_abundance(ds, min_prevalence=0.25, reference=None, group=None, covariates=(), pseudocount=1):
    """ANCOM-BC (scikit-bio) on genus-level counts for `group` (default ds.group), optionally adjusting for
    covariates such as the participant in a paired design. A pseudocount replaces zeros, which ANCOM-BC requires."""
    group = group or ds.group
    g = ds.collapsed["Genus"]
    g = g.loc[:, (g > 0).mean() >= min_prevalence]
    levels = sorted(ds.meta[group].astype(str).unique())
    print(f"Testing {g.shape[1]} of {ds.collapsed['Genus'].shape[1]} genera for {group}"
          + (f", adjusting for {', '.join(covariates)}" if covariates else ""))
    ds.da_by = getattr(ds, "da_by", {})
    if g.shape[1] < 2 or len(levels) < 2:
        print("Not enough genera or groups to test; lower min_prevalence or check the grouping field.")
        ds.da = ds.da_by[group] = pd.DataFrame(columns=["taxon", "comparison", "log2_fold_change", "se", "q"])
        return
    ref = reference or levels[0]
    term = f"C({group}, Treatment(reference='{ref}'))"
    md = ds.meta.loc[g.index, [group, *covariates]].astype(str)
    res = ancombc(g + pseudocount, md, " + ".join([term, *covariates]))
    res = res.reset_index()
    res = res[res["Covariate"].str.startswith(term)]   # the group's terms only, not the intercept or covariates
    da = pd.DataFrame({
        "taxon": res["FeatureID"].map(short_label).values,
        "comparison": res["Covariate"].str.extract(r"\[T\.(.*)\]$")[0].values + f" vs {ref}",
        "log2_fold_change": res["Log2(FC)"].values, "se": res["SE"].values, "q": res["qvalue"].values})
    dup = da.groupby("comparison")["taxon"].transform(lambda t: t.groupby(t).cumcount())
    da.loc[dup > 0, "taxon"] += " #" + (dup[dup > 0] + 1).astype(str)   # distinct lineages sharing a short name
    out = RESULTS / ds.name
    out.mkdir(parents=True, exist_ok=True)
    da.to_csv(out / f"ancombc-{group}.tsv", sep="\t", index=False)
    ds.da = ds.da_by[group] = da
    print(f"{(da.q < 0.05).sum()} genus–comparison pairs with q < 0.05 (Holm-adjusted); full table in {out}/ancombc-{group}.tsv")


def plot_forest(ds, comparison, top_n=15, q_max=0.05, group=None):
    da = ds.da_by[group] if group else ds.da
    df = da[da["comparison"] == comparison]
    df = df.reindex(df["log2_fold_change"].abs().sort_values(ascending=False).index).head(top_n)
    df = df.sort_values("log2_fold_change")
    sig = df["q"] < q_max
    color = np.where(~sig, NS, np.where(df["log2_fold_change"] > 0, UP, DOWN))
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(df) + 1.4))
    y = np.arange(len(df))
    ax.hlines(y, df["log2_fold_change"] - 1.96 * df["se"], df["log2_fold_change"] + 1.96 * df["se"], color=color, lw=1.6)
    ax.scatter(df["log2_fold_change"], y, color=color, s=36, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(df["taxon"])
    ax.set_xlabel("Log2 fold change, 95% CI"); ax.set_ylabel("")
    ax.set_title(f"{comparison}: {sig.sum()} of {len(df)} shown with q < {q_max:g}")
    save_and_show(fig, ds, "forest", comparison=comparison, top=top_n, q=f"{q_max:g}")


def explore_forest(ds, group=None):
    """Forest plot of the ANCOM-BC results for `group` (default: the most recent run)."""
    da = ds.da_by[group] if group else ds.da
    comparisons = sorted(da["comparison"].unique())
    if not comparisons:
        print("No differential abundance results to plot.")
        return
    widgets.interact(plot_forest, ds=widgets.fixed(ds), group=widgets.fixed(group),
                     comparison=dropdown(comparisons, comparisons[0], "Comparison"),
                     top_n=slider(5, 40, 15, "Genera shown"),
                     q_max=widgets.FloatLogSlider(value=0.05, base=10, min=-4, max=0, step=0.25, description="q cut-off",
                                                  continuous_update=False, style={"description_width": "initial"}))
