import logging
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import pydantic
from scipy import sparse


class Module(pydantic.BaseModel):
    index: int
    name: str
    is_biomarker: bool


MODULE_INFO = (
    Module(index=0, name="mPFCventral-amyg", is_biomarker=False),
    Module(index=1, name="OFC-amyg-hipp", is_biomarker=False),
    Module(index=2, name="mPFCdorsal-amyg-NAc", is_biomarker=True),
)


def read_matrix1(src: Path) -> pl.DataFrame:
    # need to symmetrize because probtrackx stores
    # both source -> target and target -> source,
    # but only one of each edge
    i, j, value = np.loadtxt(src).T
    coo = sparse.coo_array((value, (np.astype(i - 1, int), np.astype(j - 1, int))))
    coo += coo.T
    coo /= 2
    return (
        pl.DataFrame(coo.todense())
        .with_row_index("source")
        .unpivot(index="source", variable_name="target")
        .with_columns(
            pl.col("target").str.strip_prefix("column_").cast(pl.UInt16) + 1,
            pl.col("source").cast(pl.UInt16) + 1,
        )
        .filter(pl.col("source") < pl.col("target"))
        .with_columns(value=pl.col("value") / pl.col("value").abs().max())
    )


def read_matrix1_coordinates(src: Path) -> pl.DataFrame:
    return (
        pl.read_csv(src, has_header=False)
        .with_columns(
            pl.col("column_1")
            .str.split("  ")
            .list.to_struct(fields=["x", "y", "z", "roi", "index"])
        )
        .unnest("column_1")
        .drop("x", "y", "z")
        .with_columns(pl.col("roi").cast(pl.UInt8), pl.col("index").cast(pl.UInt16))
    )


def threshold_proportional_bin(
    d: pl.DataFrame, target_density: float = 0.1
) -> pl.DataFrame:
    density: float = (d["value"] > 0).mean()  # type: ignore
    if density <= target_density:
        return d.with_columns(pl.col("value").cast(pl.Boolean))

    return d.with_columns(
        value=pl.col("value") > pl.col("value").quantile(1 - target_density)
    )


def read_weighted_matrix1_df(src: Path, target_density: float = 0.1) -> pl.DataFrame:
    coordinates = read_matrix1_coordinates(src / "coords_for_fdt_matrix1")
    matrix1 = read_matrix1(src / "fdt_matrix1.dot.gz")
    return (
        matrix1.join(
            coordinates.rename({"roi": "roi_source"}),
            left_on="source",
            right_on="index",
        )
        .join(
            coordinates.rename({"roi": "roi_target"}),
            left_on="target",
            right_on="index",
        )
        .pipe(threshold_proportional_bin, target_density)
        .with_columns(pl.col("source") - 1, pl.col("target") - 1)
    )


def matrix1_to_graph(d: pl.DataFrame) -> nx.Graph:
    g = nx.from_pandas_edgelist(d.filter(pl.col("value")).to_pandas())
    g.add_nodes_from(d["source"].unique().to_list())
    return g


def get_roi_idx_from_df(d: pl.DataFrame, roi: int) -> np.ndarray:
    return np.flatnonzero(
        d.select("source", "roi_source")
        .unique()
        .sort("source")
        .select("roi_source")
        .to_series()
        .to_numpy()
        == roi
    )


def matrix1_to_subset(d: pl.DataFrame, roi: int) -> nx.Graph:
    g = matrix1_to_graph(d)
    roi_idx = get_roi_idx_from_df(d, roi)
    return g.subgraph(roi_idx)


def summarize_adjacency(g: nx.Graph) -> pl.DataFrame:
    Degs: list[int] = list(dict(g.degree).values())  # type: ignore
    nvox = len(g.nodes)
    WM_conn = np.sum(Degs) / (nvox * (nvox - 1) / 2)
    return pl.DataFrame(
        {"Deg_mean": np.mean(Degs), "Density": nx.density(g), "WM_connections": WM_conn}
    )


def dwi_biomarker1_flow(
    outdir: Path, sub: str, ses: str, target_density: float = 0.1
) -> None:
    participant_label = f"sub-{sub}"
    session_label = f"ses-{ses}"

    logging.info("reading matrix1")
    d = read_weighted_matrix1_df(
        outdir / "probtrackx" / participant_label / session_label / "dwi",
        target_density=target_density,
    )

    # Module-level summary
    logging.info("summarizing modules")
    rows: list[pl.DataFrame] = []
    for module in MODULE_INFO:
        g = matrix1_to_subset(d, module.index)
        rows.append(
            summarize_adjacency(g).with_columns(
                ModuleNumber=module.index,
                ModuleName=pl.lit(module.name),
                IsBiomarker=module.is_biomarker,
            )
        )

    # Whole-network summary
    logging.info("summarizing network")
    rows.append(
        summarize_adjacency(matrix1_to_graph(d)).with_columns(
            ModuleNumber=None, ModuleName=pl.lit("WholeNetwork"), IsBiomarker=False
        )
    )
    logging.info("concatenating summaries")
    out: pl.DataFrame = pl.concat(rows).with_columns(sub=pl.lit(sub), ses=pl.lit(ses))
    logging.info("saving summaries")
    out.lazy().sink_csv(
        outdir
        / "networks"
        / participant_label
        / session_label
        / "dwi"
        / "network_summaries.tsv",
        separator="\t",
        mkdir=True,
    )
    logging.info("finished")
