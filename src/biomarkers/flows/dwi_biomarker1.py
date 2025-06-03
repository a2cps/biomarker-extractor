import logging
from pathlib import Path

import networkx as nx
import numpy as np
import polars as pl
import pydantic
from networkx import algorithms


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
    return (
        pl.read_csv(src, has_header=False)
        .with_columns(
            pl.col("column_1")
            .str.split("  ")
            .list.to_struct(fields=["source", "target", "value"])
        )
        .unnest("column_1")
        .with_columns(
            pl.col("source").cast(pl.UInt16),
            pl.col("target").cast(pl.UInt16),
            pl.col("value").cast(pl.Float64),
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
        .with_columns(
            pl.col("roi").cast(pl.UInt8),
            pl.col("index").cast(pl.UInt16),
        )
    )


def threshold_proportional_bin(
    d: pl.DataFrame, target_density: float = 0.1
) -> pl.DataFrame:
    density: float = (d["value"] > 0).mean()  # type: ignore
    if density < target_density:
        return d.with_columns(pl.col("value").cast(pl.Boolean))

    return d.with_columns(
        value=pl.when(pl.col("value") < pl.col("value").quantile(1 - target_density))
        .then(False)
        .otherwise(True)
    )


def read_weighted_matrix1_df(src: Path, target_density: float = 0.1) -> pl.DataFrame:
    matrix1 = read_matrix1(src / "fdt_matrix1.dot.gz")
    coordinates = read_matrix1_coordinates(src / "coords_for_fdt_matrix1")
    d = (
        coordinates.rename({"index": "source", "roi": "roi_source"})
        .join(coordinates.rename({"index": "target", "roi": "roi_target"}), how="cross")
        .filter(pl.col("source") < pl.col("target"))
        .join(matrix1, on=["source", "target"], how="left")
        .fill_null(strategy="zero")
        .pipe(threshold_proportional_bin, target_density)
        .with_columns(pl.col("source") - 1, pl.col("target") - 1)
    )

    return d


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
    Eglob = algorithms.global_efficiency(g)
    Ccoefs = algorithms.clustering(g)
    if not isinstance(Ccoefs, dict):
        msg = "ccoefs should be dict"
        raise AssertionError(msg)
    Ccoef = np.array(list(Ccoefs.values())).mean()
    Betws = algorithms.betweenness_centrality(g)
    Degs: list[int] = list(dict(g.degree).values())  # type: ignore
    nvox = len(g.nodes)
    WM_conn = np.sum(Degs) / (nvox * (nvox - 1) / 2)
    return pl.DataFrame(
        {
            "Eglob": Eglob,
            "Ccoef": Ccoef,
            "Betw": np.mean(list(Betws.values())),
            "Dist": 1.0 / Eglob,
            "Deg_mean": np.mean(Degs),
            "Density": nx.density(g),
            "WM_connections": WM_conn,
        }
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
            ModuleNumber=0, ModuleName=pl.lit("WholeNetwork"), IsBiomarker=False
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
