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
    # both source -> target and target -> source
    i, j, value = np.loadtxt(src).T
    # multiply by 2 because probtrackx stores either integers
    # or half numbers, so x2 brings us to integers. This is good
    # because the only 16-bit dtype sparse arrays allow are integers
    coo = sparse.coo_array(
        (
            np.astype(value * 2, np.uint16),
            (np.astype(i - 1, np.uint16), np.astype(j - 1, np.uint16)),
        )
    )
    # not worrying about normalizing
    coo += coo.T
    # unpivoting would hog memory, so creating index manually
    return pl.DataFrame(
        {
            "value": coo.todense().ravel(),
            "source": np.repeat(np.arange(coo.shape[0], dtype=np.uint16), coo.shape[1]),
            "target": np.tile(np.arange(coo.shape[1], dtype=np.uint16), coo.shape[0]),
        }
    ).filter(pl.col("source") < pl.col("target"))


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
        .with_columns(pl.col("roi").cast(pl.UInt8), pl.col("index").cast(pl.UInt16) - 1)
    )


def threshold_proportional_bin(
    d: pl.DataFrame, target_density: float = 0.1, col="value"
) -> pl.DataFrame:
    density: float = (d[col] > 0).mean()  # type: ignore
    if density <= target_density:
        return d.with_columns(pl.col(col).cast(pl.Boolean))

    return d.with_columns(value=pl.col(col) > pl.col(col).quantile(1 - target_density))


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
    )


def matrix1_to_graph(d: pl.DataFrame) -> nx.Graph:
    g = nx.from_pandas_edgelist(d.filter(pl.col("value")).to_pandas())
    g.add_nodes_from(d["source"].unique().to_list())
    return g


def get_roi_idx_from_df(d: pl.DataFrame, roi: int) -> np.ndarray:
    return (
        d.filter(pl.col("roi_source") == roi)
        .select("source")
        .unique()
        .to_series()
        .to_numpy()
    )


def matrix1_to_subset(d: pl.DataFrame, roi: int) -> nx.Graph:
    g = matrix1_to_graph(d)
    roi_idx = get_roi_idx_from_df(d, roi)
    return g.subgraph(roi_idx)


def summarize_graph(g: nx.Graph) -> pl.DataFrame:
    Degs: list[int] = list(dict(g.degree).values())  # type: ignore
    nvox = g.number_of_nodes()
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
        rows.append(
            summarize_graph(matrix1_to_subset(d, module.index)).with_columns(
                ModuleNumber=module.index,
                ModuleName=pl.lit(module.name),
                IsBiomarker=module.is_biomarker,
            )
        )

    # Whole-network summary
    logging.info("summarizing network")
    rows.append(
        summarize_graph(matrix1_to_graph(d)).with_columns(
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
