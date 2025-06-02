import logging
from pathlib import Path

import bct
import networkx as nx
import numpy as np
import polars as pl
import pydantic


class Module(pydantic.BaseModel):
    index: int
    name: str
    is_biomarker: bool


MODULE_INFO = (
    Module(index=0, name="mPFCventral-amyg", is_biomarker=False),
    Module(index=1, name="OFC-amyg-hipp", is_biomarker=False),
    Module(index=2, name="mPFCdorsal-amyg-NAc", is_biomarker=True),
)


def get_weighted_adjacency_matrix(
    graph: nx.Graph, attribute: str = "value"
) -> np.ndarray:
    """
    Returns a symmetric NumPy matrix representing the weighted adjacency matrix
    of a NetworkX graph.

    Args:
        graph (nx.Graph): The input NetworkX graph with weighted edges.

    Returns:
        np.ndarray: A symmetric NumPy array where the (i, j)-th entry is the
                      weight of the edge between the i-th and j-th nodes (and
                      vice versa), and 0 if no edge exists.
    """
    n = graph.number_of_nodes()
    node_to_index = {node: i for i, node in enumerate(graph.nodes())}
    weighted_adj_matrix = np.zeros((n, n))

    for u, v, data in graph.edges(data=True):
        u_index = node_to_index[u]
        v_index = node_to_index[v]
        weight = data.get(attribute, 0)  # Default to 0 if 'weight' attribute is missing
        weighted_adj_matrix[u_index, v_index] = weight
        weighted_adj_matrix[v_index, u_index] = weight  # Ensure symmetry

    return weighted_adj_matrix


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
            pl.col("source").cast(pl.UInt32),
            pl.col("target").cast(pl.UInt32),
            pl.col("value").cast(pl.Float64),
        )
        .filter(pl.col("source") < pl.col("target"))
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
        .with_columns(
            pl.col("x").cast(pl.UInt32),
            pl.col("y").cast(pl.UInt32),
            pl.col("z").cast(pl.UInt32),
            pl.col("roi").cast(pl.UInt32),
            pl.col("index").cast(pl.UInt32),
        )
    )


def read_weighted_matrix1_df(src: Path, target_density: float = 0.1) -> pl.DataFrame:
    matrix1 = read_matrix1(src / "fdt_matrix1.dot.gz")
    coordinates = read_matrix1_coordinates(src / "coords_for_fdt_matrix1")
    d = (
        coordinates.select("index")
        .rename({"index": "source"})
        .join(coordinates.select("index").rename({"index": "target"}), how="cross")
        .filter(pl.col("source") < pl.col("target"))
        .join(matrix1, on=["source", "target"], how="left")
        .fill_null(strategy="zero")
        .join(
            coordinates.drop(["x", "y", "z"]).rename({"roi": "roi_source"}),
            left_on="source",
            right_on="index",
            how="left",
        )
        .join(
            coordinates.drop(["x", "y", "z"]).rename({"roi": "roi_target"}),
            left_on="target",
            right_on="index",
            how="left",
        )
    )
    mat = matrix1_to_weighted_adjacency(d)
    # Normalize weights
    bct.weight_conversion(mat, "normalize", copy=False)
    bct.threshold_proportional(mat, target_density, copy=False)
    mat = (mat > 0).astype(np.float64)

    return (
        pl.DataFrame(mat)
        .with_row_index("source", offset=1)
        .unpivot(index="source", variable_name="target")
        .with_columns(pl.col("target").str.strip_prefix("column_").cast(pl.UInt32) + 1)
        .filter(pl.col("source") < pl.col("target"))
        .join(d.drop("value"), on=["source", "target"], how="left")
    )


def matrix1_to_weighted_adjacency(d: pl.DataFrame) -> np.ndarray:
    g = nx.from_pandas_edgelist(d.to_pandas(), edge_attr=True)
    return get_weighted_adjacency_matrix(g)


def matrix1_to_adjacency(d: pl.DataFrame) -> np.ndarray:
    g = nx.from_pandas_edgelist(d.filter(pl.col("value") > 0.5).to_pandas())
    g.add_nodes_from(d["source"].unique().to_list())
    return nx.adjacency_matrix(g).toarray()


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


def matrix1_to_adjacency_subset(d: pl.DataFrame, roi: int) -> np.ndarray:
    mat = matrix1_to_adjacency(d)
    roi_idx = get_roi_idx_from_df(d, roi)
    return mat[np.ix_(roi_idx, roi_idx)]


def summarize_adjacency(mat: np.ndarray) -> pl.DataFrame:
    Eglob = bct.efficiency_bin(mat)
    Ccoef = bct.clustering_coef_bu(mat).mean()
    Betw = bct.betweenness_bin(mat).mean()
    Dist = 1.0 / Eglob
    Deg = bct.degrees_und(mat)
    Deg_mean = np.mean(Deg)
    density, _, _ = bct.density_und(mat)
    nvox = mat.shape[0]
    WM_conn = np.sum(Deg) / (nvox * (nvox - 1) / 2)
    return pl.DataFrame(
        {
            "Eglob": Eglob,
            "Ccoef": Ccoef,
            "Betw": Betw,
            "Dist": Dist,
            "Deg_mean": Deg_mean,
            "Density": density,
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
        mat = matrix1_to_adjacency_subset(d, module.index)
        rows.append(
            summarize_adjacency(mat).with_columns(
                ModuleNumber=module.index,
                ModuleName=pl.lit(module.name),
                IsBiomarker=module.is_biomarker,
            )
        )

    # Whole-network summary
    logging.info("summarizing network")
    rows.append(
        summarize_adjacency(matrix1_to_adjacency(d)).with_columns(
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
