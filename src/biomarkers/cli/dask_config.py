from dask import config


def set_config() -> None:
    # It would be prefereable to have paralellism that never spilled to the disk
    # Since we're using Dask, we'll just turn off that feature
    # config.set({"distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_": 0})
    config.set(
        {"distributed.worker.memory.rebalance.measure": "managed_in_memory"}
    )
    config.set({"distributed.worker.memory.spill": False})
    config.set({"distributed.worker.memory.target": False})
    config.set({"distributed.worker.memory.pause": False})
    config.set({"distributed.worker.memory.terminate": False})
    config.set({"distributed.comm.timeouts.connect": "90s"})
    config.set({"distributed.comm.timeouts.tcp": "90s"})
