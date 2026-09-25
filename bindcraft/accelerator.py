def _oneapi_plugin_installed() -> bool:
    from importlib.util import find_spec

    if find_spec('jax_plugins') is None:
        return False
    return any(find_spec(name) is not None for name in ('jax_plugins.xla_oneapi', 'jax_plugins.oneapi'))


def _local_devices() -> tuple:
    import jax

    return tuple(jax.local_devices())


def _oneapi_devices() -> tuple:
    if not _oneapi_plugin_installed():
        return ()
    devices = _local_devices()
    oneapi_devices = tuple(device for device in devices if device.platform == 'oneapi')
    if oneapi_devices:
        return oneapi_devices
    if not devices or all(device.platform == 'cpu' for device in devices):
        import sys

        raise RuntimeError(f'JAX did not initialize the installed oneAPI plugin; check Intel GPU access and add {sys.prefix}/lib to LD_LIBRARY_PATH')
    return ()


def _oneapi_compiler_options() -> dict[str, str] | None:
    if _oneapi_devices():
        return {'xla_gpu_enable_command_buffer': ''}
    return None


def _oneapi_smallest_indices(values, count: int):
    """Return the k smallest indices without XLA's unsupported SYCL sort."""
    import jax
    import jax.numpy as jnp

    length = values.shape[-1]
    count = min(max(int(count), 0), length)
    positions = jnp.arange(length)
    available = jnp.ones(values.shape, dtype=bool)
    indices = jnp.zeros(values.shape[:-1] + (count,), dtype=jnp.int32)

    def select_next(index, state):
        remaining, selected_indices = state
        next_index = jax.lax.stop_gradient(jnp.argmin(
            jnp.where(remaining, values, jnp.inf),
            axis=-1,
        ))
        selected_indices = selected_indices.at[..., index].set(next_index)
        remaining = remaining & (positions != next_index[..., None])
        return remaining, selected_indices

    return jax.lax.fori_loop(
        0,
        count,
        select_next,
        (available, indices),
    )[1]
