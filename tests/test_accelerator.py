import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bindcraft.accelerator import _oneapi_compiler_options, _oneapi_devices
from bindcraft.cli import design_card_name
from bindcraft.design_workers import campaign_subbatch_size, dispatch_design_workers


class OneapiDetectionTests(unittest.TestCase):

    @patch('bindcraft.accelerator._local_devices', return_value=(SimpleNamespace(platform='oneapi'),))
    @patch('bindcraft.accelerator._oneapi_plugin_installed', return_value=True)
    def test_returns_oneapi_devices(self, _installed, _devices):
        self.assertEqual(_oneapi_devices()[0].platform, 'oneapi')

    @patch('bindcraft.accelerator._local_devices')
    @patch('bindcraft.accelerator._oneapi_plugin_installed', return_value=False)
    def test_does_not_start_jax_when_plugin_is_absent(self, _installed, devices):
        self.assertEqual(_oneapi_devices(), ())
        devices.assert_not_called()

    @patch('bindcraft.accelerator._local_devices', return_value=(SimpleNamespace(platform='cpu'),))
    @patch('bindcraft.accelerator._oneapi_plugin_installed', return_value=True)
    def test_refuses_cpu_fallback_when_oneapi_plugin_is_installed(self, _installed, _devices):
        with self.assertRaisesRegex(RuntimeError, 'LD_LIBRARY_PATH'):
            _oneapi_devices()

    @patch('bindcraft.accelerator._oneapi_devices', return_value=(SimpleNamespace(platform='oneapi'),))
    def test_disables_command_buffers_for_oneapi_jits(self, _devices):
        self.assertEqual(_oneapi_compiler_options(), {'xla_gpu_enable_command_buffer': ''})

    @patch('bindcraft.accelerator._oneapi_devices', return_value=())
    def test_leaves_non_oneapi_jit_options_unchanged(self, _devices):
        self.assertIsNone(_oneapi_compiler_options())


class OneapiWorkerTests(unittest.TestCase):

    @patch('bindcraft.design_workers.plan_design_workers')
    @patch('bindcraft.design_workers._oneapi_devices', return_value=(SimpleNamespace(platform='oneapi'),))
    def test_oneapi_uses_one_in_process_worker(self, _devices, plan_workers):
        self.assertIsNone(dispatch_design_workers({'auto_multi_gpu': True}, 'logs'))
        plan_workers.assert_not_called()

    @patch('bindcraft.design_workers._oneapi_devices', return_value=(SimpleNamespace(platform='oneapi'),))
    def test_rejects_nvidia_gpu_selection_and_worker_packing(self, _devices):
        for settings in (
            {'gpu_ids': '0'},
            {'workers_per_gpu': 2},
            {'design_workers': 2},
            {'attention_backend': 'cudnn'},
            {'use_cueq': True},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                dispatch_design_workers(settings, 'logs')

    @patch('bindcraft.design_workers.design_gpu_memory_gb')
    @patch('bindcraft.design_workers._oneapi_devices', return_value=(SimpleNamespace(platform='oneapi'),))
    def test_auto_subbatch_does_not_read_nvidia_memory(self, _devices, gpu_memory):
        self.assertEqual(campaign_subbatch_size({}, 400), 'auto')
        gpu_memory.assert_not_called()

    @patch('bindcraft.design_workers.launch_design_workers', return_value=0)
    @patch('bindcraft.design_workers.plan_design_workers', return_value=[{'gpu': '0'}, {'gpu': '1'}])
    @patch('bindcraft.design_workers._oneapi_devices', return_value=())
    def test_cuda_worker_plan_still_fans_out(self, _devices, plan_workers, launch_workers):
        self.assertEqual(dispatch_design_workers({'auto_multi_gpu': True}, 'logs', worker_command=['python']), 0)
        plan_workers.assert_called_once()
        launch_workers.assert_called_once()

    @patch('bindcraft.cli.subprocess.run')
    @patch('bindcraft.accelerator._oneapi_devices', return_value=(SimpleNamespace(device_kind='Intel Arc Pro B70'),))
    def test_compile_cache_uses_oneapi_device_kind(self, _devices, nvidia_smi):
        self.assertEqual(design_card_name(), 'Intel Arc Pro B70')
        nvidia_smi.assert_not_called()


class OneapiLossTests(unittest.TestCase):

    def test_best_contact_mean_compiles_at_campaign_length(self):
        import jax
        import jax.numpy as jnp

        from bindcraft.accelerator import _oneapi_compiler_options
        from bindcraft.loss import best_contact_mean

        devices = _oneapi_devices()
        if not devices:
            self.skipTest('requires an available oneAPI device')
        length = 160
        values = jnp.arange(length * length, dtype=jnp.float32).reshape(length, length)
        mask = jnp.broadcast_to(jnp.arange(length) % 4 != 0, values.shape)
        row_start = jnp.arange(length, dtype=jnp.float32) * length
        expected_results = (
            (1, (row_start + 1) / 1.0001),
            (2, (2 * row_start + 3) / 2.0001),
            (float('inf'), (120 * row_start + 9600) / 120.0001),
        )
        for contact_count, expected in expected_results:
            with self.subTest(contact_count=contact_count):
                operation = jax.jit(
                    lambda distances, valid: best_contact_mean(distances, contact_count, valid),
                    compiler_options=_oneapi_compiler_options(),
                )
                result = operation(values, mask)
                jax.block_until_ready(result)
                self.assertEqual(result.shape, (length,))
                self.assertTrue(bool(jnp.allclose(result, expected)))


class OneapiProteinFeaturesTests(unittest.TestCase):

    def test_nearest_neighbors_compile_at_campaign_length(self):
        import haiku as hk
        import jax
        import jax.numpy as jnp
        import numpy as np

        from bindcraft.accelerator import _oneapi_compiler_options
        from bindcraft.mpnn.modules import ProteinFeatures

        devices = _oneapi_devices()
        if not devices:
            self.skipTest('requires an available oneAPI device')
        length, neighbor_count = 160, 48
        positions = jnp.arange(length, dtype=jnp.float32)
        coordinates = jnp.stack((positions, jnp.zeros_like(positions), jnp.zeros_like(positions)), axis=-1)
        mask = jnp.ones((length,), dtype=jnp.float32)

        def nearest_neighbors(points, residue_mask):
            return ProteinFeatures(
                edge_features=1,
                node_features=1,
                top_k=neighbor_count,
            )._get_edge_idx(points, residue_mask)

        transformed = hk.transform(nearest_neighbors)
        params = transformed.init(jax.random.PRNGKey(0), coordinates, mask)
        operation = jax.jit(
            lambda points, residue_mask: transformed.apply(
                params,
                jax.random.PRNGKey(1),
                points,
                residue_mask,
            ),
            compiler_options=_oneapi_compiler_options(),
        )
        result = np.asarray(operation(coordinates, mask))
        expected = np.argsort(
            np.abs(np.arange(length)[:, None] - np.arange(length)[None, :]),
            axis=-1,
            kind='stable',
        )[:, :neighbor_count]
        self.assertEqual(result.shape, (length, neighbor_count))
        np.testing.assert_array_equal(result, expected)


if __name__ == '__main__':
    unittest.main()
