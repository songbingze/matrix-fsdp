import tempfile
import unittest
from pathlib import Path
from unittest import mock

from matrix_fsdp.kernels.gin.probe import probe_gin_device_api


class GinDeviceApiProbeTest(unittest.TestCase):
    def test_probe_reports_old_nccl_runtime_before_header_readiness(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.nccl.version",
            return_value=(2, 27, 5),
        ):
            include_dir = Path(tmpdir)
            (include_dir / "nccl.h").write_text("typedef struct ncclDevComm ncclDevComm;")

            probe = probe_gin_device_api(extra_include_dirs=(tmpdir,))

        self.assertEqual(probe.torch_nccl_version, (2, 27, 5))
        self.assertTrue(probe.header_has_device_api)
        self.assertFalse(probe.device_api_version_ready)
        self.assertFalse(probe.device_api_ready)
        self.assertIn("older", probe.reason)

    def test_probe_reports_gin_ready_when_runtime_and_headers_match(self):
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.nccl.version",
            return_value=(2, 30, 4),
        ):
            include_dir = Path(tmpdir)
            (include_dir / "nccl.h").write_text(
                "\n".join(
                    (
                        "typedef struct ncclDevComm ncclDevComm;",
                        "ncclResult_t ncclDevCommCreate();",
                        "typedef struct ncclGin ncclGin;",
                        "#define NCCL_GIN_CONNECTION_FULL 1",
                    )
                )
            )

            probe = probe_gin_device_api(extra_include_dirs=(tmpdir,))

        self.assertTrue(probe.device_api_ready)
        self.assertTrue(probe.gin_ready)
        self.assertIn("visible", probe.reason)


if __name__ == "__main__":
    unittest.main()
