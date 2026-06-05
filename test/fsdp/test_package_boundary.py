import importlib
from pathlib import Path
import tomllib
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class PackageBoundaryTest(unittest.TestCase):
    def test_legacy_module_import_paths_still_resolve(self):
        module_symbols = {
            "matrix_fsdp.checkpoint": "save_matrix_dcp",
            "matrix_fsdp.flat_buffer": "MatrixFlatBuffer",
            "matrix_fsdp.layout": "MatrixGroupLayout",
            "matrix_fsdp.optim_state": "MatrixFSDPOptimizerStateManager",
            "matrix_fsdp.optim": "prepare_matrix_optimizer",
            "matrix_fsdp.param_group": "MatrixFSDPParamGroup",
            "matrix_fsdp.placement": "MatrixShard",
            "matrix_fsdp.planner": "contiguous_even_plan",
            "matrix_fsdp.wrap": "module_type_policy",
        }

        for module_name, symbol_name in module_symbols.items():
            with self.subTest(module_name=module_name, symbol_name=symbol_name):
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, symbol_name))

    def test_param_group_canonical_helpers_are_public(self):
        matrix_fsdp = importlib.import_module("matrix_fsdp")

        self.assertTrue(hasattr(matrix_fsdp, "collect_param_groups"))
        self.assertTrue(hasattr(matrix_fsdp, "summarize_param_groups"))
        self.assertTrue(hasattr(matrix_fsdp, "MatrixFSDPParamGroup"))

    def test_pyproject_declares_installable_package_metadata(self):
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

        self.assertEqual(pyproject["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(pyproject["project"]["name"], "matrix-fsdp")
        self.assertEqual(pyproject["project"]["version"], importlib.import_module("matrix_fsdp").__version__)
        self.assertIn("torch>=2.10", pyproject["project"]["dependencies"])
        self.assertEqual(
            pyproject["tool"]["setuptools"]["packages"]["find"]["include"],
            ["matrix_fsdp", "matrix_fsdp.*"],
        )
        package_data = pyproject["tool"]["setuptools"]["package-data"]["matrix_fsdp"]
        self.assertIn("kernels/native/*.cpp", package_data)
        self.assertIn("kernels/native/*.cu", package_data)

    def test_usage_docs_are_present(self):
        docs_dir = REPO_ROOT / "docs"
        usage_docs = {"introduction.md", "tutorial.md"}

        self.assertEqual({path.name for path in docs_dir.glob("*.md")}, usage_docs)
        self.assertIn("MatrixFSDP", (REPO_ROOT / "README.md").read_text())
        introduction_doc = (docs_dir / "introduction.md").read_text()
        tutorial_doc = (docs_dir / "tutorial.md").read_text()
        self.assertIn("fully_shard", introduction_doc)
        self.assertNotIn("matrix_fully_shard", introduction_doc)
        self.assertNotIn("matrix_fully_shard", tutorial_doc)


if __name__ == "__main__":
    unittest.main()
