import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import onnx

from holosoma_inference.motion_paths import MOTIONS_DIR, resolve_motion_path
from holosoma_inference.picking_scene import PickingReference


class MotionPathTests(unittest.TestCase):
    def test_bundled_reference_takes_priority(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            bundled = root / "motions"
            bundled.mkdir()
            original = root / "reference.npz"
            original.touch()
            local = bundled / original.name
            local.touch()
            with patch("holosoma_inference.motion_paths.MOTIONS_DIR", bundled):
                self.assertEqual(resolve_motion_path(original), local)

    def test_original_path_fallback(self):
        with TemporaryDirectory() as folder:
            original = Path(folder) / "custom_reference.npz"
            original.touch()
            with patch("holosoma_inference.motion_paths.MOTIONS_DIR", Path(folder) / "missing"):
                self.assertEqual(resolve_motion_path(original), original)

    def test_missing_reference_reports_both_paths(self):
        with TemporaryDirectory() as folder:
            original = Path(folder) / "missing.npz"
            bundled = Path(folder) / "motions"
            with patch("holosoma_inference.motion_paths.MOTIONS_DIR", bundled):
                with self.assertRaises(FileNotFoundError) as error:
                    resolve_motion_path(original)
            self.assertIn(str(original), str(error.exception))
            self.assertIn(str(bundled / original.name), str(error.exception))

    def test_all_models_load_with_unavailable_training_paths(self):
        model_dir = MOTIONS_DIR.parent / "onnx"
        for name, frames in (("pick_17000.onnx", 296), ("shoot_82000.onnx", 180),
                             ("recover_39999.onnx", 79)):
            with self.subTest(model=name):
                model = onnx.load(str(model_dir / name), load_external_data=False)
                config = json.loads(next(p.value for p in model.metadata_props if p.key == "experiment_config"))
                metadata_path = config["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"]
                unavailable = Path("/unavailable-training-machine") / Path(metadata_path).name
                resolved = resolve_motion_path(unavailable)
                self.assertEqual(resolved.parent, MOTIONS_DIR)
                with np.load(resolved, allow_pickle=False) as data:
                    self.assertEqual(len(data["joint_pos"]), frames)
                if name == "pick_17000.onnx":
                    config["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"] = str(unavailable)
                    for prop in model.metadata_props:
                        if prop.key == "experiment_config":
                            prop.value = json.dumps(config)
                    with patch("holosoma_inference.picking_scene.onnx.load", return_value=model):
                        reference = PickingReference(model_dir / name)
                    self.assertEqual(reference.joint_pos.shape, (29,))


if __name__ == "__main__":
    unittest.main()
