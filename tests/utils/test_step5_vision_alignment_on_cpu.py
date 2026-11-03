"""Stdlib-only AST contract tests; imports no model package and uses no device."""

import ast
from pathlib import Path
import types
import unittest


SOURCE = Path(__file__).resolve().parents[2] / "verl/models/transformers/step5.py"


def load_alignment(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "align_step5_vision_calls")
    namespace = {"types": types}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["align_step5_vision_calls"]


class Tensor:
    def __init__(self, values=(1.0, -2.0, 3.0), shape=(1, 3, 56, 56), dtype="bf16"):
        self.values, self.shape, self.dtype = tuple(values), shape, dtype

    def new_zeros(self, shape):
        return Tensor((0.0,), shape, self.dtype)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def float(self):
        return Tensor(self.values, self.shape, "fp32")

    def sum(self):
        return Tensor((sum(self.values),), (), self.dtype)

    def to(self, dtype):
        return Tensor(self.values, self.shape, dtype)

    def __mul__(self, value):
        return Tensor(tuple(item * value for item in self.values), self.shape, self.dtype)

    def __add__(self, other):
        assert len(other.values) == 1
        return Tensor(tuple(item + other.values[0] for item in self.values), self.shape, self.dtype)


class Module:
    def __init__(self, trainable, patch_size=14):
        self.items = [types.SimpleNamespace(requires_grad=trainable)]
        self.patch_size = patch_size

    def parameters(self):
        return iter(self.items)


class Model:
    def __init__(self, flags=(True, True), trainable=(False, False), patch_size=14):
        self.config = types.SimpleNamespace()
        if flags is not None:
            self.config.freeze_vision_encoder, self.config.freeze_vision_projector = flags
        self.vision_model = Module(trainable[0], patch_size)
        self.vit_large_projector = Module(trainable[1])
        self.calls = []
        self.real_features = Tensor()

    def get_multimodal_embeddings(self, *args):
        self.calls.append(("original", args))
        return self.real_features

    def _encode_images(self, pixels):
        self.calls.append(("dummy", pixels.shape))
        return Tensor((0.25, -0.5), (1, 1, 3))


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate = staticmethod(load_alignment(SOURCE))

    def check(self, flags=(True, True), trainable=(False, False), side=56,
              patch=None, pixels=True, embeds=None, patch_size=14):
        after = Model(flags, trainable, patch_size)
        self.candidate(after)
        arguments = (Tensor() if pixels else None, patch, None, embeds)
        expected = after.real_features
        actual = after.get_multimodal_embeddings(*arguments)
        self.assertEqual((expected.values, expected.shape, expected.dtype),
                         (actual.values, actual.shape, actual.dtype))
        self.assertEqual(after.calls[0], ("original", arguments))
        dummy_calls = [item for item in after.calls if item[0] == "dummy"]
        self.assertEqual(dummy_calls, [] if side is None else [("dummy", (1, 3, side, side))])
        self.assertEqual(len(after.calls), 1 + len(dummy_calls))

    def test_explicit_frozen_pair_uses_four_patch_grid(self):
        for patch_size in (14, 16):
            with self.subTest(patch_size=patch_size):
                self.check(side=4 * patch_size, patch_size=patch_size)

    def test_every_other_trainability_configuration_retains_504(self):
        for flags in (None, (False, False), (True, False), (False, True)):
            with self.subTest(flags=flags):
                self.check(flags=flags, side=504)
        for trainable in ((True, False), (False, True), (True, True)):
            with self.subTest(trainable=trainable):
                self.check(trainable=trainable, side=504)

    def test_real_crops_skip_dummy(self):
        self.check(patch=Tensor(shape=(2, 3, 28, 28)), side=None)

    def test_empty_crops_receive_tiny_dummy(self):
        self.check(patch=Tensor(shape=(0, 3, 28, 28)))

    def test_supplied_features_and_missing_pixels_skip_dummy(self):
        self.check(embeds=Tensor(), side=None)
        self.check(pixels=False, side=None)

    def test_alignment_installation_is_idempotent(self):
        model = Model()
        self.candidate(model)
        installed = model.get_multimodal_embeddings
        self.candidate(model)
        self.assertIs(model.get_multimodal_embeddings, installed)
        model.get_multimodal_embeddings(Tensor())
        self.assertEqual(len(model.calls), 2)


if __name__ == "__main__":
    unittest.main()
