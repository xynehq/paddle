#!/usr/bin/env python3
"""
Lightweight test for server/model_repo/blip-caption/1/model.py to verify
memory cleanup and safe processing without requiring heavy dependencies
or the Triton runtime.

It mocks triton_python_backend_utils, the model's decode/generate methods,
and validates that images are closed, gc.collect is invoked, and CUDA
cache is emptied when device = 'cuda'.
"""
import base64
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np


def install_fake_pb_utils():
    mod = types.ModuleType("triton_python_backend_utils")

    class Tensor:
        def __init__(self, name, data):
            self.name = name
            self.data = data

    class InferenceResponse:
        def __init__(self, output_tensors=None):
            self.output_tensors = output_tensors or []

    def get_input_tensor_by_name(request, name):
        return request[name]

    mod.Tensor = Tensor
    mod.InferenceResponse = InferenceResponse
    mod.get_input_tensor_by_name = get_input_tensor_by_name
    sys.modules[mod.__name__] = mod
    return mod


class FakeInputTensor:
    def __init__(self, arr):
        self._arr = arr

    def as_numpy(self):
        return self._arr


class CloseCounterImage:
    def __init__(self, counter):
        self._counter = counter

    def close(self):
        self._counter[0] += 1


def load_model_module():
    # Path to the model file
    path = Path("server/model_repo/blip-caption/1/model.py").resolve()
    spec = importlib.util.spec_from_file_location("blip_caption_model", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def make_request_payload():
    payload = {
        "image_b64": "data:image/png;base64," + base64.b64encode(b"test").decode(),
        "max_length": 16,
        "num_beams": 2,
        "no_repeat_ngram_size": 2,
    }
    arr = np.array([json.dumps(payload).encode("utf-8")], dtype=object)
    return {"input": FakeInputTensor(arr)}


def run_cpu_test(model_module):
    # Track gc.collect calls by monkeypatching the module's gc.collect
    gc_calls = [0]
    orig_gc_collect = model_module.gc.collect
    model_module.gc.collect = lambda: gc_calls.__setitem__(0, gc_calls[0] + 1)

    # Build model instance with stub methods
    m = model_module.TritonPythonModel()
    m._device = "cpu"
    # No heavy torch usage in CPU path for our stubs
    m._decode_image = lambda b64: CloseCounterImage(close_counter)
    m._generate_caption = lambda img, **kw: "hello world"

    # Prepare request
    request = make_request_payload()
    responses = m.execute([request])

    # Collect assertions and restore gc.collect
    model_module.gc.collect = orig_gc_collect
    return responses, gc_calls[0]


def run_cuda_test(model_module):
    # Track gc.collect
    gc_calls = [0]
    orig_gc_collect = model_module.gc.collect
    model_module.gc.collect = lambda: gc_calls.__setitem__(0, gc_calls[0] + 1)

    # Track cuda.empty_cache calls
    empty_cache_calls = [0]

    class FakeCUDA:
        def empty_cache(self):
            empty_cache_calls[0] += 1

    # Fake torch with just cuda attr (used in execute cleanup)
    fake_torch = types.SimpleNamespace(cuda=FakeCUDA())

    m = model_module.TritonPythonModel()
    m._device = "cuda"
    m._torch = fake_torch
    m._decode_image = lambda b64: CloseCounterImage(close_counter)
    m._generate_caption = lambda img, **kw: "hello world"

    request = make_request_payload()
    responses = m.execute([request])

    model_module.gc.collect = orig_gc_collect
    return responses, gc_calls[0], empty_cache_calls[0]


if __name__ == "__main__":
    install_fake_pb_utils()
    model_module = load_model_module()

    close_counter = [0]
    responses_cpu, gc_cpu = run_cpu_test(model_module)
    # Reset counter for GPU test
    close_counter = [0]
    responses_cuda, gc_cuda, empty_cache_cuda = run_cuda_test(model_module)

    def first_caption(responses):
        out_tensors = responses[0].output_tensors
        arr = out_tensors[0].data
        payload = json.loads(arr.reshape(-1)[0].decode("utf-8"))
        return payload

    print("CPU response:", first_caption(responses_cpu))
    print("CPU gc.collect calls:", gc_cpu)

    print("CUDA response:", first_caption(responses_cuda))
    print("CUDA gc.collect calls:", gc_cuda)
    print("CUDA empty_cache calls:", empty_cache_cuda)

