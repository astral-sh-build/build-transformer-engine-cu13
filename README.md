# build-transformer-engine-cu13

Pre-built CUDA 13 core wheels for
[NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine), across
CUDA versions and CPU architectures.

## Installation

Following the PyTorch convention, artifacts are published to a separate index
for each CUDA version. Each architecture-specific CUDA 13 core wheel has a CUDA
local version and is shared across PyTorch and Python versions. For example,
`transformer_engine_cu13-2.16.0+cu.13.0-py3-none-manylinux_2_28_x86_64.whl`
provides the CUDA 13.0 core for every supported PyTorch version.

Once released, pre-built wheels will be available on
[Astral's GPU indexes](https://wheels.astral.sh/index.html).
For example, to install the Transformer Engine PyTorch extension and its
matching CUDA 13 core:

```console
$ uv add 'transformer-engine[pytorch]' --index astral-cu130=https://wheels.astral.sh/simple/cu130/
```

This configures the index and uses it as the source for the metapackage,
CUDA 13 core, and PyTorch extension:

```toml
[tool.uv.sources]
transformer-engine = { index = "astral-cu130" }
transformer-engine-cu13 = { index = "astral-cu130" }
transformer-engine-torch = { index = "astral-cu130" }

[[tool.uv.index]]
name = "astral-cu130"
url = "https://wheels.astral.sh/simple/cu130/"
```

Or, with `uv pip`:

```console
$ uv pip install --index https://wheels.astral.sh/simple/cu130/ 'transformer-engine[pytorch]'
```

The core and metapackage apply the same compatibility patch so NVIDIA's version
checks compare the public Transformer Engine version rather than the CUDA and
PyTorch local versions. The packages remain compatible regardless of
installation order.

## Supported versions

Wheels can be built for the following NVIDIA Transformer Engine version:

- [`2.16.0`](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.16)

The native CUDA core is built once per CUDA version and CPU architecture using
the corresponding pre-built PyTorch CUDA manylinux image. The resulting wheel
is independent of the PyTorch and Python versions.

## License

build-transformer-engine-cu13 is licensed under the
[Apache License, Version 2.0](LICENSE).

<div align="center">
  <a target="_blank" href="https://astral.sh" style="background:none">
    <img src="https://raw.githubusercontent.com/astral-sh/ruff/main/assets/svg/Astral.svg" alt="Made by Astral">
  </a>
</div>
