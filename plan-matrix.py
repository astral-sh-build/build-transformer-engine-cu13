# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "packaging",
# ]
# ///

import json
import os

from packaging.version import Version

# Transformer Engine v2.16 supports Python 3.10+, PyTorch 2.1+, and CUDA 12+.
MIN_PYTHON_VERSION = "3.10"

TRANSFORMER_ENGINE_SUPPORTED_TORCH_VERSIONS = [
    "2.4.1",
    "2.5.1",
    "2.6.0",
    "2.7.1",
    "2.8.0",
    "2.9.1",
    "2.10.0",
    "2.11.0",
    "2.12.1",
]

ARCH_TORCH_PAIRS = {
    "x86_64": [
        "2.4.1",
        "2.5.1",
        "2.6.0",
        "2.7.1",
        "2.8.0",
        "2.9.1",
        "2.10.0",
        "2.11.0",
        "2.12.1",
    ],
    "aarch64": ["2.6.0", "2.7.1", "2.8.0", "2.9.1", "2.10.0", "2.11.0", "2.12.1"],
}

# Supported Python versions for each PyTorch version.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_PYTHON_SUPPORT = {
    "2.4": ["3.9", "3.10", "3.11", "3.12"],
    "2.5": ["3.9", "3.10", "3.11", "3.12"],
    "2.6": ["3.9", "3.10", "3.11", "3.12"],
    "2.7": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.8": ["3.9", "3.10", "3.11", "3.12", "3.13"],
    "2.9": ["3.10", "3.11", "3.12", "3.13"],
    "2.10": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "2.11": ["3.10", "3.11", "3.12", "3.13", "3.14"],
    "2.12": ["3.10", "3.11", "3.12", "3.13", "3.14"],
}

# Minimum and maximum CUDA versions for each PyTorch version.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
PYTORCH_CUDA_RANGES: dict[str, tuple[str, str]] = {
    "2.4": ("11.8", "12.4"),
    "2.5": ("11.8", "12.4"),
    "2.6": ("11.8", "12.6"),
    "2.7": ("11.8", "12.8"),
    "2.8": ("11.8", "12.9"),
    "2.9": ("12.6", "13.0"),
    "2.10": ("12.6", "13.0"),
    "2.11": ("12.6", "13.0"),
    "2.12": ("12.6", "13.2"),
}

# Actual CUDA versions to build against for each PyTorch version.
PYTORCH_CUDA_VERSIONS: dict[tuple[str, str], list[str]] = {
    ("2.4", "x86_64"): ["12.1", "12.4"],
    ("2.4", "aarch64"): ["12.4"],
    ("2.5", "x86_64"): ["12.1", "12.4"],
    ("2.5", "aarch64"): ["12.4"],
    ("2.6", "x86_64"): ["12.4", "12.6"],
    ("2.6", "aarch64"): ["12.6"],
    ("2.7", "x86_64"): ["12.6", "12.8"],
    ("2.7", "aarch64"): ["12.8"],
    ("2.8", "x86_64"): ["12.6", "12.8", "12.9"],
    ("2.8", "aarch64"): ["12.9"],
    ("2.9", "x86_64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.9", "aarch64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.10", "x86_64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.10", "aarch64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.11", "x86_64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.11", "aarch64"): ["12.6", "12.8", "12.9", "13.0"],
    ("2.12", "x86_64"): ["12.6", "13.0", "13.2"],
    ("2.12", "aarch64"): ["12.6", "13.0", "13.2"],
}

# CUDA architectures to build against for each PyTorch version.
TORCH_CUDA_ARCH_LIST = {
    # https://github.com/pytorch/pytorch/blob/d990dada86a8ad94882b5c23e859b88c0c255bda/torch/utils/cpp_extension.py#L1938
    ("2.4", "12.1"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.4", "12.4"): "7.0;7.5;8.0;8.6;9.0+PTX",
    # https://github.com/pytorch/pytorch/blob/32f585d9346e316e554c8d9bf7548af9f62141fc/torch/utils/cpp_extension.py#L1937
    ("2.5", "12.1"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.5", "12.4"): "7.0;7.5;8.0;8.6;9.0+PTX",
    # https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/utils/cpp_extension.py#L2030
    ("2.6", "12.4"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.6", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    # https://github.com/pytorch/pytorch/blob/134179474539648ba7dee1317959529fbd0e7f89/.ci/manywheel/build_cuda.sh#L55
    ("2.7", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.7", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    # https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/.ci/manywheel/build_cuda.sh#L56
    ("2.8", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.8", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.8", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    # https://github.com/pytorch/pytorch/blob/0fabc3ba44823f257e70ce397d989c8de5e362c1/.ci/manywheel/build_cuda.sh#L56
    ("2.9", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.9", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.9", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.9", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    ("2.10", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.10", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.10", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.10", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    ("2.11", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.11", "12.8"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.11", "12.9"): "7.0;7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.11", "13.0"): "7.5;8.0;8.6;9.0;10.0;11.0;12.0+PTX",
    # https://github.com/pytorch/pytorch/blob/main/RELEASE.md#pytorch-cuda-support-matrix
    ("2.12", "12.6"): "7.0;7.5;8.0;8.6;9.0+PTX",
    ("2.12", "13.0"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
    ("2.12", "13.2"): "7.5;8.0;8.6;9.0;10.0;12.0+PTX",
}

# The glibc version to use for each PyTorch version, for manylinux builds.
# See: https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix
TORCH_GLIBC_VERSION: dict[str, str] = {
    "2.4": "2_17",
    "2.5": "2_17",
    "2.6": "2_27",
    "2.7": "2_27",
    "2.8": "2_28",
    "2.9": "2_28",
    "2.10": "2_28",
    "2.11": "2_28",
    "2.12": "2_28",
}

AUDITWHEEL_BLANKET_EXCLUDES = [
    "libcuda.so",
    "libcuda.so.1",
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_python.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_cuda_cpp.so",
    "libtorch_cuda_cu.so",
    "libcufile_rdma.so",
    "libcufile_rdma.so.1",
    "libcufile.so.1",
    "libcufile.so.0",
    "libcufile.so",
    "libcurand.so.10",
]

AUDITWHEEL_CUDA_VERSION_EXCLUDES = {
    "11": [
        "libcudart.so.11",
        "libcudart.so.11.0",
    ],
    "12": [
        "libcudart.so.12",
        "libcudart.so.12.0",
    ],
    "13": [
        "libcudart.so.13",
        "libcudart.so.13.0",
    ],
}

# Matrix exclusions.
EXCLUSIONS = [
    # No exclusions yet.
]


def main() -> None:
    # Every matrix member is a primary 5-tuple of:
    # `torch-version`: the PyTorch version as "X.Y.Z", e.g. "2.7.0"
    # `python-version`: the Python version as "3.X", e.g. "3.10"
    # `cuda-version`: the CUDA version as "X.Y.Z", e.g. "11.8.0"
    # `cxx11-abi`: "TRUE" or "FALSE"
    # `target-arch`: the target architecture, e.g. "x86_64" or "aarch64"

    rows = []
    for target_arch, torch_versions in ARCH_TORCH_PAIRS.items():
        for torch_version in torch_versions:
            if torch_version not in TRANSFORMER_ENGINE_SUPPORTED_TORCH_VERSIONS:
                continue

            torch_version_parsed = Version(torch_version)
            torch_x_y = f"{torch_version_parsed.major}.{torch_version_parsed.minor}"
            for python_version in TORCH_PYTHON_SUPPORT[torch_x_y]:
                python_version_parsed = Version(python_version)
                if python_version_parsed < Version(MIN_PYTHON_VERSION):
                    continue

                cuda_versions = PYTORCH_CUDA_VERSIONS[(torch_x_y, target_arch)]
                for cuda_version in cuda_versions:
                    cuda_version_parsed = Version(cuda_version)

                    # The CXX11 ABI became the default in PyTorch 2.7.0, but was also used in
                    # PyTorch 2.6.0 (but _only_ for the CUDA 12.6 builds).
                    #
                    # See: https://pytorch.org/blog/pytorch2-6/
                    cxx11_abi = torch_version_parsed >= Version("2.7.0") or (
                        torch_version_parsed == Version("2.6.0")
                        and cuda_version_parsed >= Version("12.6")
                    )

                    row = {
                        "target-arch": target_arch,
                        "torch-version": str(torch_version_parsed),
                        "python-version": python_version,
                        "cuda-version": cuda_version,
                        "cxx11-abi": "TRUE" if cxx11_abi else "FALSE",
                    }

                    if row not in EXCLUSIONS:
                        rows.append(row)

    # Transform each row to add various nice-to-have representations of fields.
    for row in rows:
        # `CI_*` variables: same as the original ones.
        row["CI_CUDA_VERSION"] = row["cuda-version"]
        row["CI_TORCH_VERSION"] = row["torch-version"]
        row["CI_PYTHON_VERSION"] = row["python-version"]

        # `MATRIX_CUDA_VERSION`: XY instead of X.Y
        cuda_version = Version(row["cuda-version"])
        row["MATRIX_CUDA_VERSION"] = f"{cuda_version.major}{cuda_version.minor}"

        # `MATRIX_TORCH_VERSION`: `torch-version`, but only X.Y, no patch
        torch_version = Version(row["torch-version"])
        row["MATRIX_TORCH_VERSION"] = f"{torch_version.major}.{torch_version.minor}"

        # `MATRIX_PYTHON_VERSION`: same as `python-version`, but with the dot removed
        row["MATRIX_PYTHON_VERSION"] = row["python-version"].replace(".", "")

        # `MANYLINUX_CUDA_VERSION`: X.Y instead of X.Y.Z
        row["MANYLINUX_CUDA_VERSION"] = f"{cuda_version.major}.{cuda_version.minor}"

        # `MANYLINUX_CUDA_COMPAT_VERSION`: X-Y instead of X.Y.Z
        row["MANYLINUX_CUDA_COMPAT_VERSION"] = (
            f"{cuda_version.major}-{cuda_version.minor}"
        )

        # MANYLINUX_GLIBC_VERSION: the glibc version to use for manylinux builds.
        row["MANYLINUX_GLIBC_VERSION"] = TORCH_GLIBC_VERSION[
            row["MATRIX_TORCH_VERSION"]
        ]

        # `CI_AUDITWHEEL_EXCLUDES`: `--exclude {lib}` for each lib that should
        # be excluded when running `auditwheel repair`.
        cuda_major = str(cuda_version.major)
        auditwheel_excludes = (
            AUDITWHEEL_BLANKET_EXCLUDES
            + AUDITWHEEL_CUDA_VERSION_EXCLUDES.get(cuda_major, [])
        )
        row["CI_AUDITWHEEL_EXCLUDES"] = " ".join(
            f"--exclude {lib}" for lib in auditwheel_excludes
        )

        row["TORCH_CUDA_ARCH_LIST"] = TORCH_CUDA_ARCH_LIST[
            (
                f"{torch_version.major}.{torch_version.minor}",
                f"{cuda_version.major}.{cuda_version.minor}",
            )
        ]

        # RUNNER: the GitHub Actions runner to use.
        if row["target-arch"] == "x86_64":
            row["RUNNER"] = "depot-ubuntu-24.04"
        elif row["target-arch"] == "aarch64":
            row["RUNNER"] = "depot-ubuntu-24.04-arm"
        else:
            raise ValueError(f"Unknown target arch: {row['target-arch']}")

        # `NVTE_CUDA_ARCHS`: the CUDA architectures to build against for each PyTorch version.
        # Like `TORCH_CUDA_ARCH_LIST`, but without the major-minor split (e.g., `75` instead of `7.5`).
        nvte_cuda_archs = []
        for arch in row["TORCH_CUDA_ARCH_LIST"].split(";"):
            arch = arch.removesuffix("+PTX")
            major, minor = arch.split(".")
            nvte_cuda_archs.append(f"{major}{minor}")
        row["NVTE_CUDA_ARCHS"] = ";".join(nvte_cuda_archs)

    # The native core is Torch-independent. Compile once per CUDA image and
    # CPU architecture, then emit a matching wheel for each Torch version.
    grouped_rows = {}
    for row in rows:
        if not row["MANYLINUX_CUDA_VERSION"].startswith("13."):
            continue

        key = (row["MANYLINUX_CUDA_VERSION"], row["target-arch"])
        if key not in grouped_rows:
            grouped_rows[key] = {
                **row,
                "MATRIX_TORCH_VERSIONS": [row["MATRIX_TORCH_VERSION"]],
            }
            continue

        group = grouped_rows[key]
        torch_version = row["MATRIX_TORCH_VERSION"]
        if torch_version not in group["MATRIX_TORCH_VERSIONS"]:
            group["MATRIX_TORCH_VERSIONS"].append(torch_version)

        cuda_architectures = group["NVTE_CUDA_ARCHS"].split(";")
        for cuda_architecture in row["NVTE_CUDA_ARCHS"].split(";"):
            if cuda_architecture not in cuda_architectures:
                cuda_architectures.append(cuda_architecture)
        group["NVTE_CUDA_ARCHS"] = ";".join(cuda_architectures)

    rows = list(grouped_rows.values())

    # For pull requests, compile one image and emit all of its Torch versions.
    if os.environ.get("LIMIT_MATRIX") == "1":
        smoke_cuda_version = "13.0"
        rows = [
            next(
                row
                for row in rows
                if row["MANYLINUX_CUDA_VERSION"] == smoke_cuda_version
                and row["target-arch"] == "x86_64"
                and "2.10" in row["MATRIX_TORCH_VERSIONS"]
            )
        ]

    print(json.dumps(rows))


if __name__ == "__main__":
    main()
