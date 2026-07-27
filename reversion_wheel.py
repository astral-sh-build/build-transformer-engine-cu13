# /// script
# requires-python = ">=3.12"
# ///
"""Give a Python-independent native wheel another exact package version."""

import argparse
import base64
import copy
import csv
import hashlib
import io
import zipfile
from pathlib import Path


def wheel_hash(content: bytes) -> str:
    """Return the PEP 427 SHA-256 RECORD hash for wheel content."""
    digest = hashlib.sha256(content).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def rewrite_wheel_version(source_path: Path, target_version: str) -> Path:
    """Copy a native wheel with consistent metadata, paths, and RECORD."""
    if source_path.suffix != ".whl":
        raise ValueError(f"Expected a wheel: {source_path}")

    filename_parts = source_path.name.removesuffix(".whl").split("-")
    if len(filename_parts) != 5:
        raise ValueError(f"Expected a five-part wheel filename: {source_path.name}")

    distribution, source_version, python_tag, abi_tag, platform_tag = filename_parts
    target_name = (
        f"{distribution}-{target_version}-{python_tag}-{abi_tag}-{platform_tag}.whl"
    )
    target_path = source_path.with_name(target_name)
    if target_path == source_path:
        print(target_path)
        return target_path

    source_dist_info = f"{distribution}-{source_version}.dist-info"
    target_dist_info = f"{distribution}-{target_version}.dist-info"
    source_data = f"{distribution}-{source_version}.data"
    target_data = f"{distribution}-{target_version}.data"
    source_metadata = f"{source_dist_info}/METADATA"
    source_record = f"{source_dist_info}/RECORD"
    target_record = f"{target_dist_info}/RECORD"
    temporary_path = target_path.with_suffix(".whl.tmp")

    with (
        zipfile.ZipFile(source_path, "r") as source,
        zipfile.ZipFile(temporary_path, "w") as target,
    ):
        record_info = source.getinfo(source_record)
        record_content = io.StringIO(newline="")
        record_writer = csv.writer(record_content, lineterminator="\n")

        for source_info in source.infolist():
            source_name = source_info.filename
            if source_name == source_record:
                continue

            target_name = source_name
            if source_name.startswith(f"{source_dist_info}/"):
                target_name = (
                    f"{target_dist_info}/{source_name[len(source_dist_info) + 1 :]}"
                )
            elif source_name.startswith(f"{source_data}/"):
                target_name = f"{target_data}/{source_name[len(source_data) + 1 :]}"

            target_info = copy.copy(source_info)
            target_info.filename = target_name
            content = source.read(source_info)

            if source_name == source_metadata:
                metadata = content.decode("utf-8")
                if f"\nVersion: {source_version}\n" not in f"\n{metadata}":
                    raise ValueError(
                        f"Wheel metadata does not contain {source_version}"
                    )
                content = metadata.replace(source_version, target_version).encode(
                    "utf-8"
                )

            target.writestr(
                target_info, content, compress_type=source_info.compress_type
            )
            if not source_info.is_dir():
                record_writer.writerow(
                    [target_name, wheel_hash(content), str(len(content))]
                )

        record_writer.writerow([target_record, "", ""])
        target_record_bytes = record_content.getvalue().encode("utf-8")
        target_record_info = copy.copy(record_info)
        target_record_info.filename = target_record
        target.writestr(
            target_record_info,
            target_record_bytes,
            compress_type=record_info.compress_type,
        )

    temporary_path.replace(target_path)
    print(target_path)
    return target_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a Python-independent native wheel with an exact new version."
    )
    parser.add_argument("wheel", type=Path, help="Source Python-independent wheel")
    parser.add_argument("version", help="Exact local version for the generated wheel")
    arguments = parser.parse_args()
    rewrite_wheel_version(arguments.wheel, arguments.version)


if __name__ == "__main__":
    main()
