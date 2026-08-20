"""Source-of-truth contract for the fused-slab native extension build.

This module is deliberately standard-library-only.  ``setup.py`` imports it
before invoking Torch's build machinery, and the repository verifiers import
the same contract without compiling or dispatching Metal.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


CONTRACT_SCHEMA_VERSION = 1
VARIANT_NAME = "world_foam_lane2_fused_slab_v0"
PACKAGE_NAME = "torch_world_foam_lane2_fused_slab"
EXTENSION_NAME = f"{PACKAGE_NAME}._C"

TRANSLATION_UNITS = (
    "csrc/bindings.cpp",
    "csrc/metal/world_foam_lane2_metal.mm",
)
RUNTIME_METAL_SOURCES = (
    "csrc/metal/world_foam_lane2_power_boundary_tensor.metal",
    "csrc/metal/world_foam_lane2_shared_replay_tensor.metal",
)
NATIVE_DEPENDENCIES = (
    "setup.py",
    "native_build_contract.py",
    "csrc/shared/world_foam_lane2_types.h",
    *RUNTIME_METAL_SOURCES,
)
PYTHON_ABI_SOURCES = (
    f"{PACKAGE_NAME}/__init__.py",
    f"{PACKAGE_NAME}/ops.py",
    f"{PACKAGE_NAME}/certificate_binding.py",
)
ATTESTED_SOURCE_FILES = tuple(
    dict.fromkeys((*TRANSLATION_UNITS, *NATIVE_DEPENDENCIES, *PYTHON_ABI_SOURCES))
)

# These are the thirty schemas added after the currently retained 103-schema
# binary was built.  They are named explicitly so a future edit cannot update a
# count/digest while accidentally dropping the memory-light and full-geometry
# ABI that motivated this rebuild.
REQUIRED_POST_103_SCHEMA_NAMES = (
    "endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_mse_vjp_direct_atomic_rgb_boundary",
    "endpoint_record_delta_replace_factorized_packed_framegroup16_constant_state_p0_mse_vjp_sparse_mobius_rgb_boundary",
    "fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary",
    "fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary_launch_only",
    "fixed_word_p0_lie_material_node_vjp_accumulate_launch_only",
    "fixed_word_p0_lie_material_world_grad_init_launch_only",
    "fixed_word_p0_lie_node_forward_launch_only",
    "fixed_word_p0_lie_node_vjp_accumulate_launch_only",
    "fixed_word_p0_lie_sample_accumulate_launch_only",
    "fixed_word_p0_lie_sample_accumulate_loss_only_launch_only",
    "fixed_word_p0_lie_sample_state_init_launch_only",
    "fixed_word_p0_lie_world_grad_init_launch_only",
    "fixed_word_p0_sparse_mobius_boundary_finalize_launch_only",
    "fixed_word_p0_sparse_mobius_lower_launch_only",
    "kinetic_fused_direct_full_vjp_accumulate_launch_only_v1",
    "kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1",
    "kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1",
    "kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1",
    "kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2",
    "kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2",
    "kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2",
    "kinetic_memory_light_selected_kernel_resource_attestation",
    "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
    "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1",
    "kinetic_precompiled_length_p0_lie_node_forward_launch_only",
    "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only",
    "kinetic_ragged_p0_lie_sample_accumulate_launch_only",
    "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only",
    "sparse_power_boundary_from_sites_launch_only",
    "sparse_power_boundary_vjp_to_sites_launch_only",
)

EXPECTED_SCHEMA_COUNT = 133
EXPECTED_SCHEMA_NAME_INVENTORY_SHA256 = (
    "818d42fd3c45c89cc55fb886f16be0d7a6a9479ba66867bdac3dc77fe4a810d8"
)
EXPECTED_FULL_SCHEMA_INVENTORY_SHA256 = (
    "4296969b4943bf685d3e4e7fec5a211c5a2f85dff5f07d71821c4252c5f91168"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inventory_sha256(values: list[str] | tuple[str, ...] | set[str]) -> str:
    return _sha256(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _schema_strings(bindings_source: str) -> list[str]:
    return re.findall(r'm\.def\(\s*"([^"]+)"', bindings_source, flags=re.DOTALL)


def _schema_names(schema_strings: list[str]) -> list[str]:
    return [schema.split("(", 1)[0].strip() for schema in schema_strings]


def _impl_names(bindings_source: str) -> list[str]:
    return re.findall(r'm\.impl\(\s*"([^"]+)"', bindings_source, flags=re.DOTALL)


def _duplicates(values: list[str]) -> list[str]:
    return sorted(value for value in set(values) if values.count(value) > 1)


def validate_source_contract(variant_dir: Path | None = None) -> dict[str, Any]:
    """Validate the exact source inventory consumed by a clean native build."""

    root = Path(__file__).resolve().parent if variant_dir is None else Path(variant_dir).resolve()
    failures: list[str] = []
    missing_files = [relative for relative in ATTESTED_SOURCE_FILES if not (root / relative).is_file()]
    if missing_files:
        failures.append(f"missing attested source files: {missing_files}")

    bindings_path = root / TRANSLATION_UNITS[0]
    host_path = root / TRANSLATION_UNITS[1]
    if failures or not bindings_path.is_file() or not host_path.is_file():
        raise RuntimeError("; ".join(failures or ["native translation units are missing"]))

    bindings_source = bindings_path.read_text(encoding="utf-8")
    host_source = host_path.read_text(encoding="utf-8")
    schema_strings = _schema_strings(bindings_source)
    schema_names = _schema_names(schema_strings)
    impl_names = _impl_names(bindings_source)

    duplicate_schemas = _duplicates(schema_names)
    duplicate_impls = _duplicates(impl_names)
    if duplicate_schemas:
        failures.append(f"duplicate TORCH_LIBRARY schemas: {duplicate_schemas}")
    if duplicate_impls:
        failures.append(f"duplicate TORCH_LIBRARY_IMPL names: {duplicate_impls}")
    if set(schema_names) != set(impl_names):
        failures.append(
            "schema/implementation inventory mismatch: "
            f"schemas_without_impl={sorted(set(schema_names) - set(impl_names))}, "
            f"impls_without_schema={sorted(set(impl_names) - set(schema_names))}"
        )

    required_missing = sorted(set(REQUIRED_POST_103_SCHEMA_NAMES) - set(schema_names))
    if required_missing:
        failures.append(f"required post-103 schemas are missing: {required_missing}")
    if len(schema_names) != EXPECTED_SCHEMA_COUNT:
        failures.append(
            f"schema count changed: expected {EXPECTED_SCHEMA_COUNT}, found {len(schema_names)}"
        )

    name_digest = _inventory_sha256(schema_names)
    full_digest = _inventory_sha256(schema_strings)
    if name_digest != EXPECTED_SCHEMA_NAME_INVENTORY_SHA256:
        failures.append(
            "schema-name inventory digest changed: "
            f"expected {EXPECTED_SCHEMA_NAME_INVENTORY_SHA256}, found {name_digest}"
        )
    if full_digest != EXPECTED_FULL_SCHEMA_INVENTORY_SHA256:
        failures.append(
            "full-schema inventory digest changed: "
            f"expected {EXPECTED_FULL_SCHEMA_INVENTORY_SHA256}, found {full_digest}"
        )

    declared_runtime_metal = {
        f"csrc/metal/{filename}"
        for filename in re.findall(
            r'stringByAppendingPathComponent:@"([^"]+\.metal)"', host_source
        )
    }
    if declared_runtime_metal != set(RUNTIME_METAL_SOURCES):
        failures.append(
            "runtime Metal source inventory mismatch: "
            f"contract={sorted(RUNTIME_METAL_SOURCES)}, "
            f"host={sorted(declared_runtime_metal)}"
        )

    for macro in ("TORCH_LIBRARY", "TORCH_LIBRARY_IMPL"):
        if not re.search(rf"{macro}\(\s*{re.escape(VARIANT_NAME)}\s*,", bindings_source):
            failures.append(f"bindings omit {macro} namespace {VARIANT_NAME}")

    if failures:
        raise RuntimeError("; ".join(failures))

    return {
        "contract_schema_version": CONTRACT_SCHEMA_VERSION,
        "variant": VARIANT_NAME,
        "package": PACKAGE_NAME,
        "extension": EXTENSION_NAME,
        "translation_units": list(TRANSLATION_UNITS),
        "native_dependencies": list(NATIVE_DEPENDENCIES),
        "runtime_metal_sources": list(RUNTIME_METAL_SOURCES),
        "python_abi_sources": list(PYTHON_ABI_SOURCES),
        "attested_source_files": list(ATTESTED_SOURCE_FILES),
        "schema_count": len(schema_names),
        "schema_names": sorted(schema_names),
        "schema_name_inventory_sha256": name_digest,
        "full_schema_inventory_sha256": full_digest,
        "required_post_103_schema_count": len(REQUIRED_POST_103_SCHEMA_NAMES),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(validate_source_contract(), indent=2, sort_keys=True))
