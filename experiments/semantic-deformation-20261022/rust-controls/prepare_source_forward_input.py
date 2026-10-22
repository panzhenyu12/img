"""Prepare an independent source-forward input with the Rust target controller.

This utility deliberately stops at the solver boundary.  It takes a frozen
whole-width input at the same signed amount, verifies the source photo and the user
source audit, replaces only the first 128 face destinations with targets from
the Rust controller, and rebuilds the objective right hand side/Hessian.  All
material supports, weights, locks, line files and source coordinates are
copied byte-for-byte from the parent input.  No image is warped and no solver
is invoked.

Example (from repository root)::

  services/background-worker/.venv/bin/python \
    experiments/semantic-deformation-20261022/rust-controls/prepare_source_forward_input.py \
    --output test/results/semantic-deformation-20261022/user-plus30-rust-input \
    --parent test/results/face-source-forward-photo-20261021/whole-width-signed-direct-v1/+30/input

The output is an independent experiment input.  It is not a quality or
beauty approval and must be consumed by a constraint-aware downstream solver.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import diags, load_npz, save_npz


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
from input_contract import require_same_signed_amount, validate_input
DEFAULT_PARENT = ROOT / "test/results/face-source-forward-photo-20261021/whole-width-signed-direct-v1/+30/input"
DEFAULT_AUDIT = ROOT / "test/results/semantic-deformation-20261022/controller-source-audit/user-audit.json"
DEFAULT_BINARY = HERE / "target/debug/face-width-targets"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def invoke(binary: Path, request: dict) -> dict:
    process = subprocess.run(
        [str(binary)], input=json.dumps(request), text=True,
        capture_output=True, check=False,
    )
    if process.returncode:
        raise RuntimeError(f"Rust controller failed ({process.returncode}): {process.stderr}")
    return json.loads(process.stdout)


def validate_parent(parent: Path, audit_path: Path, data: dict, contract: dict, audit: dict) -> dict:
    """Check source/frame/coordinate provenance before any output is written."""
    if not parent.is_dir():
        raise ValueError(f"parent input directory does not exist: {parent}")
    if contract.get("source_photo_sha256") != audit["input_sha256"][audit["source_image"]]:
        raise ValueError("parent and user source audit reference different source-photo hashes")
    source_photo = Path(contract["source_photo"])
    if sha(source_photo) != contract["source_photo_sha256"]:
        raise ValueError(f"frozen source photo changed: {source_photo}")
    # The audit intentionally hashes its evidence inputs, not itself.  When a
    # self hash is present (for a future signed audit format), verify it too.
    recorded = next((v for k, v in audit.get("input_sha256", {}).items()
                     if Path(k).resolve() == audit_path.resolve()), None)
    if recorded is not None and recorded != sha(audit_path):
        raise ValueError("user source audit hash is not the frozen audit consumed")
    source = np.asarray(data["target_sources"], dtype=np.float64)
    if source.shape[0] < 128 or source.shape[1] != 2:
        raise ValueError("parent must contain at least 128 two-dimensional targets")
    audit_source = np.asarray(audit["source_contour"], dtype=np.float64)
    if audit_source.shape != (128, 2) or not np.array_equal(source[:128], audit_source):
        raise ValueError("parent first 128 source coordinates differ from the audited user contour")
    frame = audit.get("frame_raw")
    if not isinstance(frame, dict) or set(("origin", "across", "down", "bounds")) - set(frame):
        raise ValueError("user source audit has no complete source frame")
    for key in ("origin", "across", "down", "bounds"):
        if not np.isfinite(np.asarray(frame[key], float)).all():
            raise ValueError(f"non-finite source frame field: {key}")
    if float(audit.get("midline_u", 0.0)) != 0.0:
        raise ValueError("this experiment requires the explicitly audited source midline_u=0")
    k = float(audit["calibration_k_full"])
    if not np.isfinite(k) or not 0.0 <= k <= 0.45:
        raise ValueError("audited Rust calibration k is outside the controller contract")
    # Every original source/destination delta must be internally consistent.
    destinations = np.asarray(data["target_destinations"], dtype=np.float64)
    deltas = np.asarray(data["target_deltas"], dtype=np.float64)
    if destinations.shape != source.shape or deltas.shape != source.shape:
        raise ValueError("parent target arrays have inconsistent shapes")
    if not np.array_equal(destinations - source, deltas):
        raise ValueError("parent target destination/delta ledger is inconsistent")
    return {"source_photo": source_photo, "source": source, "frame": frame, "k": k}


def controller_request(frame: dict, source: np.ndarray, amount: float, k: float) -> dict:
    return {
        "frame": {**frame, "midline_u": 0.0},
        "calibration": {"max_radius_fraction": k, "chin_fade_start": 0.88},
        "amount": float(amount),
        "contour": [{"id": f"source-{i}", "source": p.tolist()} for i, p in enumerate(source)],
        "parts": [], "relations": [],
    }


def rebuild_quadratic(out: Path, targets: np.ndarray) -> dict:
    """Rebuild H and rhs from the copied, explicit assembly blocks."""
    assembly = read(out / "assembly.json")
    nvars = int(assembly["nvars"])
    target = load_npz(out / "target.npz").tocsr()
    if target.shape != (len(targets), nvars):
        raise ValueError("target operator shape changed while preparing controller input")
    weights = np.load(out / "target_weights.npy")
    deltas = np.load(out / "target_deltas.npy")
    if len(weights) != len(targets) or deltas.shape != targets.shape:
        raise ValueError("target arrays do not match target operator")
    hessian = target.T @ diags(weights) @ target
    rhs = target.T @ (weights[:, None] * deltas)
    for block in assembly["blocks"][1:]:
        matrix = load_npz(out / block["matrix_file"]).tocsr()
        raw_weight = block.get("weights")
        if raw_weight is not None:
            hessian = hessian + float(raw_weight) * (matrix.T @ matrix)
        else:
            block_weights = np.load(out / block["weights_file"])
            hessian = hessian + matrix.T @ diags(block_weights) @ matrix
            prior = np.load(out / block["rhs_file"])
            rhs = rhs + matrix.T @ (block_weights[:, None] * prior)
    hessian = hessian.tocsc()
    # The target operator is unchanged.  Preserve the parent's byte-frozen
    # Hessian so the input contract can prove that only the target RHS moved;
    # a fresh sparse accumulation can differ in harmless floating-point sums.
    frozen_hessian = load_npz(out / "proposal/hessian.npz").tocsc()
    if frozen_hessian.shape != hessian.shape:
        raise ValueError("target rewrite changed Hessian dimensions")
    delta = hessian - frozen_hessian
    if delta.nnz and float(np.max(np.abs(delta.data))) > 1e-10 * max(1., float(np.max(np.abs(hessian.data)))):
        raise ValueError("unchanged target/material operator produced a materially different Hessian")
    hessian = frozen_hessian
    (out / "quadratic").mkdir(exist_ok=True)
    np.save(out / "quadratic/rhs.npy", np.asarray(rhs))
    save_npz(out / "quadratic/hessian.npz", hessian)
    if (out / "proposal").is_dir():
        np.save(out / "proposal/rhs.npy", np.asarray(rhs))
        save_npz(out / "proposal/hessian.npz", hessian)
    return {"nvars": nvars, "hessian_nnz": int(hessian.nnz), "rhs_shape": list(rhs.shape)}


def copy_input_artifacts(parent: Path, output: Path) -> list[str]:
    """Copy explicit solver inputs, never inherited runs, renderers or reports."""
    names = {
        'contract.json', 'source-contract.json', 'target-intent.json', 'data.npz',
        'assembly.json', 'prolongation.npz', 'target.npz', 'target_weights.npy',
        'target_deltas.npy', 'group_priors.npy', 'group_prior_weights.npy',
        'proposal/hessian.npz', 'proposal/rhs.npy',
    }
    for optional in ('complete-target-ledgers.npz', 'Dx.npz', 'Dy.npz'):
        if (parent / optional).is_file():
            names.add(optional)
    assembly = read(parent / 'assembly.json')
    for block in [*assembly['blocks'], *assembly.get('hard_equalities', [])]:
        for key in ('matrix_file', 'weights_file', 'rhs_file'):
            if not block.get(key):
                continue
            name = block[key]
            relative = Path(name)
            if (relative.is_absolute() or '..' in relative.parts
                    or relative.suffix not in ('.npy', '.npz')):
                raise ValueError(f'non-array solver input artifact: {name}')
            names.add(name)
    static = read(parent / 'signed-input-audit.json')['static_files_byte_identical']
    if not set(static).issubset(names):
        raise ValueError('parent static audit contains an undeclared solver input artifact')
    for name in names:
        if not (parent / name).is_file():
            raise ValueError(f'missing solver input artifact: {name}')
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(names):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / name, destination)
    return sorted(names)


def prepare(parent: Path, output: Path, audit_path: Path, binary: Path, amount: float) -> dict:
    parent = parent.resolve(); output = output.resolve(); audit_path = audit_path.resolve(); binary = binary.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    parent_validation = validate_input(parent)
    require_same_signed_amount(parent_validation['source_amount'], amount)
    if not binary.exists():
        raise FileNotFoundError(f"Rust controller binary not found: {binary}; run cargo build first")
    parent_contract = read(parent / "contract.json")
    parent_data = dict(np.load(parent / "data.npz"))
    user_audit = read(audit_path)
    provenance = validate_parent(parent, audit_path, parent_data, parent_contract, user_audit)
    source = provenance["source"]
    frame = provenance["frame"]
    k = provenance["k"]
    # Build the controller request and verify all source coordinates in output.
    request = controller_request(frame, source[:128], amount, k)
    rust = invoke(binary, request)
    rust_targets = np.asarray([row["target"] for row in rust["targets"]], dtype=np.float64)
    rust_sources = np.asarray([row["source"] for row in rust["targets"]], dtype=np.float64)
    if not np.array_equal(rust_sources, source[:128]):
        raise ValueError("Rust controller changed source coordinates")
    if rust_targets.shape != (128, 2) or not np.isfinite(rust_targets).all():
        raise ValueError("Rust controller returned invalid 128 targets")
    new_destinations = np.asarray(parent_data["target_destinations"], dtype=np.float64).copy()
    new_deltas = np.asarray(parent_data["target_deltas"], dtype=np.float64).copy()
    new_destinations[:128] = rust_targets
    new_deltas[:128] = rust_targets - source[:128]

    copied_inputs = copy_input_artifacts(parent, output)
    np.save(output / "target_deltas.npy", new_deltas)
    np.savez_compressed(output / "data.npz", **{**parent_data, "target_destinations": new_destinations,
                                                  "target_deltas": new_deltas})
    target_intent = read(output / "target-intent.json")
    target_intent.pop('solving_authorized', None)
    target_intent.update({
        "name": "rust-whole-face-width-controller-v1",
        "amount": float(amount),
        "parent_amount": float(amount),
        "controller": "rust-controls/face-width-targets",
        "controller_binary_sha256": sha(binary),
        "controller_request": request,
        "controller_output": rust,
        "source": source[:128].tolist(), "requested": rust_targets.tolist(),
        "changed_target_indices": np.flatnonzero(np.any(new_destinations != source, axis=1)).tolist(),
        "parent_input": str(parent), "parent_input_sha256": sha(parent / "target-intent.json"),
        "independent_source_frame": frame, "midline_u": 0.0,
        "calibration_max_radius_fraction": k, "chin_fade_start": 0.88,
        "no_solve_no_render": True, "not_aesthetic_approval": True,
    })
    save(output / "target-intent.json", target_intent)

    source_contract = read(output / "source-contract.json")
    source_contract.pop('solving_authorized', None)
    # Keep the human-readable complete source contract synchronized with the
    # numeric arrays.  Historical ledgers remain available; only the current
    # signed destination view is replaced.
    contract_targets = source_contract.get("targets")
    if not isinstance(contract_targets, list) or len(contract_targets) != len(new_destinations):
        raise ValueError("parent source-contract target ledger is incomplete")
    for index, row in enumerate(contract_targets):
        source_point = np.asarray(row["source"], dtype=np.float64)
        if not np.array_equal(source_point, source[index]):
            raise ValueError(f"source-contract target source changed at row {index}")
        row["destination"] = new_destinations[index].tolist()
        row["forward_delta"] = new_deltas[index].tolist()
    ledger_path = output / "complete-target-ledgers.npz"
    if ledger_path.exists():
        ledger = dict(np.load(ledger_path))
        if "source" not in ledger or not np.array_equal(ledger["source"], source):
            raise ValueError("complete target ledger source changed")
        ledger["signed_current"] = new_destinations
        np.savez_compressed(ledger_path, **ledger)
    source_contract.update({
        "target_definition": "rust-whole-face-width-controller-v1",
        "controller": "experiments/semantic-deformation-20261022/rust-controls",
        "controller_binary_sha256": sha(binary), "controller_request": request,
        "controller_source_audit": str(audit_path), "controller_source_audit_sha256": sha(audit_path),
        "explicit_target_redefinition": {
            **source_contract.get("explicit_target_redefinition", {}),
            "name": "rust-whole-face-width-controller-v1",
            "amount": float(amount),
            "parent_amount": float(amount),
        },
        "parent_input": str(parent), "parent_input_sha256": sha(parent / "source-contract.json"),
        "source_target_rule": "Rust controller replaces only the independently audited first 128 face destinations; all other source rows, attachment priors, weights, materials, locks and background files are copied from the frozen parent at the same signed amount.",
        "source_frame_rule": "Rust request uses the audited source frame, explicit midline_u=0, k from the source calibration ledger and chin_fade_start=.88; no bbox midpoint or previous solved field is used.",
        "no_solver_or_photo_edit": True, "not_aesthetic_approval": True,
    })
    save(output / "source-contract.json", source_contract)

    quadratic = rebuild_quadratic(output, new_deltas)
    # Fresh signed audit: all signs are independently generated from this source.
    signed_cases = {}
    signed_magnitude = abs(float(amount))
    for signed_amount in sorted({-signed_magnitude, 0.0, signed_magnitude}):
        signed_request = controller_request(frame, source[:128], signed_amount, k)
        signed_output = invoke(binary, signed_request)
        signed_cases[format(signed_amount, 'g')] = {"request": signed_request, "output": signed_output}
    negative_key = format(-signed_magnitude if signed_magnitude else 0., 'g')
    positive_key = format(signed_magnitude, 'g')
    signed = {
        "status": "prepared_source_only",
        "source_photo": str(provenance["source_photo"]), "source_photo_sha256": sha(provenance["source_photo"]),
        "source_audit": str(audit_path), "source_audit_sha256": sha(audit_path),
        "parent_input": str(parent), "parent_input_sha256": sha(parent / "contract.json"),
        "source_coordinates_unchanged": True, "frame": frame, "midline_u": 0.0,
        "calibration_max_radius_fraction": k, "chin_fade_start": 0.88,
        "cases": signed_cases,
        "signed_cases_scope": "128 source face contour rows only; non-face drives retained from the validated same-amount parent",
        "signed_source_symmetry_checked": bool(np.max(np.abs(
            np.asarray([v["target"] for v in signed_cases[negative_key]["output"]["targets"]])
            + np.asarray([v["target"] for v in signed_cases[positive_key]["output"]["targets"]])
            - 2 * source[:128])) < 1e-10),
        "no_solver_no_render": True, "not_aesthetic_approval": True,
    }
    save(output / "signed-input-audit.json", signed)

    # Update contracts only after all generated artifacts exist.
    contract = read(output / "contract.json")
    contract.pop('solving_authorized', None)
    contract.update({
        "source_contract": str(output / "source-contract.json"),
        "source_contract_sha256": sha(output / "source-contract.json"),
        "target_intent_sha256": sha(output / "target-intent.json"),
        "signed_input_audit_sha256": sha(output / "signed-input-audit.json"),
        "parent_input": str(parent), "parent_input_sha256": sha(parent / "contract.json"),
        "target_definition": "rust-whole-face-width-controller-v1",
        "controller_binary_sha256": sha(binary), "controller_source_audit_sha256": sha(audit_path),
        "independent_first_128_targets": True,
        "parent_input_contract_sha256": sha(parent / "contract.json"),
        "requested_amount": float(amount),
        "parent_requested_amount": float(amount),
        "all_parent_material_weights_locks_background_copied": True,
        "no_solver_no_render": True, "execution_status": "inputs_prepared_no_solve_no_render", "not_aesthetic_approval": True,
    })
    save(output / "contract.json", contract)
    # Complete the signed-input evidence after all mutable files are final.
    # Contract and this audit are intentionally excluded to avoid a hash cycle;
    # the contract stores the final audit hash below.
    signed["parent_inputs_sha256"] = {
        str(parent / name): sha(parent / name)
        for name in ["contract.json", "data.npz", "source-contract.json",
                     "proposal/rhs.npy", "proposal/hessian.npz"]
    }
    baseline = Path(read(output / "contract.json")["baseline"])
    signed["parent_inputs_sha256"][str(baseline / "provenance.json")] = sha(baseline / "provenance.json")
    parent_signed = read(parent / "signed-input-audit.json")
    signed["static_files_byte_identical"] = dict(parent_signed.get("static_files_byte_identical", {}))
    # Retain the ancestor control/provenance hashes required by the frozen
    # input validator; these files are copied or referenced unchanged.
    signed["parent_inputs_sha256"].update(parent_signed.get("parent_inputs_sha256", {}))
    signed["output_sha256"] = {
        str(output / name): sha(output / name)
        for name in ["data.npz", "target_deltas.npy", "target-intent.json",
                     "source-contract.json", "proposal/rhs.npy",
                     "proposal/hessian.npz", "target.npz", "target_weights.npy"]
    }
    save(output / "signed-input-audit.json", signed)
    contract = read(output / "contract.json")
    contract["signed_input_audit_sha256"] = sha(output / "signed-input-audit.json")
    save(output / "contract.json", contract)
    report = {
        "status": "PreparedForIndependentReview",
        "parent_input": str(parent), "output": str(output),
        "parent_input_sha256": sha(parent / "contract.json"),
        "source_photo": str(provenance["source_photo"]), "source_photo_sha256": sha(provenance["source_photo"]),
        "source_audit": str(audit_path), "source_audit_sha256": sha(audit_path),
        "controller_binary_sha256": sha(binary), "amount": amount,
        "source_rows": int(len(source)), "target_rows": int(len(new_deltas)),
        "controller_rows_replaced": 128,
        "parent_validation": parent_validation,
        "copied_input_artifacts": copied_inputs,
        "other_target_rows_unchanged": bool(np.array_equal(new_deltas[128:], parent_data["target_deltas"][128:])),
        "source_coordinates_unchanged": bool(np.array_equal(source, parent_data["target_sources"])),
        "target_weights_unchanged": bool(np.array_equal(np.load(output / "target_weights.npy"), parent_data["target_weights"])),
        "quadratic": quadratic, "signed_audit": str(output / "signed-input-audit.json"),
        "copied_material_and_background_files": True,
        "no_solve_no_render": True, "execution_status": "inputs_prepared_no_solve_no_render", "not_aesthetic_approval": True,
        "limitations": [
            "This is a source target/controller experiment; it has no image output or beauty judgement.",
            "The Rust controller target variant is intentionally not bit-identical to the old lateral-gated request.",
            "Downstream validation must retain the original coupled hard-line and visibility constraints; no solve or render has been executed.",
        ],
    }
    save(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--amount", type=float, default=30.0)
    args = parser.parse_args()
    result = prepare(args.parent, args.output, args.audit, args.binary, args.amount)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
