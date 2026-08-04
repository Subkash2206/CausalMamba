#!/usr/bin/env python3
"""
verify_phase0.py — Phase 0 CSV regression checker.

Compares every *.csv in a new (Phase 0) results directory against its exact
counterpart in an old reference results directory, ensuring all numeric values
(Dice, IoU, BF1, AVR, ...) match to 4 decimal places.

Usage:
    python verify_phase0.py <reference_dir> <verification_dir> [--tol 5e-5]

Matching:
    1. exact relative-path match inside <reference_dir>;
    2. fallback: timestamp-stripped filename match (the experiments write
       both 'results.csv' and 'results_2026-07-26_19-49-17.csv' variants).

Comparison:
    - numeric columns are compared with numpy.isclose(..., atol=tol, rtol=0.0);
    - the default tol=5e-5 is the rounding boundary for 4 decimal places
      (|a - b| < 5e-5  <=>  a and b round to the same 4-dp value);
    - non-numeric columns are reported but skipped;
    - rows are aligned on a key column when one is identifiable (e.g.
      Metric / Layer / cutoff), otherwise positionally.

Exit code:
    0  -> ✅ VERIFIED: All metrics match
    1  -> ❌ MISMATCH FOUND
    2  -> usage / I/O error (bad paths, no CSVs found)
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

# The report uses Unicode glyphs (✅ ❌ ⚠ ✓ ✖ ≈). On Windows the default
# console encoding (cp1252) cannot encode these when stdout is piped or
# redirected, so force UTF-8 output with a lossless-enough fallback.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass  # stdout not in text mode / already live console — nothing to do

DEFAULT_TOL = 5e-5  # equality to 4 decimal places (rounding boundary = 5e-5)

# Matches the timestamp suffix used by the experiments:
#   filename_YYYY-MM-DD_HH-MM-SS.csv  ->  filename.csv
_TS_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?=\.[^.]*$)")


def strip_timestamp(name):
    """Remove a trailing 'YYYY-MM-DD_HH-MM-SS' timestamp from a filename."""
    return _TS_RE.sub("", name)


def find_counterpart(rel_path, reference_dir):
    """Locate the reference file for a verification CSV.

    Returns (abs_path, method) or (None, None).

    Matching strategies, in order:
      1. exact relative-path match inside <reference_dir>;
      2. verification has timestamp, reference has the plain variant;
      3. verification is plain, reference has a timestamped variant of the
         same name (the experiments write both results.csv and
         results_<timestamp>.csv in the same directory).
    """
    exact = os.path.join(reference_dir, rel_path)
    if os.path.isfile(exact):
        return exact, "exact"

    directory, filename = os.path.split(rel_path)
    cand_dir = os.path.join(reference_dir, directory) if directory else reference_dir

    if os.path.isdir(cand_dir):
        # 2. Strip the timestamp from the verification filename.
        stripped = strip_timestamp(filename)
        if stripped != filename:
            cand = os.path.join(cand_dir, stripped)
            if os.path.isfile(cand):
                return cand, "fuzzy(verification timestamp-stripped)"

        # 3. Reference may hold the timestamped variant; find it by comparing
        #    timestamp-stripped reference filenames with the plain verification
        #    filename.
        for ref_name in sorted(os.listdir(cand_dir)):
            if ref_name.lower().endswith(".csv") and strip_timestamp(ref_name) == filename:
                return os.path.join(cand_dir, ref_name), f"fuzzy(reference {ref_name})"

    return None, None


def key_column(df_ref, df_new):
    """Pick a column usable as a row identifier, or None for positional compare."""
    common = [c for c in df_ref.columns if c in df_new.columns]

    # Preferred well-known key names (case-insensitive).
    preferred = ("cutoff", "metric", "layer", "name", "image", "id")
    for name in preferred:
        cand = next((c for c in common if c.lower() == name), None)
        if cand is not None:
            return cand

    # Otherwise: exactly one shared non-numeric column -> use it.
    nonnum = []
    for c in common:
        try:
            pd.to_numeric(df_ref[c], errors="raise")
            pd.to_numeric(df_new[c], errors="raise")
        except (ValueError, TypeError):
            nonnum.append(c)
    if len(nonnum) == 1:
        return nonnum[0]
    return None


def numeric_columns(df):
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _as_float(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def compare_file(verif_path, ref_path, tol):
    """Compare one CSV pair. Returns (ok, messages)."""
    messages = []
    try:
        df_ref = pd.read_csv(ref_path)
        df_new = pd.read_csv(verif_path)
    except Exception as e:  # noqa: BLE001
        return False, [f"    ✖ could not read CSV: {e}"]

    # --- empty-file sanity --------------------------------------------------
    if df_ref.empty or df_new.empty:
        if len(df_ref) != len(df_new):
            return False, [f"    ✖ row count differs: ref={len(df_ref)}, new={len(df_new)}"]
        return True, ["    ✓ (both files empty)"]

    num_cols_ref = numeric_columns(df_ref)
    num_cols_new = numeric_columns(df_new)
    shared = [c for c in num_cols_ref if c in num_cols_new]

    for c in num_cols_ref:
        if c not in num_cols_new:
            messages.append(
                f"    ⚠ column '{c}' present in reference but missing in verification (skipped)")
    for c in num_cols_new:
        if c not in num_cols_ref:
            messages.append(
                f"    ⚠ column '{c}' present in verification but missing in reference (skipped)")

    if not shared:
        return False, [
            f"    ✖ no shared numeric columns to compare "
            f"(ref numeric={num_cols_ref}, new numeric={num_cols_new})"
        ]

    kcol = key_column(df_ref, df_new)
    if kcol is not None and (df_ref[kcol].duplicated().any() or df_new[kcol].duplicated().any()):
        kcol = None  # key is not unique -> fall back to positional alignment

    did_fail = False
    if kcol is not None:
        # Align rows on the key column (outer join preserves row identity).
        cols = list(dict.fromkeys([kcol] + shared))  # dedupe, keep order
        merged = (
            df_ref[cols]
            .merge(df_new[cols], on=kcol, how="outer", suffixes=("_ref", "_new"), indicator=True)
        )

        missing_rows = merged[merged["_merge"] != "both"]
        for _, row in missing_rows.iterrows():
            side = "reference" if row["_merge"] == "left_only" else "verification"
            messages.append(f"    ✖ row '{row[kcol]}' only in {side} results")
        did_fail = bool(len(missing_rows) > 0)

        both = merged[merged["_merge"] == "both"]
        for col in shared:
            if col == kcol:
                # Join key: identical by construction on matched rows.
                continue
            rv = _as_float(both[f"{col}_ref"])
            nv = _as_float(both[f"{col}_new"])
            ok_mask = np.isclose(rv, nv, atol=tol, rtol=0.0, equal_nan=True)
            if not np.all(ok_mask):
                did_fail = True
                for i in np.where(~ok_mask)[0][:20]:
                    messages.append(
                        f"    ✖ row '{both.iloc[i][kcol]}' col '{col}': "
                        f"ref={rv[i]:.10g} vs new={nv[i]:.10g} "
                        f"(diff={abs(rv[i] - nv[i]):.3g} > tol={tol:g})"
                    )
    else:
        # Positional fallback: row identity by index.
        if len(df_ref) != len(df_new):
            return False, [
                f"    ✖ row count differs: ref={len(df_ref)} rows vs "
                f"new={len(df_new)} rows"
            ]
        for col in shared:
            rv = _as_float(df_ref[col])
            nv = _as_float(df_new[col])
            ok_mask = np.isclose(rv, nv, atol=tol, rtol=0.0, equal_nan=True)
            if not np.all(ok_mask):
                did_fail = True
                for i in np.where(~ok_mask)[0][:20]:
                    messages.append(
                        f"    ✖ row #{i} col '{col}': "
                        f"ref={rv[i]:.10g} vs new={nv[i]:.10g} "
                        f"(diff={abs(rv[i] - nv[i]):.3g} > tol={tol:g})"
                    )

    if did_fail:
        extra = "" if not messages or "✖ row count" in messages[0] else " ✖ (details above)"
        messages.append(f"    ✖ MISMATCH in this file{extra}")
    else:
        skipped = f" ({len(messages)} column(s) skipped)" if messages else ""
        messages.append(f"    ✓ all {len(shared)} shared numeric column(s) match to {tol:.0e}{skipped}")
    return (not did_fail), messages


def main():
    parser = argparse.ArgumentParser(
        description="Compare Phase 0 result CSVs against a reference results "
        "directory (numeric equality to 4 decimal places)."
    )
    parser.add_argument("reference_dir", help="old results directory (the reference)")
    parser.add_argument("verification_dir", help="new Phase 0 results directory")
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"absolute tolerance for numeric comparison (default: {DEFAULT_TOL:g})",
    )
    args = parser.parse_args()

    ref_dir = os.path.abspath(args.reference_dir)
    ver_dir = os.path.abspath(args.verification_dir)

    if not os.path.isdir(ref_dir):
        print(f"ERROR: reference_dir not found: {ref_dir}")
        return 2
    if not os.path.isdir(ver_dir):
        print(f"ERROR: verification_dir not found: {ver_dir}")
        return 2

    verif_csvs = []
    for root, _dirs, files in os.walk(ver_dir):
        for fname in sorted(files):
            if fname.lower().endswith(".csv"):
                rel = os.path.relpath(os.path.join(root, fname), ver_dir)
                verif_csvs.append(rel)

    if not verif_csvs:
        print(f"No .csv files found under {ver_dir}")
        return 2

    print("=" * 78)
    print("Phase 0 Metrics Verification")
    print("=" * 78)
    print(f"Reference dir:    {ref_dir}")
    print(f"Verification dir: {ver_dir}")
    print(f"Tolerance (abs):  {args.tol:g}  (default ≈ 4 decimal places)")
    print("=" * 78)

    n_compared = 0
    n_passed = 0
    n_missing = 0
    all_ok = True

    for rel in verif_csvs:
        verif_path = os.path.join(ver_dir, rel)
        ref_path, method = find_counterpart(rel, ref_dir)

        print(f"\n[{rel}]")
        if ref_path is None:
            print("    ⚠ SKIPPED — no counterpart found in reference (exact & fuzzy match failed)")
            n_missing += 1
            continue

        print(f"    counterpart: {os.path.relpath(ref_path, ref_dir)}  ({method})")
        ok, msgs = compare_file(verif_path, ref_path, args.tol)
        n_compared += 1
        for m in msgs:
            print(m)
        if ok:
            n_passed += 1
        else:
            all_ok = False

    print("\n" + "=" * 78)
    missing_str = f", {n_missing} skipped (no counterpart)" if n_missing else ""
    print(f"Compared {n_compared} CSV pair(s): {n_passed} passed, "
          f"{n_compared - n_passed} failed{missing_str}.")
    if all_ok and n_compared > 0:
        print("✅ VERIFIED: All metrics match")
        return 0
    print("❌ MISMATCH FOUND")
    return 1


if __name__ == "__main__":
    sys.exit(main())