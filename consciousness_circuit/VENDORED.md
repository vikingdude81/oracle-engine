# Vendored Package Notice

This directory is a **vendored snapshot** of the canonical `consciousness_circuit`
package. Do not edit files here directly — changes will be overwritten on the next sync.

- **Canonical source:** `harmonic-field-consciousness/consciousness_circuit`
  (https://github.com/vikingdude81/harmonic-field-consciousness)
- **Synced version:** 3.5.1
- **Synced from commit:** 783d65f (harmonic-field-consciousness)
- **Synced on:** 2026-07-12

## Why vendored?

oracle-engine imports `consciousness_circuit` from the repo root. Vendoring keeps
the demo/API scripts working with zero install steps. The long-term plan is to
consume the package as a pip dependency instead:

```bash
pip install "git+https://github.com/vikingdude81/harmonic-field-consciousness#subdirectory=consciousness_circuit"
```

## How to re-sync

From the GitHub folder root:

```bash
rm -rf oracle-engine/consciousness_circuit
cp -r harmonic-field-consciousness/consciousness_circuit oracle-engine/consciousness_circuit
cd oracle-engine/consciousness_circuit
rm -rf __pycache__ */__pycache__ consciousness_circuit.egg-info
# then update the commit hash and date in this file
```

## Local additions (survive re-sync — re-add if lost)

- `test_trajectory_standalone.py` — standalone unit tests unique to oracle-engine
- `VENDORED.md` — this file
