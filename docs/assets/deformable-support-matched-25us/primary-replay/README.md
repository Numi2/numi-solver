# Matched L3/L4 primary spatial replay

These are the lossless primary CSV trajectories from the same 0.5 s, 25 µs support-Verlet drop: 2,057/14,993 nodes, 10,240/81,920 tetrahedra, and 101 exported states each. The two energy ledgers and topology files are included. `compression-receipt.json` records native source hashes before and after compression and the exact decompressed and archive hashes. `prepare.py` is the source-side compression recipe.

From this directory, run `python3 replay.py .` to verify all six input hashes against the receipt and the published [spatial result](../spatial.json), [L3 geometry audit](../l3-primary-geometry.json), and [L4 geometry audit](../../deformable-support-81920-geometry.json). It streams the CSV rows, checks topology-derived mass and accepted step grids, then independently recomputes the matched-node, center-of-mass, and height differences. The saved [fresh Python 3.9 replay](actual-replay.json) matches every published comparison field exactly, including the **9.340864644828101 mm** maximum at frame 98/node 11; the 1 mm spatial target **fails**. Python 3.14 differs by only 6.5e-19 m in the center-of-mass maximum, within the script's explicit 1e-15 m oracle envelope; its maximum node difference also matches exactly.

Run `python3 controls.py .` for three fail-closed mutations: changed energy bytes, truncated trace gzip, and a false published maximum. The [control result](controls-result.json) records actual nonzero exits.

This replay checks sampled native exports. It does not rerun native integration, bundle OBJ surfaces, or establish continuous-time convergence, material calibration, fruit coupling, or contact-work closure.
