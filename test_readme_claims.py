"""Checks for claims the README makes that numpy can verify without Desmos: the generator still writes the exact
files behind the two saved graphs, and numpy training from seed 0 reaches the walking test.

Which generation seed 0 passes at depends on the CPU: numpy picks vectorised code by instruction set, and the
last-bit differences grow through the chaotic walk (generation 13 on the laptop the README was written on, 26 there
with AVX2 turned off; 19 on a GitHub runner with AVX-512, 13 on one with AVX2 only). So the generation is printed, with the CPU features
numpy found, not asserted.

The hashes are of the files that were loaded into https://www.desmos.com/calculator/zzd5dnayog and
https://www.desmos.com/calculator/yywath1bwc; on 2026-10-24 every expression of both saved graphs matched these files.
The Desmos-against-numpy comparison itself needs a browser: see check_desmos.py.
"""
import hashlib
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SAVED_GRAPHS = {
    "zzd5dnayog": ([], "648affbc0d00e11e58ba4f19a3a31ca44a94a855e69851849ecd12d7b7397d9c"),
    "yywath1bwc": (["--long", "--dirs", "48", "--top", "24"], "1373a08ee996b51d4f20cb8aab81df68dd9af18a0b201791cfc4294a58cfb7ea"),
}
TIMEOUT_S = 600


def simd_features():
    """The vector instruction sets numpy dispatches to here, for the printout (numpy keeps this table private)."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__
    except ImportError:
        return "unknown"
    return " ".join(k for k in ("AVX2", "FMA3", "AVX512F", "AVX512_SKX", "ASIMD") if __cpu_features__.get(k)) or "baseline only"


def run(args, cwd):
    p = subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True, text=True, timeout=TIMEOUT_S)
    assert p.returncode == 0, f"{' '.join(args)} failed ({p.returncode}):\n{p.stderr[-2000:]}"
    return p.stdout


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for name, (flags, want) in SAVED_GRAPHS.items():
            out = os.path.join(tmp, name + ".json")
            run([os.path.join(HERE, "gen_desmos.py"), out, *flags, "--paused"], tmp)
            with open(out, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
            assert got == want, f"gen_desmos.py no longer writes the file behind {name}: sha256 {got}"
        log = run([os.path.join(HERE, "proto", "ars.py"), "200", "0"], tmp)
        passed = re.search(r"seed 0: mean policy passed the strict bar at generation (\d+) ", log)
        assert passed, "numpy seed 0 no longer reaches the walking test within 200 generations:\n" + log[-2000:]
    print(f"numpy seed 0 passed at generation {passed.group(1)} on this machine (numpy SIMD: {simd_features()})")
    print("all checks passed")


if __name__ == "__main__":
    main()
