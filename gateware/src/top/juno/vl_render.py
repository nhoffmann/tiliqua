# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Fast offline renders of ``JunoCore`` with Verilator.

Same inputs and bit-identical output as ``sim_render.render`` (amaranth.sim),
but compiled: seconds instead of ~15 minutes per second of audio. Unit tests
keep using amaranth.sim; ``tests/test_juno_voice.py`` checks both engines agree.

The Verilog is generated from ``JunoCore`` and built with ``vl_harness.cpp`` into
``build/juno_vl/<hash>/``; the hash covers the Verilog, the harness and the
Verilator version, so it only rebuilds when the design changes.

Requires ``verilator`` on PATH (macOS: ``brew install verilator``).
"""

import hashlib
import os
import shutil
import subprocess
import tempfile

import numpy as np

from amaranth.back import verilog

from top.juno.juno_core import JunoCore
from top.juno.sim_render import N_VOICES, prepare

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vl_harness.cpp")
BUILD_ROOT = os.path.join("build", "juno_vl")


def available():
    return shutil.which("verilator") is not None


def build(n_voices=N_VOICES, verbose=False):
    """Generate Verilog + build the harness (cached). Returns the binary path."""
    if n_voices != N_VOICES:
        raise ValueError(f"vl_harness.cpp is written for {N_VOICES} voices")
    v = verilog.convert(JunoCore(n_voices=n_voices), name="juno_core", emit_src=False)
    harness = open(HARNESS).read()
    version = subprocess.run(["verilator", "--version"], capture_output=True,
                             text=True, check=True).stdout
    key = hashlib.sha256((v + harness + version).encode()).hexdigest()[:12]
    out_dir = os.path.abspath(os.path.join(BUILD_ROOT, key))
    binary = os.path.join(out_dir, "obj_dir", "Vjuno_core")
    if os.path.exists(binary):
        return binary
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "juno_core.v"), "w") as f:
        f.write(v)
    shutil.copy(HARNESS, os.path.join(out_dir, "vl_harness.cpp"))
    cmd = ["verilator", "--cc", "--exe", "--build", "-j", "0", "-O3",
           "--x-assign", "0", "--x-initial", "0",
           "-Wno-fatal", "-Wno-lint", "-Wno-style",
           "--top-module", "juno_core",
           "-CFLAGS", "-O2",
           "juno_core.v", "vl_harness.cpp"]
    result = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.exists(binary):
        raise RuntimeError("verilator build failed:\n" + result.stdout[-4000:]
                           + result.stderr[-4000:])
    if verbose:
        print(f"built {binary}")
    return binary


def render_vl(notes, seconds, params=None, n_voices=N_VOICES):
    """Like ``sim_render.render``: float array (n_samples, 2)."""
    binary = build(n_voices)
    n_samples, settings, events = prepare(notes, seconds, params)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.txt")
        out = os.path.join(tmp, "out.raw")
        with open(inp, "w") as f:
            f.write(f"samples {n_samples}\n")
            for name, raw in settings.items():
                f.write(f"set {name} {raw}\n")
            for e in events:
                f.write("event " + " ".join(str(x) for x in e) + "\n")
        subprocess.run([binary, inp, out], check=True)
        raw = np.fromfile(out, dtype="<i2").reshape(-1, 2)
    return raw.astype(np.float64) / 32768
