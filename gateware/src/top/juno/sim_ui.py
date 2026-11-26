# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Menu screenshots of the juno build without hardware: one PNG per menu page.

Runs the upstream full-SoC Verilator simulation (``juno sim``: gateware + CPU +
real firmware, video frames as BMP) with the firmware built with
``--features sim_pages``, which steps to the next menu page every 450 ms (the
harness has no encoder) and logs ``sim_pages: page <name>`` on the UART. The last
frame before each page change is saved as ``<nn>_<page>.png``.

Usage (from ``gateware/``, takes a few minutes; Verilator build ~2 min, ~1 frame/s)::

    pdm run python -m top.juno.sim_ui [--out build/juno_ui] [--modeline 1280x720p60]

Upstream ``tiliqua.build.sim`` is used unchanged; this script only wraps its
subprocess calls at runtime:

- ``verilator``: adds ``-Wno-fatal`` (Verilator >= 5.05x warns ``UNOPTPUBLIC`` on
  the VexiiRiscv netlist, fatal by default) and ``-std=c++17`` (the harness uses
  class template argument deduction), and points ``FIRMWARE_BIN_PATH`` at the
  ``sim_pages`` firmware (``firmware_sim.bin``, built next to ``firmware.bin``).
- the simulator binary: not run by ``sim.py``; started here under a pty
  (line-buffered UART log), stopped after the last page. The upstream harness
  ends after 5 s simulated time (``sim_time``), so boot + 9 pages x 450 ms fit.
"""

import argparse
import os
import re
import shutil
import signal
import subprocess
import sys
import time

GATEWARE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JUNO = os.path.dirname(os.path.abspath(__file__))
FW = os.path.join(JUNO, "fw")

LAST_PAGE = "MISC"          # page names as logged by the firmware
FRAMES_LAST_PAGE = 24      # frames to wait on the last page before stopping
TIMEOUT_S = 1800


def build_sim(modeline):
    """Run ``juno sim`` up to (not including) the simulator run. Returns the binary path."""
    import tiliqua.build.sim as tsim
    sys.path.insert(0, JUNO)    # top.py imports its sibling modules directly
    from top.juno.top import JunoSoc
    from tiliqua.build.cli import top_level_cli

    real_check_call = subprocess.check_call
    state = {}

    def check_call(cmd, *args, **kwargs):
        if cmd and cmd[0] == "verilator":
            fw_bin = next(c for c in cmd if "FIRMWARE_BIN_PATH" in c)
            path = re.search(r'FIRMWARE_BIN_PATH=\\"(.*)\\"', fw_bin).group(1)
            sim_bin = os.path.join(os.path.dirname(path), "firmware_sim.bin")
            print(f"sim_ui: building sim_pages firmware -> {sim_bin}")
            real_check_call(["cargo", "objcopy", "--release", "--features", "sim_pages",
                             "--", "-Obinary", sim_bin], cwd=FW)
            cmd = [c.replace(path, sim_bin) for c in cmd]
            cmd = cmd[:1] + ["-Wno-fatal", "-CFLAGS", "-std=c++17"] + cmd[1:]
            return real_check_call(cmd, *args, **kwargs)
        if cmd and cmd[0].endswith("Vtiliqua_soc"):
            state["binary"] = os.path.abspath(cmd[0])
            return 0
        return real_check_call(cmd, *args, **kwargs)

    tsim.subprocess.check_call = check_call
    sys.argv = ["src/top/juno/top.py", "sim", "--hw", "r5", "--modeline", modeline]
    try:
        top_level_cli(JunoSoc, path=JUNO,
                      archiver_callback=lambda archiver: archiver.with_option_storage())
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        tsim.subprocess.check_call = real_check_call
    return state["binary"]


def capture(binary, out):
    from PIL import Image

    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.endswith(".bmp") or f.endswith(".png"):
            os.remove(os.path.join(out, f))
    # the harness loads build/bootinfo.bin relative to its cwd
    link = os.path.join(out, "build")
    if os.path.islink(link):
        os.remove(link)
    os.symlink(os.path.join(GATEWARE, "build"), link)

    proc = subprocess.Popen(["script", "-q", "/dev/null", binary], cwd=out,
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    page, candidate, n_last, saved = "HELP", None, 0, []

    def save(frame, name):
        idx = len(saved) + 1
        png = os.path.join(out, f"{idx:02d}_{name.lower()}.png")
        Image.open(os.path.join(out, frame)).save(png)
        saved.append(png)
        print(f"sim_ui: {name:6s} <- {frame} -> {png}")

    def on_frame(n):
        nonlocal candidate, n_last
        if n == 0:
            return False
        prev = f"frame{n - 1:02d}.bmp"       # the previous frame is complete now
        if candidate and candidate != prev:
            os.remove(os.path.join(out, candidate))
        candidate = prev
        if page.upper() == LAST_PAGE:
            n_last += 1
            if n_last >= FRAMES_LAST_PAGE:
                return True
        return False

    def on_uart(line):
        nonlocal candidate, page
        if "sim_pages: page" in line:
            if candidate:
                save(candidate, page)
                os.remove(os.path.join(out, candidate))
                candidate = None
            page = line.split()[-1]
        elif "panic" in line.lower():
            print("sim_ui: firmware:", line)

    # The harness prints firmware UART bytes one by one and its own frame messages
    # with printf on the same stdout, so a frame message can land in the middle of
    # a UART line: cut frame messages out of the stream first, then split lines.
    dvi = re.compile(r"DVIDriver: frame(\d+)\.bmp\r?\n")
    buf, done = "", False
    t0 = time.time()
    try:
        while not done:
            chunk = os.read(proc.stdout.fileno(), 4096)
            if not chunk:
                break                           # harness reached its sim_time (5 s)
            buf += chunk.decode(errors="replace")
            while not done:
                m = dvi.search(buf)
                nl = buf.find("\n")
                if m and (nl == -1 or nl >= m.start()):
                    buf = buf[:m.start()] + buf[m.end():]
                    done = on_frame(int(m.group(1)))
                elif nl != -1:
                    line, buf = buf[:nl].strip(), buf[nl + 1:]
                    on_uart(line)
                else:
                    break
            if time.time() - t0 > TIMEOUT_S:
                print("sim_ui: timeout")
                break
        if candidate:                           # last page (stop or end of sim)
            save(candidate, page)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        subprocess.run(["pkill", "-f", binary])
        proc.wait()
        for f in os.listdir(out):
            if f.endswith(".bmp"):
                os.remove(os.path.join(out, f))
        os.remove(link)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--out", default="build/juno_ui")
    parser.add_argument("--modeline", default="1280x720p60")
    args = parser.parse_args()
    os.chdir(GATEWARE)
    binary = build_sim(args.modeline)
    saved = capture(binary, os.path.abspath(args.out))
    print(f"sim_ui: {len(saved)} screenshots in {args.out}")


if __name__ == "__main__":
    main()
