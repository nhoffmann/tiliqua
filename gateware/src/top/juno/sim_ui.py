# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
Menu screenshots of the juno build without hardware: one PNG per menu page.

Runs the upstream full-SoC Verilator simulation (``juno sim``: gateware + CPU +
real firmware, video frames as BMP) with the firmware built with
``--features sim_pages``, which steps to the next menu page every 400 ms (the
harness has no encoder) and logs ``sim_pages: page <name>`` on the UART. The last
frame before each page change is saved as ``<nn>_<page>.png``, cropped to the
menu block (``--full`` keeps the whole frame).

Usage (from ``gateware/``, takes a few minutes; Verilator build ~2 min, ~1 frame/s)::

    pdm run python -m top.juno.sim_ui [--out build/juno_ui] [--modeline 1280x720p60] [--full]

Stack high-water measurement (``--features sim_stack``: scripted MIDI, preset and
option activity on a PSRAM-backed option storage, ``stack_watch`` logging; runs
the whole 5 s and prints the firmware log)::

    pdm run python -m top.juno.sim_ui --features sim_stack --out build/juno_stack

Upstream ``tiliqua.build.sim`` is used unchanged; this script only wraps its
subprocess calls at runtime:

- ``verilator``: adds ``-Wno-fatal`` (Verilator >= 5.05x warns ``UNOPTPUBLIC`` on
  the VexiiRiscv netlist, fatal by default) and ``-std=c++17`` (the harness uses
  class template argument deduction), and points ``FIRMWARE_BIN_PATH`` at the
  ``sim_pages`` firmware (``firmware_sim.bin``, built next to ``firmware.bin``).
- the simulator binary: not run by ``sim.py``; started here under a pty
  (line-buffered UART log), stopped after the last page. The upstream harness
  ends after 5 s simulated time (``sim_time``), so boot + 10 pages x 400 ms fit.
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

# Menu crop: the menu is a small block on an otherwise black frame. The previous
# page is still fading out when a page is captured (400 ms per page); its pixels
# are dimmer (green <= ~52) than the live menu text and drawings (green >= ~70).
CROP_GREEN = 60
CROP_FOOTER = 80            # bottom rows with the title / version line
CROP_PAD = 16


def crop_menu(im):
    """Crop a frame to the live menu block (bounding box of bright pixels)."""
    w, h = im.size
    green = im.convert("RGB").getchannel("G").crop((0, 0, w, h - CROP_FOOTER))
    box = green.point(lambda v: 255 if v > CROP_GREEN else 0).getbbox()
    if box is None:
        return im
    x0, y0, x1, y1 = box
    return im.crop((max(x0 - CROP_PAD, 0), max(y0 - CROP_PAD, 0),
                    min(x1 + CROP_PAD, w), min(y1 + CROP_PAD, h)))


def build_sim(modeline, features):
    """Run ``juno sim`` up to (not including) the simulator run. Returns the binary path."""
    import tiliqua.build.sim as tsim
    sys.path.insert(0, JUNO)    # top.py imports its sibling modules directly
    from top.juno.top import JunoSoc, archiver_callback
    from tiliqua.build.cli import top_level_cli

    real_check_call = subprocess.check_call
    state = {}

    def check_call(cmd, *args, **kwargs):
        if cmd and cmd[0] == "verilator":
            fw_bin = next(c for c in cmd if "FIRMWARE_BIN_PATH" in c)
            path = re.search(r'FIRMWARE_BIN_PATH=\\"(.*)\\"', fw_bin).group(1)
            sim_bin = os.path.join(os.path.dirname(path), "firmware_sim.bin")
            print(f"sim_ui: building firmware ({features}) -> {sim_bin}")
            real_check_call(["cargo", "objcopy", "--release", "--features", features,
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
        top_level_cli(JunoSoc, path=JUNO, archiver_callback=archiver_callback)
    except SystemExit as e:
        if e.code not in (0, None):
            raise
    finally:
        tsim.subprocess.check_call = real_check_call
    return state["binary"]


def capture(binary, out, full=False, log=False):
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
        im = Image.open(os.path.join(out, frame))
        (im if full else crop_menu(im)).save(png)
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
        if page.upper() == LAST_PAGE and not log:
            n_last += 1
            if n_last >= FRAMES_LAST_PAGE:
                return True
        return False

    def on_uart(line):
        nonlocal candidate, page
        if log and line:
            print("uart:", line)
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
    parser.add_argument("--full", action="store_true", help="keep whole frames (no menu crop)")
    parser.add_argument("--reuse", action="store_true",
                        help="keep the last Verilator build (build/obj_dir), only rebuild the firmware")
    parser.add_argument("--features", default="sim_pages",
                        help="firmware features (sim_stack: stack measurement, prints the log)")
    args = parser.parse_args()
    os.chdir(GATEWARE)
    binary = os.path.join(GATEWARE, "build", "obj_dir", "Vtiliqua_soc")
    if args.reuse and os.path.exists(binary):
        # the simulator loads FIRMWARE_BIN_PATH (firmware_sim.bin) at startup
        sim_bin = os.path.join(GATEWARE, "build", "juno-r5", "firmware_sim.bin")
        print(f"sim_ui: reusing {binary}, firmware ({args.features}) -> {sim_bin}")
        subprocess.check_call(["cargo", "objcopy", "--release", "--features", args.features,
                               "--", "-Obinary", sim_bin], cwd=FW)
    else:
        binary = build_sim(args.modeline, args.features)
    saved = capture(binary, os.path.abspath(args.out), args.full,
                    log="stack" in args.features)
    print(f"sim_ui: {len(saved)} screenshots in {args.out}")


if __name__ == "__main__":
    main()
