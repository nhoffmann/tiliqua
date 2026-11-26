# SPDX-License-Identifier: CERN-OHL-S-2.0
"""
``JunoFx``: Juno global effects on the mono voice sum (Juno port, Phase 3):
4-position HPF, then a BBD-style stereo chorus (Off / I / II / I+II).

Bit-exact with ``top.juno.fx_model.FxModel`` (see there for the algorithm).
One ``PipelinedMAC``, ~12 multiplies per sample (~60 cycles), one 512-sample
delay line (1 BRAM).
"""

from amaranth import *
from amaranth.lib import data, stream, wiring
from amaranth.lib.memory import Memory
from amaranth.lib.wiring import In, Out

from tiliqua.dsp import ASQ, mac

# top.py runs as a script (sibling imports); tests / sim_render import the package.
try:
    import fx_model as FM
except ImportError:
    from top.juno import fx_model as FM


def clamp_bits(m, x, bits):
    """Saturate ``x`` to a signed ``bits``-wide range [-2**(bits-1), 2**(bits-1) - 1].
    Checks that all bits above ``bits - 1`` equal the sign (a few LUTs), instead of
    two full-width comparisons (carry chains) per call site."""
    x = Value.cast(x)
    v = Signal(signed(bits))
    upper = x[bits - 1:]
    in_range = (upper == 0) | (upper == (1 << len(upper)) - 1)
    m.d.comb += v.eq(Mux(in_range, x[:bits].as_signed(),
                         Mux(x[-1], -(1 << (bits - 1)), (1 << (bits - 1)) - 1)))
    return v


class JunoFx(wiring.Component):

    def __init__(self, macp):
        """``macp``: a ``juno_voice.PipelinedMAC`` (passed in, see ``JunoLfo``)."""
        self.macp = macp
        super().__init__({
            "i":           In(stream.Signature(ASQ)),
            "o":           Out(stream.Signature(data.ArrayLayout(ASQ, 2))),
            "hpf_mode":    In(2),     # 0 bass boost, 1 flat, 2 240 Hz, 3 720 Hz
            "chorus_mode": In(2),     # 0 off, 1 I, 2 II, 3 I+II
        })

    def elaborate(self, platform):
        m = Module()
        ONE, SAT = FM.ONE, FM.SAT
        W = 22
        mp = m.submodules.macp = self.macp

        def sat(x):
            return clamp_bits(m, x, 18)

        mac_a = Signal(mac.SQNative)
        mac_b = Signal(mac.SQNative)
        z = Signal(signed(40))
        m.d.comb += z.eq(mp.result.z.as_value() >> 15)

        def multiply(a, b, assigns, next_state):
            m.d.comb += [mac_a.as_value().eq(a), mac_b.as_value().eq(b)]
            with mp.Multiply(m, a=mac_a, b=mac_b):
                m.d.sync += assigns(z)
                m.next = next_state

        # delay line
        n_buf = 1 << FM.DELAY_BITS
        m.submodules.buf = buf = Memory(shape=signed(18), depth=n_buf, init=[])
        rd = buf.read_port()
        wr = buf.write_port()
        m.d.comb += rd.en.eq(1)

        # state
        hpf_s  = Signal(signed(W))
        pre_s  = Signal(signed(W))
        post_s = [Signal(signed(W), name=f"post_s{c}") for c in range(2)]
        cur_post = Signal(signed(W))   # post-filter state of the current channel
        wptr   = Signal(FM.DELAY_BITS)
        ph_i   = Signal(32)
        ph_ii  = Signal(32)

        # per-sample working registers
        x     = Signal(signed(W))
        lp    = Signal(signed(W))
        y     = Signal(signed(W))
        pre   = Signal(signed(W))
        mod   = Signal(signed(W))
        d     = Signal(signed(W))
        frac  = Signal(FM.DELAY_FRAC)
        a     = Signal(signed(W))
        b     = Signal(signed(W))
        tap   = Signal(signed(W))
        wet   = Signal(signed(W))
        dry_t = Signal(signed(W))
        ch    = Signal()
        outs  = [Signal(signed(16), name=f"out{c}") for c in range(2)]

        hpf_g = Array([C(g, 16) for g in FM.HPF_G])
        chorus_on = self.chorus_mode != FM.CHORUS_OFF
        use_i = (self.chorus_mode == FM.CHORUS_I) | (self.chorus_mode == FM.CHORUS_I_II)
        use_ii = (self.chorus_mode == FM.CHORUS_II) | (self.chorus_mode == FM.CHORUS_I_II)

        def tri(ph):
            hi = ph[16:32]
            dd = Mux(hi >= ONE, hi - ONE, ONE - hi)
            return ONE - 2 * dd

        def onepole(G, inp, s, assigns_lp, next_state):
            """lp = sat(v + s), s' = sat(lp + v) with v = G * sat(inp - s)."""
            def done(z):
                lpv = sat(z + s)
                return [s.eq(sat(lpv + z))] + assigns_lp(lpv)
            multiply(G, sat(inp - s), done, next_state)

        with m.FSM():

            with m.State('IDLE'):
                m.d.comb += self.i.ready.eq(1)
                with m.If(self.i.valid):
                    m.d.sync += x.eq(self.i.payload.as_value())
                    m.next = 'HPF'

            with m.State('HPF'):
                onepole(hpf_g[self.hpf_mode], x, hpf_s, lambda v: [lp.eq(v)], 'SHELF')

            with m.State('SHELF'):
                with m.If(self.hpf_mode == 0):
                    multiply(FM.SHELF_K, lp, lambda z: [y.eq(sat(x + z))], 'PRE')
                with m.Elif(self.hpf_mode == 1):
                    m.d.sync += y.eq(x)
                    m.next = 'PRE'
                with m.Else():
                    m.d.sync += y.eq(sat(x - lp))
                    m.next = 'PRE'

            with m.State('PRE'):
                onepole(FM.G_15K, y, pre_s, lambda v: [pre.eq(v)], 'WRITE')

            with m.State('WRITE'):
                m.d.comb += [wr.addr.eq(wptr), wr.data.eq(pre), wr.en.eq(1)]
                m.d.sync += [
                    ph_i.eq(ph_i + FM.RATE_I),
                    ph_ii.eq(ph_ii + FM.RATE_II),
                    mod.eq(0),
                    ch.eq(0),
                ]
                m.next = 'MOD-I'

            with m.State('MOD-I'):
                with m.If(use_i):
                    multiply(tri(ph_i), FM.DEPTH_I, lambda z: [mod.eq(z)], 'MOD-II')
                with m.Else():
                    m.next = 'MOD-II'

            with m.State('MOD-II'):
                with m.If(use_ii):
                    multiply(tri(ph_ii), FM.DEPTH_II, lambda z: [mod.eq(mod + z)], 'TAP')
                with m.Else():
                    m.next = 'TAP'

            # --- per channel: delay tap, interpolation, post-filter, mix -------
            with m.State('TAP'):
                dv = Signal(signed(W))
                m.d.comb += dv.eq(Mux(ch, FM.CENTER - mod, FM.CENTER + mod))
                m.d.sync += [d.eq(dv), frac.eq(dv[:FM.DELAY_FRAC]),
                             cur_post.eq(Mux(ch, post_s[1], post_s[0]))]
                m.next = 'READ-A'

            with m.State('READ-A'):
                n = d >> FM.DELAY_FRAC
                m.d.comb += rd.addr.eq(wptr - n)
                m.next = 'READ-B'

            with m.State('READ-B'):
                n = d >> FM.DELAY_FRAC
                m.d.comb += rd.addr.eq(wptr - n - 1)
                m.d.sync += a.eq(rd.data)
                m.next = 'LATCH-B'

            with m.State('LATCH-B'):
                m.d.sync += b.eq(rd.data)
                m.next = 'INTERP'

            with m.State('INTERP'):
                multiply(sat(b - a), frac << (15 - FM.DELAY_FRAC),
                         lambda z: [tap.eq(a + z)], 'POST')

            with m.State('POST'):
                # one shared post-filter instance; state copied in (TAP) and back (MIX-DRY)
                onepole(FM.G_15K, tap, cur_post, lambda v: [wet.eq(v)], 'MIX-DRY')

            with m.State('MIX-DRY'):
                with m.If(ch):
                    m.d.sync += post_s[1].eq(cur_post)
                with m.Else():
                    m.d.sync += post_s[0].eq(cur_post)
                multiply(FM.DRY, y, lambda z: [dry_t.eq(z)], 'MIX-WET')

            with m.State('MIX-WET'):
                def mix(z):
                    t = Signal(signed(W + 1))
                    m.d.comb += t.eq(Mux(chorus_on, dry_t + z, y))
                    clamped = clamp_bits(m, t, 16)
                    return [outs[0].eq(Mux(ch, outs[0], clamped)),
                            outs[1].eq(Mux(ch, clamped, outs[1]))]
                def next_ch(z):
                    return mix(z) + [ch.eq(1)]
                with m.If(ch == 0):
                    multiply(FM.WET, wet, next_ch, 'TAP')
                with m.Else():
                    multiply(FM.WET, wet, mix, 'EMIT')

            with m.State('EMIT'):
                m.d.comb += [
                    self.o.valid.eq(1),
                    self.o.payload[0].as_value().eq(outs[0]),
                    self.o.payload[1].as_value().eq(outs[1]),
                ]
                with m.If(self.o.ready):
                    m.d.sync += wptr.eq(wptr + 1)
                    m.next = 'IDLE'

        return m
