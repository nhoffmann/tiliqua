// SPDX-License-Identifier: CERN-OHL-S-2.0
//
// Verilator harness for JunoCore (Juno port): fast offline renders.
// Driven by vl_render.py, which also generates the Verilog and builds this.
//
// Mirrors the amaranth.sim testbench in sim_render.render() exactly:
//   1. apply settings, one clock edge,
//   2. per output sample: apply due events, then clock until o.valid and take
//      the payload at that edge (o.ready is held high).
//
// Input file (text):
//   samples <n>
//   set <port> <value>                     (scalar JunoCore input ports)
//   event <sample> <voice> <gate> <pitch_raw> <velocity_raw> <cutoff_raw>
// Output file: n x 2 int16 (left, right), little endian.
//
// Note: Verilator escapes "__" in port names, so o__valid is o___05Fvalid.

#include "Vjuno_core.h"
#include "verilated.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Per-voice field widths in the flattened array ports (6 voices).
static const int N_VOICES = 6;
static const int PITCH_W = 19;   // PitchToInc.PitchSQ
static const int VEL_W = 16;     // MultiADSR.EnvUQ
static const int CUT_W = 16;     // ASQ

template <class W>
static void put_bits(W &w, int lsb, int width, int64_t value) {
    for (int i = 0; i < width; i++) {
        int b = lsb + i;
        uint32_t mask = 1u << (b % 32);
        if ((value >> i) & 1)
            w[b / 32] |= mask;
        else
            w[b / 32] &= ~mask;
    }
}

struct Event {
    long sample;
    int voice, gate;
    int64_t pitch, vel, cut;
};

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s input.txt output.raw\n", argv[0]);
        return 2;
    }
    Verilated::commandArgs(argc, argv);
    auto *top = new Vjuno_core;

    std::map<std::string, std::function<void(int64_t)>> setters = {
        {"saw_en",         [&](int64_t v) { top->saw_en = v; }},
        {"pulse_en",       [&](int64_t v) { top->pulse_en = v; }},
        {"pw_knob",        [&](int64_t v) { top->pw_knob = v; }},
        {"pwm_lfo",        [&](int64_t v) { top->pwm_lfo = v; }},
        {"vca_gate_mode",  [&](int64_t v) { top->vca_gate_mode = v; }},
        {"lfo_rate_inc",   [&](int64_t v) { top->lfo_rate_inc = v; }},
        {"lfo_delay_step", [&](int64_t v) { top->lfo_delay_step = v; }},
        {"lfo_dco_depth",  [&](int64_t v) { top->lfo_dco_depth = v; }},
        {"lfo_vcf_depth",  [&](int64_t v) { top->lfo_vcf_depth = v; }},
        {"sub_gain",       [&](int64_t v) { top->sub_gain = v; }},
        {"noise_gain",     [&](int64_t v) { top->noise_gain = v; }},
        {"attack_rate",    [&](int64_t v) { top->attack_rate = v; }},
        {"decay_rate",     [&](int64_t v) { top->decay_rate = v; }},
        {"sustain_level",  [&](int64_t v) { top->sustain_level = v; }},
        {"release_rate",   [&](int64_t v) { top->release_rate = v; }},
        {"reso",           [&](int64_t v) { top->reso = v; }},
        {"env_amt",        [&](int64_t v) { top->env_amt = v & 0xFFFF; }},
        {"cutoff_cv",      [&](int64_t v) { top->cutoff_cv = v & 0xFFFF; }},
        {"hpf_mode",       [&](int64_t v) { top->hpf_mode = v; }},
        {"chorus_mode",    [&](int64_t v) { top->chorus_mode = v; }},
    };

    long n_samples = 0;
    std::vector<Event> events;
    std::ifstream in(argv[1]);
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string kind;
        ls >> kind;
        if (kind == "samples") {
            ls >> n_samples;
        } else if (kind == "set") {
            std::string name;
            int64_t v;
            ls >> name >> v;
            auto it = setters.find(name);
            if (it == setters.end()) {
                fprintf(stderr, "unknown port: %s\n", name.c_str());
                return 1;
            }
            it->second(v);
        } else if (kind == "event") {
            Event e;
            ls >> e.sample >> e.voice >> e.gate >> e.pitch >> e.vel >> e.cut;
            events.push_back(e);
        }
    }

    auto edge = [&]() {
        top->clk = 1;
        top->eval();
        top->clk = 0;
        top->eval();
    };

    top->rst = 0;
    top->clk = 0;
    top->o___05Fready = 1;
    top->eval();
    edge();  // settings applied, one clock (as the amaranth testbench)

    std::vector<int16_t> out(2 * n_samples);
    size_t ev = 0;
    for (long i = 0; i < n_samples; i++) {
        while (ev < events.size() && events[ev].sample <= i) {
            const Event &e = events[ev];
            if (e.gate) {
                put_bits(top->voice_pitch_base, e.voice * PITCH_W, PITCH_W, e.pitch);
                put_bits(top->voice_velocity, e.voice * VEL_W, VEL_W, e.vel);
                put_bits(top->voice_cutoff_base, e.voice * CUT_W, CUT_W, e.cut);
            }
            if (e.gate)
                top->voice_gates |= (1u << e.voice);
            else
                top->voice_gates &= ~(1u << e.voice);
            ev++;
        }
        top->eval();
        while (!top->o___05Fvalid)
            edge();
        uint32_t p = top->o___05Fpayload;
        out[2 * i] = static_cast<int16_t>(p & 0xFFFF);
        out[2 * i + 1] = static_cast<int16_t>(p >> 16);
        edge();  // the transfer happens on this edge
    }

    FILE *f = fopen(argv[2], "wb");
    fwrite(out.data(), sizeof(int16_t), out.size(), f);
    fclose(f);
    top->final();
    delete top;
    (void)N_VOICES;
    return 0;
}
