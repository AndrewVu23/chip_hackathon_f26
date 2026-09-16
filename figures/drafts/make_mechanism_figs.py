"""Draft mechanism diagrams for redrawing (not part of pimkv.plots).

    .venv/bin/python figures/drafts/make_mechanism_figs.py

Every coordinate is computed from pimkv (AddrMap, coalescer), not placed by
hand, and the command counts are asserted against pimkv.pimmodel.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pimkv.addrmap import AddrMap
from pimkv.config import DEFAULT_TIMING, HBM3_PIM, LLAMA_GQA_8KV
from pimkv.pimmodel import coalesce_channel

OUT = Path(__file__).resolve().parent
G, S, T = HBM3_PIM, LLAMA_GQA_8KV, DEFAULT_TIMING
AM = AddrMap(G, "host-cacheline")
BPT = S.kv_bytes_per_token // G.burst_bytes          # 128 bursts / token
NB = G.banks_per_channel                               # 16

INK, MUTED, SURFACE, LINE = "#0b0b0b", "#898781", "#fcfcfb", "#c3c2b7"
PAGED, PIM = "#2a78d6", "#eb6834"
ACT = "#3b3a36"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                     "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def channel0(table: list[int], bt: int):
    """Channel-0 burst stream (bank, row, col) of a sequence filling ``table``."""
    bpb = bt * BPT
    lb = np.arange(len(table) * bpb)
    phys = np.asarray(table)[lb // bpb] * bpb + lb % bpb
    m = AM.map(phys)
    s = m.ch == 0
    return m.bank[s], m.ro[s], m.co[s]


def commands(table: list[int], bt: int):
    """All-bank commands for channel 0, with participating banks, built the
    way coalesce_channel groups a window (row group; j-th burst per bank ->
    j-th command). Asserted equal to the real coalescer's counts."""
    bank, row, _ = channel0(table, bt)
    cmds = []                                   # (row, set(banks))
    for s in range(0, row.size, 64):
        rw, bw = row[s:s + 64], bank[s:s + 64]
        for r in dict.fromkeys(rw.tolist()):    # rows in first-seen order
            seen: dict[int, int] = {}
            group: list[set[int]] = []
            for b in bw[rw == r].tolist():
                j = seen.get(b, 0)
                seen[b] = j + 1
                if j == len(group):
                    group.append(set())
                group[j].add(b)
            cmds += [(r, g) for g in group]
    acts = sum(1 for i, (r, _) in enumerate(cmds) if i == 0 or r != cmds[i - 1][0])
    n_cmds, n_hits, n_bursts = coalesce_channel(bank, row, banks_per_channel=NB)
    assert (len(cmds), len(cmds) - acts) == (n_cmds, n_hits), (table, bt)
    return cmds, acts, n_bursts


def bracket(ax, x0, x1, y, text, up=False, color=INK):
    h = 0.18 if up else -0.18
    ax.plot([x0, x0, x1, x1], [y - h, y, y, y - h], color=color, lw=1.4)
    ax.text((x0 + x1) / 2, y + (0.12 if up else -0.12), text, ha="center",
            va="bottom" if up else "top", color=color, fontsize=11.5)


# ---------------------------------------------------------------------------
# 1. Address map: the block ID picks the row
# ---------------------------------------------------------------------------
def fig_address_map() -> None:
    fig = plt.figure(figsize=(13, 8.2))
    ax = fig.add_axes([0.03, 0.50, 0.94, 0.46])
    ax.set_xlim(-0.5, 28.5)
    ax.set_ylim(-3.1, 2.3)
    ax.axis("off")

    fills = {"row": PIM, "co": "#e4e1d8", "bg": "#d6d1ee", "ba": "#d6d1ee",
             "ch": "#f3dfa2"}
    names = {"row": "row\n14 bits", "co": "column\n(high 4)", "bg": "bank\ngroup",
             "ba": "bank", "ch": "channel\n5 bits"}
    x = 28
    spans = {}
    for f, nb in AM.fields:                      # LSB first -> draw right to left
        f = "row" if f == "ro" else f
        x0 = x - nb
        ax.add_patch(Rectangle((x0, 0), nb, 1, facecolor=fills[f],
                               edgecolor="white", lw=2))
        label = names[f] if nb > 1 else "col\n(lo)"
        ax.text(x0 + nb / 2, 0.5, label, ha="center", va="center",
                fontsize=11 if nb > 1 else 8.5,
                color="white" if f == "row" else INK,
                fontweight="bold" if f == "row" else "normal")
        spans.setdefault(f, []).append((x0, x))
        x = x0
    ax.text(0, 1.08, "bit 27", ha="left", va="bottom", color=MUTED, fontsize=10)
    ax.text(28, 1.08, "bit 0", ha="right", va="bottom", color=MUTED, fontsize=10)
    ax.text(14, 2.05, "One KV address, sliced by the memory controller "
            "(HBM3-PIM, host-cacheline map)", ha="center", va="bottom",
            fontsize=14, color=INK)
    bracket(ax, 0, 14, 1.45, "changes only every 8 blocks  →  row = block ID ÷ 8",
            up=True, color=PIM)

    bits_tok = int(np.log2(BPT))                                     # 7
    bits_blk = int(np.log2(16 * BPT))                                # 11
    bits_reg = int(np.log2(G.rowgroup_bytes * G.channels // G.burst_bytes))  # 14
    bracket(ax, 28 - bits_tok, 28, -0.25,
            "1 token = 32 ch × 2 banks × 2 cols")
    bracket(ax, 28 - bits_blk, 28, -1.15,
            "1 block (16 tokens) = 32 ch × 16 banks × 4 cols")
    bracket(ax, 28 - bits_reg, 28, -2.05,
            "1 region (8 blocks) = 32 ch × 16 banks × 32 cols = one full row")

    # physical view of one region in one channel
    bx = fig.add_axes([0.08, 0.06, 0.80, 0.36])
    bpb = 16 * BPT
    r = 6
    tints = ["#fbd9c7", "#f4a582"]
    for k in range(8):
        blk = 8 * r + k
        m = AM.map(np.arange(blk * bpb, (blk + 1) * bpb))
        s = m.ch == 0
        assert set(m.ro.tolist()) == {r}
        for b, c in set(zip(m.bank[s].tolist(), m.co[s].tolist())):
            bx.add_patch(Rectangle((c, NB - 1 - b), 1, 1, facecolor=tints[k % 2],
                                   edgecolor="white", lw=0.8))
        cols = sorted(set(m.co[s].tolist()))
        bx.text(cols[0] + len(cols) / 2, NB + 0.35, f"block {blk}",
                ha="center", va="bottom", fontsize=10.5, color=INK)
    bx.set_xlim(0, G.cols_per_row)
    bx.set_ylim(0, NB + 1.6)
    bx.set_xticks(np.arange(0.5, G.cols_per_row, 4), [str(c) for c in range(0, 32, 4)])
    bx.set_yticks([NB - 0.5, 0.5], ["bank 0", "bank 15"])
    bx.tick_params(length=0, colors=MUTED)
    bx.set_xlabel("column", color=MUTED)
    for sp in bx.spines.values():
        sp.set_visible(False)
    bx.set_title(f"Region {r} = row {r} in every bank (one channel shown; "
                 "same in all 32)   ·   blocks 48–55 = 8 × 6 + offset 0–7",
                 fontsize=12.5, color=INK, loc="left", pad=22)
    fig.savefig(OUT / "mech_address_map.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Trace: block table -> row -> commands on a time axis -> row-hit
# ---------------------------------------------------------------------------
def fig_trace() -> None:
    lanes = [("paged (vLLM)", PAGED, [1403, 977, 2210, 58]),
             ("KV-PIMple", PIM, [48, 49, 50, 51])]
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(-0.2, 22.0)
    ax.set_ylim(-1.55, 3.55)
    ax.axis("off")

    t_scale = 9.0 / 240.0          # layout units per ns
    x_time = 9.6
    hdr_y = 3.25
    for x, txt in [(1.0, "block table"), (4.35, "row = ID ÷ 8"),
                   (x_time, "one channel's 16 all-bank commands  (width = time)"),
                   (20.3, "row-hit · time")]:
        ax.text(x, hdr_y, txt, ha="left" if x == x_time else "center",
                va="center", color=MUTED, fontsize=11.5)

    results = []
    for i, (name, color, table) in enumerate(lanes):
        y = 1.75 - 1.75 * i
        cmds, acts, _ = commands(table, 16)
        hits = len(cmds) - acts
        ax.text(-0.1, y + 1.02, name, ha="left", va="bottom", color=color,
                fontsize=13, fontweight="bold")
        for j, b in enumerate(table):
            bx, by = j % 2, 1 - j // 2
            ax.add_patch(Rectangle((0.1 + bx * 0.95, y + by * 0.45), 0.85, 0.38,
                                   facecolor="white", edgecolor=color, lw=1.4))
            ax.text(0.525 + bx * 0.95, y + 0.19 + by * 0.45, str(b), ha="center",
                    va="center", fontsize=11)
            ax.add_patch(Rectangle((3.45 + bx * 0.95, y + by * 0.45), 0.85, 0.38,
                                   facecolor=color, alpha=0.18, edgecolor=color,
                                   lw=1.0))
            ax.text(3.875 + bx * 0.95, y + 0.19 + by * 0.45, str(b // 8),
                    ha="center", va="center", fontsize=11)
        ax.annotate("", xy=(3.3, y + 0.42), xytext=(2.15, y + 0.42),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))
        ax.annotate("", xy=(x_time - 0.25, y + 0.42), xytext=(5.5, y + 0.42),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.3))

        x = x_time
        prev = None
        for r, _ in cmds:
            miss = r != prev
            w = (T.trc_ns if miss else T.tccd_ab_ns) * t_scale
            ax.add_patch(Rectangle((x, y + 0.1), w, 0.64,
                                   facecolor=ACT if miss else color,
                                   edgecolor=SURFACE, lw=0.9))
            if miss:
                ax.text(x + w / 2, y + 0.42, "ACT", ha="center", va="center",
                        color="white", fontsize=10, fontweight="bold")
            x += w
            prev = r
        t_ns = acts * T.trc_ns + hits * T.tccd_ab_ns
        results.append(t_ns)
        ax.text(20.3, y + 0.55, f"{hits}/{len(cmds)} = {hits/len(cmds):.2f}",
                ha="center", va="center", fontsize=13, color=INK)
        ax.text(20.3, y + 0.2, f"{t_ns:.0f} ns", ha="center", va="center",
                fontsize=12, color=MUTED)

    # time axis
    ya = -0.28
    ax.plot([x_time, x_time + 240 * t_scale], [ya, ya], color=LINE, lw=1)
    for tt in range(0, 241, 40):
        xx = x_time + tt * t_scale
        ax.plot([xx, xx], [ya, ya - 0.07], color=LINE, lw=1)
        ax.text(xx, ya - 0.12, f"{tt}", ha="center", va="top", color=MUTED,
                fontsize=9.5)
    ax.text(x_time + 240 * t_scale + 0.45, ya - 0.12, "ns", ha="left", va="top",
            color=MUTED, fontsize=9.5)

    # legend + footnote
    ly = -0.8
    ax.add_patch(Rectangle((0.1, ly - 0.14), 0.35, 0.28, facecolor=ACT))
    ax.text(0.55, ly, f"row change → activate the row in all 16 banks "
            f"({T.trc_ns:.0f} ns)", va="center", fontsize=11)
    ax.add_patch(Rectangle((x_time, ly - 0.14), 0.12, 0.28, facecolor=MUTED))
    ax.text(x_time + 0.25, ly, f"row already open → read ({T.tccd_ab_ns} ns)",
            va="center", fontsize=11)
    ax.text(0.1, ly - 0.45, "full region (8 blocks): 1 ACT per 32 commands → 31/32 = 0.969",
            va="center", fontsize=11, color=MUTED)
    ax.text(11.0, 3.55, "Same 16 reads, same 4 blocks: paged pays an activation "
            f"per block ({results[0] / results[1]:.1f}× the time)",
            ha="center", va="bottom", fontsize=14, color=INK)
    fig.savefig(OUT / "mech_trace.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Sub-row blocks (4 tokens): row misses AND idle banks
# ---------------------------------------------------------------------------
def fig_subrow() -> None:
    lanes = [("paged (vLLM)", PAGED, [1403, 977, 2210, 58]),
             ("KV-PIMple", PIM, [48, 49, 50, 51])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6),
                             gridspec_kw=dict(width_ratios=[8, 4], wspace=0.12))
    for ax, (name, color, table) in zip(axes, lanes):
        cmds, acts, n_bursts = commands(table, 4)
        part = n_bursts / (NB * len(cmds))
        prev = None
        for x, (r, banks) in enumerate(cmds):
            for b in range(NB):
                on = b in banks
                ax.add_patch(Rectangle((x + 0.08, NB - 1 - b + 0.06), 0.84, 0.88,
                                       facecolor=color if on else "white",
                                       edgecolor=color if on else LINE,
                                       lw=0.6, ls="-" if on else (0, (2, 2))))
            miss = r != prev
            ax.add_patch(Rectangle((x + 0.08, NB + 0.2), 0.84, 0.55,
                                   facecolor=ACT if miss else "white",
                                   edgecolor=ACT if miss else LINE, lw=0.8))
            ax.text(x + 0.5, -0.3, str(r), ha="center", va="top",
                    fontsize=10, color=MUTED)
            prev = r
        ax.set_xlim(0, 8)
        ax.set_ylim(-1.3, NB + 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.text(0, NB + 1.6, name, color=color, fontsize=13, fontweight="bold",
                va="bottom")
        ax.text(0, -2.2, f"{len(cmds)} commands · {part * 16:.0f}/16 banks each "
                f"\nbank participation {part:.2f} · row-hit "
                f"{len(cmds) - acts}/{len(cmds)} = {(len(cmds) - acts) / len(cmds):.2f}",
                va="top", fontsize=11.5, color=INK)
    for a in axes:
        a.text(-0.35, NB + 0.47, "ACT", ha="right", va="center", fontsize=9.5,
               color=ACT, fontweight="bold")
        a.text(-0.35, -0.55, "row", ha="right", va="center", fontsize=9.5,
               color=MUTED)
    axes[0].text(-0.35, NB - 0.5, "bank 0", ha="right", va="center",
                 fontsize=9.5, color=MUTED)
    axes[0].text(-0.35, 0.5, "bank 15", ha="right", va="center", fontsize=9.5,
                 color=MUTED)
    fig.suptitle("4-token blocks cover only half the banks: paged idles 8 of 16 "
                 "per command,\nKV-PIMple pairs neighbouring offsets "
                 "(banks 0–7 + 8–15) into full commands",
                 fontsize=13.5, color=INK, y=1.02)
    fig.savefig(OUT / "mech_subrow_blocks.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_address_map()
    fig_trace()
    fig_subrow()
    print("wrote", *sorted(p.name for p in OUT.glob("mech_*.png")))
