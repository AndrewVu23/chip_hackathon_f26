"""DRAM geometry, PIM timing, and model-shape configuration.

All quantities that downstream modules depend on are *derived* here, in code,
so that no magic number (in particular the block size in tokens) is hardcoded
anywhere else.

Units convention: bytes for sizes, nanoseconds for times.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


def _is_pow2(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


@dataclass(frozen=True)
class DramGeometry:
    """Geometry of the DRAM region backing the KV pool.

    The unit of all-bank PIM operation is one (pseudo-)channel: an all-bank
    command activates the same row index in every bank of that channel and
    broadcasts one column command, so each command moves
    ``banks_per_channel * burst_bytes`` of operand data.

    All field sizes must be powers of two (the address map is bit-sliced).
    """

    name: str
    channels: int          # pseudo-channels visible to the PIM stack
    bank_groups: int       # bank groups per channel
    banks_per_group: int   # banks per bank group
    row_bytes: int         # row-buffer (page) size per bank, per channel
    burst_bytes: int       # bytes moved by one column access in one bank
    rows_per_bank: int     # rows per bank (sets capacity and the M1 noise floor)
    cacheline_bytes: int = 64  # host cacheline, used by the host-cacheline map

    def __post_init__(self) -> None:
        for f in ("channels", "bank_groups", "banks_per_group", "row_bytes",
                  "burst_bytes", "rows_per_bank", "cacheline_bytes"):
            v = getattr(self, f)
            if not _is_pow2(v):
                raise ValueError(f"{self.name}: {f}={v} must be a power of two")
        if self.row_bytes % self.burst_bytes:
            raise ValueError("row_bytes must be a multiple of burst_bytes")
        if self.cacheline_bytes % self.burst_bytes:
            raise ValueError("cacheline_bytes must be a multiple of burst_bytes")

    @property
    def banks_per_channel(self) -> int:
        return self.bank_groups * self.banks_per_group

    @property
    def cols_per_row(self) -> int:
        """Column (burst) slots per row per bank."""
        return self.row_bytes // self.burst_bytes

    @property
    def rowgroup_bytes(self) -> int:
        """Bytes covered by one row index across all banks of one channel.

        This is the natural alignment quantum for all-bank PIM operation.
        """
        return self.row_bytes * self.banks_per_channel

    @property
    def allbank_cmd_bytes(self) -> int:
        """Bytes moved by a single fully-participating all-bank command."""
        return self.burst_bytes * self.banks_per_channel

    @property
    def capacity_bytes(self) -> int:
        return (self.channels * self.banks_per_channel
                * self.rows_per_bank * self.row_bytes)

    @property
    def total_bursts(self) -> int:
        return self.capacity_bytes // self.burst_bytes


@dataclass(frozen=True)
class PimTiming:
    """All-bank command timing.

    Only the *ratio* tRC/tCCD_ab is load-bearing for the analytical model
    (M3's timing factor). The published measurement this project builds on
    reports nRC 10-11x larger than nCCDAB for decode GEMV on an
    AttAcc-style PIM. Absolute values are indicative and are cross-checked
    against Ramulator 2 in Phase 3.
    """

    tccd_ab_ns: float  # min interval between all-bank column (MAC) commands, row open
    trc_ns: float      # full ACT..PRE row cycle paid on an all-bank row miss

    @property
    def miss_hit_ratio(self) -> float:
        return self.trc_ns / self.tccd_ab_ns


@dataclass(frozen=True)
class ModelShape:
    """Transformer shape parameters that set KV bytes per token.

    We model ONE representative layer. vLLM keeps one physical KV tensor per
    layer, but the block table (and therefore the allocator's placement
    decisions) is shared across layers, so every layer sees the same
    inter-block placement pattern. See LIMITATIONS in README.md.
    """

    name: str
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16/bf16

    @property
    def kv_bytes_per_token(self) -> int:
        """K + V bytes per token for one layer."""
        return 2 * self.kv_heads * self.head_dim * self.dtype_bytes


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

# HBM3-PIM: one stack, 32 pseudo-channels, 16 banks per pseudo-channel
# (4 bank groups x 4), 1KB row per pseudo-channel, 32B per column burst
# (32-bit pseudo-channel x BL8). rows_per_bank=16384 -> 8 GiB stack visible
# to the KV pool. Timing ratio ~10.5 per the AttAcc/Ramulator2 measurement.
HBM3_PIM = DramGeometry(
    name="hbm3-pim",
    channels=32,
    bank_groups=4,
    banks_per_group=4,
    row_bytes=1024,
    burst_bytes=32,
    rows_per_bank=16384,
)

# GDDR6-PIM (AiM-like): 8 channels, 16 banks per channel, 2KB rows,
# 32B bursts (x16 device, BL16). 4 GiB visible.
GDDR6_PIM = DramGeometry(
    name="gddr6-pim",
    channels=8,
    bank_groups=4,
    banks_per_group=4,
    row_bytes=2048,
    burst_bytes=32,
    rows_per_bank=16384,
)

DRAM_PRESETS: dict[str, DramGeometry] = {
    "hbm3-pim": HBM3_PIM,
    "gddr6-pim": GDDR6_PIM,
}

# tCCD_ab ~ 4.3 ns between all-bank MACs with the row open; tRC ~ 45 ns
# full row cycle on a miss -> ratio ~10.5.
DEFAULT_TIMING = PimTiming(tccd_ab_ns=4.3, trc_ns=45.0)

# GQA shape in the Llama-3-8B / Mistral-7B class: 8 KV heads x 128 dim, fp16.
# 4096 KV bytes per token per layer.
LLAMA_GQA_8KV = ModelShape(name="llama-gqa-8kv", kv_heads=8, head_dim=128)

# KV-head-count sensitivity axis (Phase C follow-up): the per-channel block
# slice scales with kv_bytes_per_token, so fewer KV heads => smaller slice
# => worse alignment at a fixed block size. MQA (1 KV head, e.g. Falcon /
# PaLM-style) and classic MHA (32 KV heads, e.g. Llama-2-7B) bracket the
# GQA default.
MQA_1KV = ModelShape(name="mqa-1kv", kv_heads=1, head_dim=128)
MHA_32KV = ModelShape(name="mha-32kv", kv_heads=32, head_dim=128)

MODEL_PRESETS: dict[str, ModelShape] = {
    "llama-gqa-8kv": LLAMA_GQA_8KV,
    "mqa-1kv": MQA_1KV,
    "mha-32kv": MHA_32KV,
}


def shard_kv(geom: DramGeometry, shape: ModelShape, n_shards: int
             ) -> tuple[DramGeometry, ModelShape]:
    """Model KV-head sharding across channels (Phase C robustness axis).

    Published PIM-attention designs do not byte-interleave the KV cache over
    every channel; they assign KV heads to channel groups, so one all-bank
    domain owns a subset of heads AND a subset of channels. With ``n_shards``
    such domains, shard g owns ``channels/n_shards`` channels and
    ``kv_heads/n_shards`` heads, and is a self-contained all-bank system with
    unchanged bank/row geometry. Every shard sees the same block table (the
    allocator's placement decisions are shard-invariant), so simulating one
    shard characterizes all of them.

    Returns the effective (geometry, shape) for one shard. n_shards=1 is the
    fully-interleaved baseline used in Phases 0-2.

    Note the per-channel block slice is invariant under proportional
    sharding (both rowgroup_bytes and kv_bytes_per_token scale by
    1/n_shards), so this sweep tests exactly that prediction rather than
    assuming it.
    """
    if n_shards == 1:
        return geom, shape
    if geom.channels % n_shards or shape.kv_heads % n_shards:
        raise ValueError(f"n_shards={n_shards} must divide channels "
                         f"({geom.channels}) and kv_heads ({shape.kv_heads})")
    g = replace(geom, name=f"{geom.name}-s{n_shards}",
                channels=geom.channels // n_shards)
    s = replace(shape, name=f"{shape.name}-s{n_shards}",
                kv_heads=shape.kv_heads // n_shards)
    return g, s


def derive_block_tokens(geom: DramGeometry, shape: ModelShape,
                        rowgroups_per_block: int = 1,
                        spread_channels: bool = True) -> int:
    """Derive the PIM-natural KV block size in tokens.

    An all-bank PIM unit wants each allocation unit to cover whole row-groups
    (the same row index across every bank of a channel), so that a block never
    straddles a row boundary and every command inside a block is a row hit
    after the first.

    block_bytes = rowgroups_per_block * row_bytes * banks_per_channel
                  * (channels if the linear KV space is interleaved across
                     channels, i.e. the block must cover one row-group in
                     EVERY channel; 1 if each block lives in a single channel)

    block_tokens = block_bytes / kv_bytes_per_token   (rounded up to >= 1)

    For the hbm3-pim preset with the llama-gqa-8kv shape and channel
    interleaving this is 1024*16*32/4096 = 128 tokens — not 16.
    """
    block_bytes = rowgroups_per_block * geom.rowgroup_bytes
    if spread_channels:
        block_bytes *= geom.channels
    return max(1, -(-block_bytes // shape.kv_bytes_per_token))
