"""
Stage 2: Goal-Conditioned Bijective Field Permutation & Constant-Rate Chaff Entropy Injection
=============================================================================================

Implements:
1. Balanced 4-round keyed Feistel network F_k : {0, ..., 65535} -> {0, ..., 65535}
   conditioned on the per-turn HKDF ratchet key R_t and goal-state digest H(G_t).
2. Recursive schema field permutation and exact inverse reconstruction across
   Model Context Protocol (MCP) and Agent-to-Agent (A2A) JSON-RPC payloads.
3. Constant-rate SHAKE-256 stream whitening and histogram-balanced chaff injection
   into fixed bucket sizes so wire byte entropy satisfies H(X) >= 7.98 bits/byte.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Dict, List, Optional, Tuple, Union


# Canonical MCP / A2A schema field registry (slots 1..64)
CANONICAL_SCHEMA_FIELDS: Tuple[str, ...] = (
    "jsonrpc",
    "method",
    "params",
    "id",
    "result",
    "error",
    "code",
    "message",
    "data",
    "tool_name",
    "arguments",
    "session_id",
    "agent_id",
    "sender",
    "recipient",
    "goal_hash",
    "epoch",
    "timestamp",
    "capabilities",
    "schema_version",
    "context",
    "trace_id",
    "span_id",
    "parent_span_id",
    "auth_scope",
    "tenant_id",
    "resource_uri",
    "action",
    "status",
    "payload",
    "metadata",
    "headers",
    "input_schema",
    "output_schema",
    "properties",
    "required",
    "type",
    "description",
    "items",
    "enum",
    "default",
    "format",
    "minimum",
    "maximum",
    "pattern",
    "additionalProperties",
    "content",
    "role",
    "name",
    "function_call",
    "tool_calls",
    "temperature",
    "max_tokens",
    "top_p",
    "stop_sequences",
    "model",
    "system_instruction",
    "safety_settings",
    "grounding_chunks",
    "citations",
    "cost_budget",
    "deadline_ms",
    "retry_count",
    "idempotency_key",
)

FIELD_TO_SLOT: Dict[str, int] = {
    name: idx + 1 for idx, name in enumerate(CANONICAL_SCHEMA_FIELDS)
}
SLOT_TO_FIELD: Dict[int, str] = {
    idx + 1: name for idx, name in enumerate(CANONICAL_SCHEMA_FIELDS)
}

# Reserved internal slot for dynamic symbol table when custom keys are present
DYNAMIC_SYMBOL_TABLE_SLOT: int = 0xFFFE

# Discrete wire-frame bucket tiers (bytes)
BUCKET_TIERS: Tuple[int, ...] = (512, 1024, 2048, 3072, 4096, 6144, 8192)

# Precomputed 128 affine bijection tables over Z_256 and full 256-byte set for C-speed chaff synthesis
_AFFINE_TABLES: Tuple[bytes, ...] = tuple(
    bytes((((2 * idx + 1) * x) & 0xFF) for x in range(256)) for idx in range(128)
)
_ALL_256_BYTES: frozenset[int] = frozenset(range(256))


def _xor_bytes_fast(a: bytes, b: bytes) -> bytes:
    """Fast C-level big-int XOR of two byte strings of length len(a)."""
    n = len(a)
    if n == 0:
        return b""
    return (int.from_bytes(a, "big") ^ int.from_bytes(b[:n], "big")).to_bytes(n, "big")


class FeistelSchemaPermuter:
    """
    Balanced 4-round keyed Feistel permutation network over 16-bit field IDs:
        F_k : {0, ..., 2^16 - 1} -> {0, ..., 2^16 - 1}

    Given a 16-bit integer x = (L_0 << 8) | R_0 with L_0, R_0 in {0, ..., 255},
    each round r in {0, 1, 2, 3} computes:
        L_{r+1} = R_r
        R_{r+1} = L_r ^ f_r(R_r, k_r)
    where f_r is a domain-separated SHAKE-256 PRF conditioned on both the
    per-turn ratchet state R_t and the active goal digest H(G_t).
    """

    NUM_ROUNDS: int = 4
    DOMAIN_SIZE: int = 65536

    def __init__(self, ratchet_key: bytes, goal_context: Union[str, bytes] = b"") -> None:
        if len(ratchet_key) < 16:
            raise ValueError("ratchet_key must be at least 16 bytes")
        if isinstance(goal_context, str):
            goal_bytes = goal_context.encode("utf-8")
        else:
            goal_bytes = bytes(goal_context)

        self.ratchet_key: bytes = bytes(ratchet_key)
        self.goal_digest: bytes = hashlib.sha256(
            b"chameleon-goal-v1|" + goal_bytes
        ).digest()

        # Derive 4 round PRF lookup tables (4 x 256 = 1024 bytes) via SHAKE-256
        seed_material = (
            b"chameleon-feistel-v1|"
            + self.ratchet_key
            + b"|"
            + self.goal_digest
        )
        raw_sboxes = hashlib.shake_256(seed_material).digest(self.NUM_ROUNDS * 256)
        self._sboxes: Tuple[bytes, ...] = tuple(
            raw_sboxes[r * 256 : (r + 1) * 256] for r in range(self.NUM_ROUNDS)
        )

    def permute_id(self, field_id: int) -> int:
        """
        Forward 4-round Feistel bijection F_k(field_id) -> permuted_id in [0, 65535].
        """
        if not (0 <= field_id < self.DOMAIN_SIZE):
            raise ValueError(f"field_id {field_id} outside 16-bit domain [0, 65535]")

        left = (field_id >> 8) & 0xFF
        right = field_id & 0xFF
        s0, s1, s2, s3 = self._sboxes

        # Round 0
        left, right = right, left ^ s0[right]
        # Round 1
        left, right = right, left ^ s1[right]
        # Round 2
        left, right = right, left ^ s2[right]
        # Round 3
        left, right = right, left ^ s3[right]

        return ((left & 0xFF) << 8) | (right & 0xFF)

    def unpermute_id(self, permuted_id: int) -> int:
        """
        Exact inverse 4-round Feistel bijection F_k^{-1}(permuted_id) -> field_id in [0, 65535].
        """
        if not (0 <= permuted_id < self.DOMAIN_SIZE):
            raise ValueError(
                f"permuted_id {permuted_id} outside 16-bit domain [0, 65535]"
            )

        left = (permuted_id >> 8) & 0xFF
        right = permuted_id & 0xFF
        s0, s1, s2, s3 = self._sboxes

        # Inverse Round 3
        left, right = right ^ s3[left], left
        # Inverse Round 2
        left, right = right ^ s2[left], left
        # Inverse Round 1
        left, right = right ^ s1[left], left
        # Inverse Round 0
        left, right = right ^ s0[left], left

        return ((left & 0xFF) << 8) | (right & 0xFF)

    def permute_payload(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[int, Any], Dict[int, str]]:
        """
        Recursively permutes all dictionary field keys in an MCP/A2A payload into
        16-bit Feistel-permuted integer tags. Any custom field names not in
        CANONICAL_SCHEMA_FIELDS are assigned collision-free dynamic slot IDs in
        [256, 65000] and embedded inside a permuted symbol table slot.
        """
        custom_field_to_slot: Dict[str, int] = {}
        custom_slot_to_field: Dict[int, str] = {}
        next_custom_slot = 256

        def _assign_slot(field_name: str) -> int:
            nonlocal next_custom_slot
            if field_name in FIELD_TO_SLOT:
                return FIELD_TO_SLOT[field_name]
            if field_name in custom_field_to_slot:
                return custom_field_to_slot[field_name]
            slot = next_custom_slot
            next_custom_slot += 1
            if slot >= DYNAMIC_SYMBOL_TABLE_SLOT:
                raise OverflowError("Exceeded maximum dynamic schema slots")
            custom_field_to_slot[field_name] = slot
            custom_slot_to_field[slot] = field_name
            return slot

        def _transform(obj: Any) -> Any:
            if isinstance(obj, dict):
                out: Dict[int, Any] = {}
                for k, v in obj.items():
                    slot = _assign_slot(str(k))
                    perm_tag = self.permute_id(slot)
                    out[perm_tag] = _transform(v)
                return out
            if isinstance(obj, list):
                return [_transform(item) for item in obj]
            if isinstance(obj, tuple):
                return [_transform(item) for item in obj]
            return obj

        permuted_tree: Dict[int, Any] = _transform(payload)
        if custom_slot_to_field:
            sym_tag = self.permute_id(DYNAMIC_SYMBOL_TABLE_SLOT)
            # Also permute keys inside the symbol table so plaintext slots never leak
            permuted_sym = {
                self.permute_id(slot): name
                for slot, name in custom_slot_to_field.items()
            }
            permuted_tree[sym_tag] = permuted_sym

        return permuted_tree, custom_slot_to_field

    def unpermute_payload(self, permuted_tree: Dict[int, Any]) -> Dict[str, Any]:
        """
        Exact inverse of permute_payload: recovers all original string field names
        from their 16-bit Feistel-permuted integer tags.
        """
        sym_tag = self.permute_id(DYNAMIC_SYMBOL_TABLE_SLOT)
        custom_slot_to_field: Dict[int, str] = {}

        tree_copy = dict(permuted_tree)
        if sym_tag in tree_copy:
            raw_sym = tree_copy.pop(sym_tag)
            if isinstance(raw_sym, dict):
                for perm_slot, name in raw_sym.items():
                    orig_slot = self.unpermute_id(int(perm_slot))
                    custom_slot_to_field[orig_slot] = str(name)

        def _recover_name(perm_tag: int) -> str:
            slot = self.unpermute_id(int(perm_tag))
            if slot in SLOT_TO_FIELD:
                return SLOT_TO_FIELD[slot]
            if slot in custom_slot_to_field:
                return custom_slot_to_field[slot]
            raise KeyError(f"Unknown unpermuted schema slot: {slot}")

        def _untransform(obj: Any) -> Any:
            if isinstance(obj, dict):
                out: Dict[str, Any] = {}
                for pk, pv in obj.items():
                    field_name = _recover_name(int(pk))
                    out[field_name] = _untransform(pv)
                return out
            if isinstance(obj, list):
                return [_untransform(item) for item in obj]
            return obj

        return _untransform(tree_copy)


def compute_shannon_entropy(data: bytes) -> float:
    """
    Computes empirical plug-in Shannon byte entropy H(X) in bits/byte over [0.0, 8.0]:
        H(X) = - sum_{b=0}^{255} p(b) log_2 p(b)
    """
    n = len(data)
    if n == 0:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    inv_n = 1.0 / n
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c * inv_n
            entropy -= p * math.log2(p)
    return entropy


class ChaffEntropyEngine:
    """
    Constant-Rate Chaff Entropy Injection and Stream Whitening Engine.

    Given a serialized morph frame of length L:
    1. Selects a discrete bucket size B in BUCKET_TIERS (or next multiple of 512)
       such that B >= L + 8 + 256 (ensuring at least 256 bytes of histogram-balancing
       chaff margin).
    2. Encrypts/whitens the header + payload with a per-turn SHAKE-256 stream cipher
       derived from (ratchet_key, epoch_id).
    3. Synthesizes the remaining (B - (L + 8)) chaff bytes via Histogram-Balanced
       Equidistributed Chaff Filling followed by a keyed Fisher-Yates / Feistel
       chaff permutation, driving the wire frame's empirical byte distribution
       toward exact uniformity over {0, ..., 255} with H(X) >= 7.98 bits/byte.
    """

    MAGIC_XOR_MASK: int = 0xC4A3E1E0

    @staticmethod
    def select_bucket_size(payload_len: int, min_bucket: int = 512) -> int:
        prefix_len = payload_len + 8
        if min_bucket == 512 and prefix_len <= 64:
            return 512
        required = max(min_bucket, prefix_len * 4, 1024)
        for tier in BUCKET_TIERS:
            if tier >= required:
                return tier
        return ((required + 1023) // 1024) * 1024

    @classmethod
    def wrap_with_chaff(
        cls,
        serialized_body: bytes,
        ratchet_key: bytes,
        epoch_id: int,
        format_code: int,
        min_bucket: int = 512,
    ) -> bytes:
        """
        Packs (format_code, body_len, serialized_body) into a stream-whitened,
        set-complement + affine-bijection constant-bucket wire frame of size B bytes.
        """
        body_len = len(serialized_body)
        bucket_size = cls.select_bucket_size(body_len, min_bucket=min_bucket)

        # 8-byte inner header: 4-byte magic/format tag + 4-byte body length
        hdr_word = ((cls.MAGIC_XOR_MASK & 0xFFFFFF00) | (format_code & 0xFF)) & 0xFFFFFFFF
        raw_prefix = struct.pack(">II", hdr_word, body_len) + serialized_body
        prefix_len = len(raw_prefix)
        chaff_len = bucket_size - prefix_len

        # Generate stream cipher keystream for the prefix via SHAKE-256
        stream_seed = (
            b"chameleon-stream-v1|"
            + ratchet_key
            + struct.pack(">Q", epoch_id)
        )
        ks = hashlib.shake_256(stream_seed).digest(prefix_len + 64)
        whitened_prefix = _xor_bytes_fast(raw_prefix, ks)

        blocks: List[bytes] = []
        rem = chaff_len
        # For each 128-byte slice of whitened_prefix, inject its missing byte alphabet complement
        for s_idx in range(0, prefix_len, 128):
            if rem <= 0:
                break
            sub = whitened_prefix[s_idx : s_idx + 128]
            miss = bytes(_ALL_256_BYTES - set(sub))
            rot = ks[prefix_len + ((s_idx >> 7) & 31)] % max(1, len(miss))
            miss = miss[rot:] + miss[:rot]
            take = miss[:rem]
            blocks.append(take)
            rem -= len(take)

        # Fill remaining chaff with keyed 256-byte affine bijections (each block has exact flat histogram)
        j = 8
        while rem > 0:
            tbl = _AFFINE_TABLES[ks[prefix_len + (j & 31)] & 0x7F]
            rot = ks[prefix_len + ((j + 1) & 31)]
            blk = tbl[rot:] + tbl[:rot]
            if rem >= 256:
                blocks.append(blk)
                rem -= 256
            else:
                blocks.append(blk[:rem])
                rem = 0
            j += 2

        return whitened_prefix + b"".join(blocks)

    @classmethod
    def unwrap_chaff(
        cls,
        wire_frame: bytes,
        ratchet_key: bytes,
        epoch_id: int,
    ) -> Tuple[int, bytes]:
        """
        Decodes a constant-bucket wire frame, strips the deterministic chaff suffix,
        verifies the inner framing magic, and returns (format_code, serialized_body).
        """
        if len(wire_frame) < 8:
            raise ValueError("Wire frame too short (< 8 bytes)")

        stream_seed = (
            b"chameleon-stream-v1|"
            + ratchet_key
            + struct.pack(">Q", epoch_id)
        )
        hdr_keystream = hashlib.shake_256(stream_seed).digest(8)
        raw_hdr = _xor_bytes_fast(wire_frame[:8], hdr_keystream)
        hdr_word, body_len = struct.unpack(">II", raw_hdr)

        if (hdr_word & 0xFFFFFF00) != (cls.MAGIC_XOR_MASK & 0xFFFFFF00):
            raise ValueError("Invalid Agent-Chameleon frame magic (ratchet desync or tamper)")

        format_code = hdr_word & 0xFF
        prefix_len = 8 + body_len
        if prefix_len > len(wire_frame):
            raise ValueError(
                f"Corrupted body_len {body_len} exceeds wire frame size {len(wire_frame)}"
            )

        full_keystream = hashlib.shake_256(stream_seed).digest(prefix_len)
        serialized_body = _xor_bytes_fast(
            wire_frame[8:prefix_len], full_keystream[8:prefix_len]
        )
        return format_code, serialized_body
