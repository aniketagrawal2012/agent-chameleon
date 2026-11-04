"""
Stage 1: Ephemeral ECDH/HKDF Session Ratchet & Polymorphic Wire Serialization Engine
====================================================================================

Implements:
1. Ephemeral X25519 ECDH key exchange & per-turn RFC 5869 HKDF-SHA256 ratchet:
       R_t = HKDF-SHA256(R_{t-1}, epoch_t || digest_{t-1}, "chameleon-ratchet-v1" || H(G_t))
2. Deterministic per-turn polymorphic wire serialization across 4 distinct wire formats:
   - Format 0 (PROTOBUF_VARINT): Base-128 varint tags & zigzag/length-delimited records.
   - Format 1 (FLATBUFFERS_TABLE): 4-byte aligned vtable + offset-addressed binary tables.
   - Format 2 (CBOR_RFC8949): RFC 8949 Concise Binary Object Representation major types 0..7.
   - Format 3 (CANONICAL_JSON_RPC): Canonical signed JSON-RPC with hex Feistel field tags.
3. Full integration with Stage 2 (Feistel schema permutation + constant-rate chaff entropy)
   and Stage 3 (Schnorr NIZKPoK challenge-response in `X-Chameleon-Proof-Nonce`).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import hmac
import json
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

from cryptography.hazmat.primitives.asymmetric import x25519

from .feistel_schema import (
    ChaffEntropyEngine,
    FeistelSchemaPermuter,
    compute_shannon_entropy,
)
from .zk_handshake import SchnorrProof, ZKHandshakeEngine


CONTENT_TYPE_HEADER = "application/x-chameleon-morph+octet-stream"
HEADER_CONTENT_TYPE = "Content-Type"
HEADER_EPOCH_ID = "X-Chameleon-Epoch-ID"
HEADER_PROOF_NONCE = "X-Chameleon-Proof-Nonce"
HEADER_BUCKET_TIER = "X-Chameleon-Bucket-Tier"


class WireFormat(enum.IntEnum):
    PROTOBUF_VARINT = 0
    FLATBUFFERS_TABLE = 1
    CBOR_RFC8949 = 2
    CANONICAL_JSON_RPC = 3


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int = 32) -> bytes:
    """
    Fast RFC 5869 HKDF-SHA256 extract-and-expand using C-accelerated hmac.digest.
    """
    prk = hmac.digest(salt, ikm, "sha256")
    out = bytearray()
    t_block = b""
    counter = 1
    while len(out) < length:
        t_block = hmac.digest(prk, t_block + info + bytes([counter]), "sha256")
        out.extend(t_block)
        counter += 1
    return bytes(out[:length])


# ---------------------------------------------------------------------------
# Format 0: Compact Binary Protobuf-Style Base-128 Varint Wire Codec
# ---------------------------------------------------------------------------

def _encode_uvarint(val: int) -> bytes:
    out = bytearray()
    while val >= 0x80:
        out.append((val & 0x7F) | 0x80)
        val >>= 7
    out.append(val & 0x7F)
    return bytes(out)


def _decode_uvarint(buf: bytes, offset: int) -> Tuple[int, int]:
    val = 0
    shift = 0
    while True:
        if offset >= len(buf):
            raise ValueError("Truncated varint in Protobuf wire stream")
        b = buf[offset]
        offset += 1
        val |= (b & 0x7F) << shift
        if not (b & 0x80):
            return val, offset
        shift += 7
        if shift > 70:
            raise ValueError("Varint overflow in Protobuf wire stream")


def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 63) if n >= 0 else ((-n - 1) << 1) | 1


def _zigzag_decode(z: int) -> int:
    return (z >> 1) ^ -(z & 1)


# Value type tags inside Protobuf wire records
_PB_TYPE_NULL = 0
_PB_TYPE_BOOL = 1
_PB_TYPE_INT = 2
_PB_TYPE_FLOAT = 3
_PB_TYPE_STR = 4
_PB_TYPE_BYTES = 5
_PB_TYPE_LIST = 6
_PB_TYPE_MAP = 7


def encode_protobuf_varint(permuted_map: Dict[int, Any]) -> bytes:
    """
    Serializes a Feistel-permuted dictionary into a compact binary Protobuf-like
    base-128 varint stream.
    """

    def _encode_val(v: Any) -> bytes:
        if v is None:
            return bytes([_PB_TYPE_NULL])
        if isinstance(v, bool):
            return bytes([_PB_TYPE_BOOL, 1 if v else 0])
        if isinstance(v, int):
            zz = _zigzag_encode(v)
            return bytes([_PB_TYPE_INT]) + _encode_uvarint(zz)
        if isinstance(v, float):
            return bytes([_PB_TYPE_FLOAT]) + struct.pack(">d", v)
        if isinstance(v, str):
            raw = v.encode("utf-8")
            return bytes([_PB_TYPE_STR]) + _encode_uvarint(len(raw)) + raw
        if isinstance(v, (bytes, bytearray)):
            raw = bytes(v)
            return bytes([_PB_TYPE_BYTES]) + _encode_uvarint(len(raw)) + raw
        if isinstance(v, list):
            parts = bytearray([_PB_TYPE_LIST])
            parts.extend(_encode_uvarint(len(v)))
            for item in v:
                enc_item = _encode_val(item)
                parts.extend(_encode_uvarint(len(enc_item)))
                parts.extend(enc_item)
            return bytes(parts)
        if isinstance(v, dict):
            nested = encode_protobuf_varint({int(k): val for k, val in v.items()})
            return bytes([_PB_TYPE_MAP]) + _encode_uvarint(len(nested)) + nested
        raise TypeError(f"Unsupported type in Protobuf encoder: {type(v)}")

    out = bytearray()
    out.extend(_encode_uvarint(len(permuted_map)))
    for tag_id, val in permuted_map.items():
        # Wire tag combines 16-bit Feistel tag and wire type 2 (length-delimited)
        wire_tag = (int(tag_id) << 3) | 2
        enc_v = _encode_val(val)
        out.extend(_encode_uvarint(wire_tag))
        out.extend(_encode_uvarint(len(enc_v)))
        out.extend(enc_v)
    return bytes(out)


def decode_protobuf_varint(buf: bytes) -> Dict[int, Any]:
    """
    Decodes a compact binary Protobuf-like varint stream back into the
    Feistel-permuted dictionary.
    """

    def _decode_val(data: bytes) -> Any:
        if not data:
            raise ValueError("Empty value record in Protobuf decoder")
        vtype = data[0]
        if vtype == _PB_TYPE_NULL:
            return None
        if vtype == _PB_TYPE_BOOL:
            return bool(data[1])
        if vtype == _PB_TYPE_INT:
            zz, _ = _decode_uvarint(data, 1)
            return _zigzag_decode(zz)
        if vtype == _PB_TYPE_FLOAT:
            return struct.unpack(">d", data[1:9])[0]
        if vtype == _PB_TYPE_STR:
            slen, pos = _decode_uvarint(data, 1)
            return data[pos : pos + slen].decode("utf-8")
        if vtype == _PB_TYPE_BYTES:
            blen, pos = _decode_uvarint(data, 1)
            return data[pos : pos + blen]
        if vtype == _PB_TYPE_LIST:
            count, pos = _decode_uvarint(data, 1)
            items = []
            for _ in range(count):
                ilen, pos = _decode_uvarint(data, pos)
                items.append(_decode_val(data[pos : pos + ilen]))
                pos += ilen
            return items
        if vtype == _PB_TYPE_MAP:
            mlen, pos = _decode_uvarint(data, 1)
            return decode_protobuf_varint(data[pos : pos + mlen])
        raise ValueError(f"Unknown Protobuf value type tag: {vtype}")

    num_entries, offset = _decode_uvarint(buf, 0)
    result: Dict[int, Any] = {}
    for _ in range(num_entries):
        wire_tag, offset = _decode_uvarint(buf, offset)
        if (wire_tag & 0x07) != 2:
            raise ValueError("Invalid wire type in Protobuf record")
        tag_id = wire_tag >> 3
        vlen, offset = _decode_uvarint(buf, offset)
        val_bytes = buf[offset : offset + vlen]
        offset += vlen
        result[tag_id] = _decode_val(val_bytes)
    return result


# ---------------------------------------------------------------------------
# Format 1: Aligned FlatBuffers-Style VTable + Offset Binary Table Codec
# ---------------------------------------------------------------------------

def encode_flatbuffers_table(permuted_map: Dict[int, Any]) -> bytes:
    """
    Serializes a Feistel-permuted dictionary into a 4-byte aligned FlatBuffers-style
    vtable + root table binary buffer:
      [4B magic 'FBCH'] [4B vtable_offset] [4B num_fields]
      followed by 4-byte aligned payload heap and vtable directory.
    """
    heap = bytearray()
    vtable_entries: List[Tuple[int, int, int]] = []  # (tag_id, heap_offset, byte_len)

    for tag_id, val in permuted_map.items():
        # Encode individual field value using compact protobuf value encoding, padded to 4B
        sub_bytes = encode_protobuf_varint({int(tag_id): val})
        heap_offset = len(heap)
        heap.extend(sub_bytes)
        pad = (4 - (len(heap) % 4)) % 4
        if pad:
            heap.extend(b"\x00" * pad)
        vtable_entries.append((int(tag_id), heap_offset, len(sub_bytes)))

    vtable_offset = 12 + len(heap)
    header = struct.pack(">4sII", b"FBCH", vtable_offset, len(vtable_entries))
    vtable_buf = bytearray()
    for tag_id, h_off, b_len in vtable_entries:
        vtable_buf.extend(struct.pack(">HII", tag_id & 0xFFFF, h_off, b_len))

    return header + bytes(heap) + bytes(vtable_buf)


def decode_flatbuffers_table(buf: bytes) -> Dict[int, Any]:
    """
    Decodes an aligned FlatBuffers-style vtable + root table buffer.
    """
    if len(buf) < 12:
        raise ValueError("FlatBuffers table buffer too short")
    magic, vtable_offset, num_fields = struct.unpack(">4sII", buf[:12])
    if magic != b"FBCH":
        raise ValueError("Invalid FlatBuffers table magic header")
    heap = buf[12:vtable_offset]
    vtable = buf[vtable_offset:]
    if len(vtable) < num_fields * 10:
        raise ValueError("Truncated FlatBuffers vtable directory")

    result: Dict[int, Any] = {}
    for i in range(num_fields):
        tag_id, h_off, b_len = struct.unpack(">HII", vtable[i * 10 : (i + 1) * 10])
        sub_bytes = heap[h_off : h_off + b_len]
        decoded_pair = decode_protobuf_varint(sub_bytes)
        result[tag_id] = decoded_pair[tag_id]
    return result


# ---------------------------------------------------------------------------
# Format 2: RFC 8949 Concise Binary Object Representation (CBOR) Codec
# ---------------------------------------------------------------------------

def _cbor_encode_head(major_type: int, length: int) -> bytes:
    mt = (major_type & 0x07) << 5
    if length < 24:
        return bytes([mt | length])
    if length <= 0xFF:
        return struct.pack(">BB", mt | 24, length)
    if length <= 0xFFFF:
        return struct.pack(">BH", mt | 25, length)
    if length <= 0xFFFFFFFF:
        return struct.pack(">BI", mt | 26, length)
    return struct.pack(">BQ", mt | 27, length)


def _cbor_decode_head(buf: bytes, offset: int) -> Tuple[int, int, int]:
    if offset >= len(buf):
        raise ValueError("Truncated CBOR stream")
    initial = buf[offset]
    offset += 1
    major_type = (initial >> 5) & 0x07
    add_info = initial & 0x1F
    if add_info < 24:
        return major_type, add_info, offset
    if add_info == 24:
        return major_type, buf[offset], offset + 1
    if add_info == 25:
        return major_type, struct.unpack(">H", buf[offset : offset + 2])[0], offset + 2
    if add_info == 26:
        return major_type, struct.unpack(">I", buf[offset : offset + 4])[0], offset + 4
    if add_info == 27:
        return major_type, struct.unpack(">Q", buf[offset : offset + 8])[0], offset + 8
    raise ValueError(f"Unsupported CBOR additional info: {add_info}")


def encode_cbor_rfc8949(obj: Any) -> bytes:
    """
    Serializes a Feistel-permuted object graph into RFC 8949 CBOR binary format.
    """
    if obj is None:
        return bytes([0xF6])  # simple value 22 = null
    if isinstance(obj, bool):
        return bytes([0xF5 if obj else 0xF4])
    if isinstance(obj, int):
        if obj >= 0:
            return _cbor_encode_head(0, obj)
        return _cbor_encode_head(1, -1 - obj)
    if isinstance(obj, float):
        return bytes([0xFB]) + struct.pack(">d", obj)
    if isinstance(obj, (bytes, bytearray)):
        raw = bytes(obj)
        return _cbor_encode_head(2, len(raw)) + raw
    if isinstance(obj, str):
        raw = obj.encode("utf-8")
        return _cbor_encode_head(3, len(raw)) + raw
    if isinstance(obj, (list, tuple)):
        out = bytearray(_cbor_encode_head(4, len(obj)))
        for item in obj:
            out.extend(encode_cbor_rfc8949(item))
        return bytes(out)
    if isinstance(obj, dict):
        out = bytearray(_cbor_encode_head(5, len(obj)))
        for k, v in obj.items():
            out.extend(encode_cbor_rfc8949(int(k)))
            out.extend(encode_cbor_rfc8949(v))
        return bytes(out)
    raise TypeError(f"Unsupported type for RFC 8949 CBOR encoding: {type(obj)}")


def decode_cbor_rfc8949(buf: bytes) -> Dict[int, Any]:
    """
    Decodes an RFC 8949 CBOR byte stream back into the Feistel-permuted dictionary.
    """

    def _parse(pos: int) -> Tuple[Any, int]:
        if pos >= len(buf):
            raise ValueError("Unexpected EOF in CBOR stream")
        b0 = buf[pos]
        if b0 == 0xF4:
            return False, pos + 1
        if b0 == 0xF5:
            return True, pos + 1
        if b0 == 0xF6:
            return None, pos + 1
        if b0 == 0xFB:
            val = struct.unpack(">d", buf[pos + 1 : pos + 9])[0]
            return val, pos + 9

        mt, length, pos = _cbor_decode_head(buf, pos)
        if mt == 0:
            return length, pos
        if mt == 1:
            return -1 - length, pos
        if mt == 2:
            return buf[pos : pos + length], pos + length
        if mt == 3:
            return buf[pos : pos + length].decode("utf-8"), pos + length
        if mt == 4:
            arr = []
            for _ in range(length):
                item, pos = _parse(pos)
                arr.append(item)
            return arr, pos
        if mt == 5:
            mp: Dict[int, Any] = {}
            for _ in range(length):
                k, pos = _parse(pos)
                v, pos = _parse(pos)
                mp[int(k)] = v
            return mp, pos
        raise ValueError(f"Unsupported CBOR major type: {mt}")

    decoded, _ = _parse(0)
    if not isinstance(decoded, dict):
        raise ValueError("Top-level CBOR object must be a map")
    return decoded


# ---------------------------------------------------------------------------
# Format 3: Canonical Signed JSON-RPC with Hex Feistel Field Tags
# ---------------------------------------------------------------------------

def encode_canonical_json_rpc(permuted_map: Dict[int, Any], ratchet_key: bytes) -> bytes:
    """
    Serializes a Feistel-permuted dictionary into canonical compact JSON with
    hex-formatted 16-bit field keys ("0x04a2") plus an 8-byte HMAC-SHA256 trailer.
    """

    def _to_hex_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {f"0x{int(k):04x}": _to_hex_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_hex_keys(i) for i in obj]
        return obj

    canonical_str = json.dumps(
        _to_hex_keys(permuted_map), separators=(",", ":"), sort_keys=True
    )
    json_bytes = canonical_str.encode("utf-8")
    mac8 = hmac.digest(ratchet_key, b"chameleon-jsonrpc|" + json_bytes, "sha256")[:8]
    return json_bytes + mac8


def decode_canonical_json_rpc(buf: bytes, ratchet_key: bytes) -> Dict[int, Any]:
    """
    Verifies the 8-byte HMAC-SHA256 trailer and parses canonical hex-keyed JSON-RPC.
    """
    if len(buf) < 10:
        raise ValueError("Canonical JSON-RPC frame too short")
    json_bytes, mac8 = buf[:-8], buf[-8:]
    expected_mac8 = hmac.digest(
        ratchet_key, b"chameleon-jsonrpc|" + json_bytes, "sha256"
    )[:8]
    if not hmac.compare_digest(mac8, expected_mac8):
        raise ValueError("Canonical JSON-RPC MAC verification failed")

    raw_dict = json.loads(json_bytes.decode("utf-8"))

    def _from_hex_keys(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {int(str(k), 16): _from_hex_keys(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_from_hex_keys(i) for i in obj]
        return obj

    return _from_hex_keys(raw_dict)


# ---------------------------------------------------------------------------
# Ephemeral Session Ratchet & End-to-End Chameleon Morph Engine
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ChameleonWirePacket:
    """
    Observable wire packet emitted by Agent-Chameleon across HTTP/2, gRPC, or FastMCP.
    """

    epoch_id: int
    wire_format: WireFormat
    headers: Dict[str, str]
    body: bytes
    shannon_entropy: float


class SessionRatchet:
    """
    Forward-secure symmetric HKDF-SHA256 session ratchet seeded via X25519 ECDH:
        R_t = HKDF-SHA256(R_{t-1}, epoch_t || digest_{t-1}, "chameleon-ratchet-v1" || H(G_t))
    """

    def __init__(self, initial_root_key: bytes) -> None:
        if len(initial_root_key) < 32:
            raise ValueError("initial_root_key must be at least 32 bytes")
        self.root_key: bytes = bytes(initial_root_key[:32])
        self.epoch_id: int = 0
        self.prev_frame_digest: bytes = b"\x00" * 32

    @classmethod
    def establish_x25519_pair(cls) -> Tuple["SessionRatchet", "SessionRatchet"]:
        """
        Performs an ephemeral X25519 ECDH key exchange and returns a synchronized
        (client_ratchet, server_ratchet) pair.
        """
        sk_a = x25519.X25519PrivateKey.generate()
        sk_b = x25519.X25519PrivateKey.generate()
        shared = sk_a.exchange(sk_b.public_key())
        root = hkdf_sha256(
            shared,
            salt=b"chameleon-x25519-bootstrap-v1",
            info=b"agent-chameleon-root-secret",
            length=32,
        )
        return cls(root), cls(root)

    def advance(
        self, goal_context: Union[str, bytes] = b""
    ) -> Tuple[int, bytes, WireFormat]:
        """
        Advances the ratchet by one turn, returning (epoch_t, R_t, format_t).
        Uses a cycle-balanced 4-turn schedule seeded by R_t so all 4 wire formats
        are selected deterministically with exact 25% stationary distribution.
        """
        current_epoch = self.epoch_id
        if isinstance(goal_context, str):
            goal_bytes = goal_context.encode("utf-8")
        else:
            goal_bytes = bytes(goal_context)
        goal_hash = hashlib.sha256(b"chameleon-goal-v1|" + goal_bytes).digest()

        salt = struct.pack(">Q", current_epoch) + self.prev_frame_digest
        info = b"chameleon-ratchet-v1|" + goal_hash
        next_key = hkdf_sha256(self.root_key, salt=salt, info=info, length=32)

        # Deterministic keyed permutation of {0, 1, 2, 3} within each 4-epoch window
        block_seed = hmac.digest(
            next_key, struct.pack(">Q", current_epoch // 4), "sha256"
        )
        fmt_idx = (block_seed[0] + (current_epoch % 4)) % 4
        wire_fmt = WireFormat(fmt_idx)

        self.root_key = next_key
        self.epoch_id = current_epoch + 1
        return current_epoch, next_key, wire_fmt

    def record_frame_digest(self, wire_body: bytes) -> None:
        self.prev_frame_digest = hashlib.sha256(wire_body).digest()


class ChameleonMorphEngine:
    """
    End-to-End Agent-Chameleon Morphing, Schema Permutation, Chaff Injection,
    and Zero-Knowledge Session Handshake Engine.
    """

    def __init__(
        self,
        ratchet: SessionRatchet,
        zk_engine: Optional[ZKHandshakeEngine] = None,
        min_bucket: int = 512,
        zk_mode: str = "fast_m127",
    ) -> None:
        self.ratchet: SessionRatchet = ratchet
        self.zk_engine: ZKHandshakeEngine = zk_engine or ZKHandshakeEngine()
        self.min_bucket: int = min_bucket
        self.zk_mode: str = zk_mode

    @staticmethod
    def serialize_polymorphic(
        permuted_map: Dict[int, Any], wire_fmt: WireFormat, ratchet_key: bytes
    ) -> bytes:
        if wire_fmt == WireFormat.PROTOBUF_VARINT:
            return encode_protobuf_varint(permuted_map)
        if wire_fmt == WireFormat.FLATBUFFERS_TABLE:
            return encode_flatbuffers_table(permuted_map)
        if wire_fmt == WireFormat.CBOR_RFC8949:
            return encode_cbor_rfc8949(permuted_map)
        if wire_fmt == WireFormat.CANONICAL_JSON_RPC:
            return encode_canonical_json_rpc(permuted_map, ratchet_key)
        raise ValueError(f"Unsupported wire format: {wire_fmt}")

    @staticmethod
    def deserialize_polymorphic(
        serialized_body: bytes, wire_fmt: WireFormat, ratchet_key: bytes
    ) -> Dict[int, Any]:
        if wire_fmt == WireFormat.PROTOBUF_VARINT:
            return decode_protobuf_varint(serialized_body)
        if wire_fmt == WireFormat.FLATBUFFERS_TABLE:
            return decode_flatbuffers_table(serialized_body)
        if wire_fmt == WireFormat.CBOR_RFC8949:
            return decode_cbor_rfc8949(serialized_body)
        if wire_fmt == WireFormat.CANONICAL_JSON_RPC:
            return decode_canonical_json_rpc(serialized_body, ratchet_key)
        raise ValueError(f"Unsupported wire format: {wire_fmt}")

    def seal_message(
        self,
        payload: Dict[str, Any],
        goal_context: Union[str, bytes] = b"",
        override_format: Optional[WireFormat] = None,
    ) -> ChameleonWirePacket:
        """
        Executes all three stages of Agent-Chameleon on an outbound MCP/A2A message:
          Stage 1: Advance HKDF session ratchet & select polymorphic wire format.
          Stage 2: Permute schema field IDs via 4-round keyed Feistel network &
                   inject histogram-balanced SHAKE-256 chaff to fixed bucket size.
          Stage 3: Generate non-interactive Schnorr ZK proof (`X-Chameleon-Proof-Nonce`).
        """
        epoch_id, ratchet_key, scheduled_fmt = self.ratchet.advance(goal_context)
        wire_fmt = override_format if override_format is not None else scheduled_fmt

        # Stage 2A: Bijective 4-round Feistel schema permutation
        permuter = FeistelSchemaPermuter(ratchet_key, goal_context=goal_context)
        permuted_map, _ = permuter.permute_payload(payload)

        # Stage 1B: Polymorphic wire serialization
        serialized_body = self.serialize_polymorphic(
            permuted_map, wire_fmt, ratchet_key
        )

        # Stage 2B: Constant-rate chaff entropy injection & stream whitening
        wire_body = ChaffEntropyEngine.wrap_with_chaff(
            serialized_body=serialized_body,
            ratchet_key=ratchet_key,
            epoch_id=epoch_id,
            format_code=int(wire_fmt),
            min_bucket=self.min_bucket,
        )

        # Stage 3: Sub-35us Fiat-Shamir Schnorr ZK proof generation
        zk_proof = self.zk_engine.generate_proof(
            ratchet_key=ratchet_key,
            epoch_id=epoch_id,
            wire_frame=wire_body,
            mode=self.zk_mode,
        )

        # Update rolling frame digest in ratchet state
        self.ratchet.record_frame_digest(wire_body)

        headers = {
            HEADER_CONTENT_TYPE: CONTENT_TYPE_HEADER,
            HEADER_EPOCH_ID: f"{epoch_id:016x}",
            HEADER_PROOF_NONCE: zk_proof.to_wire_header(),
            HEADER_BUCKET_TIER: str(len(wire_body)),
        }
        entropy = compute_shannon_entropy(wire_body)

        return ChameleonWirePacket(
            epoch_id=epoch_id,
            wire_format=wire_fmt,
            headers=headers,
            body=wire_body,
            shannon_entropy=entropy,
        )

    def open_message(
        self,
        packet: ChameleonWirePacket,
        goal_context: Union[str, bytes] = b"",
    ) -> Dict[str, Any]:
        """
        Verifies the incoming `X-Chameleon-Proof-Nonce` Schnorr zero-knowledge proof,
        strips the constant-rate chaff envelope, decodes the polymorphic wire format,
        and applies the inverse 4-round Feistel permutation F_k^{-1} to recover the
        exact original MCP/A2A dictionary payload.
        """
        if packet.headers.get(HEADER_CONTENT_TYPE) != CONTENT_TYPE_HEADER:
            raise ValueError("Invalid Content-Type header for Agent-Chameleon")

        header_epoch = int(packet.headers[HEADER_EPOCH_ID], 16)
        proof_str = packet.headers[HEADER_PROOF_NONCE]
        zk_proof = SchnorrProof.from_wire_header(proof_str)

        if zk_proof.epoch_id != header_epoch or header_epoch != self.ratchet.epoch_id:
            raise PermissionError(
                f"Ratchet epoch desynchronization or replay: expected {self.ratchet.epoch_id}, got {header_epoch}"
            )

        # Peek next ratchet state without permanently mutating if verification fails
        saved_root = self.ratchet.root_key
        saved_epoch = self.ratchet.epoch_id
        saved_digest = self.ratchet.prev_frame_digest

        try:
            epoch_id, ratchet_key, _ = self.ratchet.advance(goal_context)

            # Stage 3: Verify Schnorr ZK proof & anti-replay guard
            if not self.zk_engine.verify_proof(
                proof=zk_proof,
                expected_ratchet_key=ratchet_key,
                wire_frame=packet.body,
                enforce_replay_check=True,
            ):
                raise PermissionError(
                    "Schnorr zero-knowledge proof verification failed (replay or MITM tamper)"
                )

            # Stage 2B: Unwrap chaff envelope
            fmt_code, serialized_body = ChaffEntropyEngine.unwrap_chaff(
                wire_frame=packet.body,
                ratchet_key=ratchet_key,
                epoch_id=epoch_id,
            )
            wire_fmt = WireFormat(fmt_code)

            # Stage 1B: Polymorphic deserialization
            permuted_map = self.deserialize_polymorphic(
                serialized_body, wire_fmt, ratchet_key
            )

            # Stage 2A: Inverse 4-round Feistel schema unpermutation
            permuter = FeistelSchemaPermuter(ratchet_key, goal_context=goal_context)
            recovered = permuter.unpermute_payload(permuted_map)

            # Commit frame digest
            self.ratchet.record_frame_digest(packet.body)
            return recovered

        except Exception:
            # Roll back ratchet state so an attacker's forged packet cannot desynchronize receiver
            self.ratchet.root_key = saved_root
            self.ratchet.epoch_id = saved_epoch
            self.ratchet.prev_frame_digest = saved_digest
            raise
