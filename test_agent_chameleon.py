"""
Comprehensive Unit & Property Test Suite for Agent-Chameleon (24 Tests)
=======================================================================

Tests:
- Stage 1: X25519 ECDH session ratchet, HKDF-SHA256 state evolution, and all 4
  polymorphic wire codecs (Protobuf Varint, FlatBuffers Table, RFC 8949 CBOR,
  Canonical Signed JSON-RPC).
- Stage 2: Full 65,536-element bijectivity of the 4-round keyed Feistel network,
  goal-conditioned avalanche, custom symbol table handling, and constant-rate
  SHAKE-256 chaff injection achieving Shannon byte entropy H(X) >= 7.98 bits/byte.
- Stage 3: Non-interactive Fiat-Shamir Schnorr Zero-Knowledge Proof of Knowledge
  (`X-Chameleon-Proof-Nonce`), Ed25519 epoch anchoring, 100% replay rejection,
  MITM tamper rejection, and ratchet rollback safety.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_chameleon import (
    CONTENT_TYPE_HEADER,
    HEADER_BUCKET_TIER,
    HEADER_CONTENT_TYPE,
    HEADER_EPOCH_ID,
    HEADER_PROOF_NONCE,
    ChaffEntropyEngine,
    ChameleonMorphEngine,
    ChameleonWirePacket,
    FeistelSchemaPermuter,
    SchnorrProof,
    SessionRatchet,
    WireFormat,
    ZKHandshakeEngine,
    compute_shannon_entropy,
    decode_canonical_json_rpc,
    decode_cbor_rfc8949,
    decode_flatbuffers_table,
    decode_protobuf_varint,
    encode_canonical_json_rpc,
    encode_cbor_rfc8949,
    encode_flatbuffers_table,
    encode_protobuf_varint,
)


class TestAgentChameleon(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_key = b"0123456789abcdef0123456789abcdef"
        self.sample_mcp_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": 42,
            "params": {
                "tool_name": "execute_sap_journal_entry",
                "session_id": "sess-987654321",
                "agent_id": "agent-finops-01",
                "arguments": {
                    "tenant_id": "acme-global-prod",
                    "cost_budget": 12500.75,
                    "retry_count": 0,
                    "required": True,
                    "items": ["ledger_a", "ledger_b", 100, -45, None, False],
                },
            },
        }

    def test_01_feistel_full_65536_bijectivity(self) -> None:
        """Verify exact bijectivity of F_k across the entire 16-bit domain [0, 65535]."""
        permuter = FeistelSchemaPermuter(
            self.sample_key, goal_context="goal:sap_reconciliation_v1"
        )
        seen_outputs = set()
        for x in range(65536):
            y = permuter.permute_id(x)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 65536)
            seen_outputs.add(y)
            x_rec = permuter.unpermute_id(y)
            self.assertEqual(x_rec, x)
        self.assertEqual(len(seen_outputs), 65536)

    def test_02_feistel_goal_conditioned_avalanche(self) -> None:
        """Verify that changing the goal context changes >99% of permuted field IDs."""
        p1 = FeistelSchemaPermuter(self.sample_key, goal_context="goal:read_invoice")
        p2 = FeistelSchemaPermuter(self.sample_key, goal_context="goal:write_invoice")
        differing = sum(
            1 for x in range(4096) if p1.permute_id(x) != p2.permute_id(x)
        )
        avalanche_ratio = differing / 4096.0
        self.assertGreater(avalanche_ratio, 0.99)

    def test_03_feistel_recursive_payload_with_custom_keys(self) -> None:
        """Verify recursive permutation and recovery with unseen custom schema keys."""
        permuter = FeistelSchemaPermuter(self.sample_key, goal_context="goal:custom")
        payload = dict(self.sample_mcp_payload)
        payload["custom_enterprise_field_alpha"] = {
            "nested_custom_beta": [1, 2, {"deep_custom_gamma": "ok"}]
        }
        permuted_tree, custom_map = permuter.permute_payload(payload)
        self.assertEqual(len(custom_map), 3)
        # Ensure no plaintext key strings remain at top level
        for k in permuted_tree.keys():
            self.assertIsInstance(k, int)
        recovered = permuter.unpermute_payload(permuted_tree)
        self.assertEqual(recovered, payload)

    def test_04_feistel_domain_bounds_validation(self) -> None:
        """Verify ValueError on out-of-range 16-bit field identifiers."""
        permuter = FeistelSchemaPermuter(self.sample_key)
        with self.assertRaises(ValueError):
            permuter.permute_id(-1)
        with self.assertRaises(ValueError):
            permuter.permute_id(65536)
        with self.assertRaises(ValueError):
            permuter.unpermute_id(70000)

    def test_05_protobuf_varint_codec_roundtrip(self) -> None:
        """Verify Format 0 (PROTOBUF_VARINT) round-trip serialization."""
        permuter = FeistelSchemaPermuter(self.sample_key, "goal:pb")
        permuted, _ = permuter.permute_payload(self.sample_mcp_payload)
        encoded = encode_protobuf_varint(permuted)
        decoded = decode_protobuf_varint(encoded)
        recovered = permuter.unpermute_payload(decoded)
        self.assertEqual(recovered, self.sample_mcp_payload)

    def test_06_flatbuffers_table_codec_roundtrip(self) -> None:
        """Verify Format 1 (FLATBUFFERS_TABLE) 4-byte aligned vtable round-trip."""
        permuter = FeistelSchemaPermuter(self.sample_key, "goal:fb")
        permuted, _ = permuter.permute_payload(self.sample_mcp_payload)
        encoded = encode_flatbuffers_table(permuted)
        self.assertEqual(encoded[:4], b"FBCH")
        decoded = decode_flatbuffers_table(encoded)
        recovered = permuter.unpermute_payload(decoded)
        self.assertEqual(recovered, self.sample_mcp_payload)

    def test_07_cbor_rfc8949_codec_roundtrip(self) -> None:
        """Verify Format 2 (CBOR_RFC8949) binary object representation round-trip."""
        permuter = FeistelSchemaPermuter(self.sample_key, "goal:cbor")
        permuted, _ = permuter.permute_payload(self.sample_mcp_payload)
        encoded = encode_cbor_rfc8949(permuted)
        decoded = decode_cbor_rfc8949(encoded)
        recovered = permuter.unpermute_payload(decoded)
        self.assertEqual(recovered, self.sample_mcp_payload)

    def test_08_canonical_json_rpc_codec_and_mac_guard(self) -> None:
        """Verify Format 3 (CANONICAL_JSON_RPC) round-trip and MAC tamper detection."""
        permuter = FeistelSchemaPermuter(self.sample_key, "goal:jsonrpc")
        permuted, _ = permuter.permute_payload(self.sample_mcp_payload)
        encoded = encode_canonical_json_rpc(permuted, self.sample_key)
        decoded = decode_canonical_json_rpc(encoded, self.sample_key)
        recovered = permuter.unpermute_payload(decoded)
        self.assertEqual(recovered, self.sample_mcp_payload)

        # Flip 1 bit in the JSON body and verify MAC rejection
        tampered = bytearray(encoded)
        tampered[2] ^= 0x01
        with self.assertRaises(ValueError):
            decode_canonical_json_rpc(bytes(tampered), self.sample_key)

    def test_09_chaff_entropy_engine_wrap_unwrap(self) -> None:
        """Verify constant-rate chaff wrapping, bucket alignment, and unwrapping."""
        raw_body = b"sample_serialized_payload_bytes_1234567890"
        frame = ChaffEntropyEngine.wrap_with_chaff(
            serialized_body=raw_body,
            ratchet_key=self.sample_key,
            epoch_id=7,
            format_code=int(WireFormat.CBOR_RFC8949),
            min_bucket=512,
        )
        self.assertEqual(len(frame), 512)
        fmt_code, recovered_body = ChaffEntropyEngine.unwrap_chaff(
            wire_frame=frame,
            ratchet_key=self.sample_key,
            epoch_id=7,
        )
        self.assertEqual(fmt_code, int(WireFormat.CBOR_RFC8949))
        self.assertEqual(recovered_body, raw_body)

    def test_10_shannon_byte_entropy_exceeds_7_98_across_all_formats(self) -> None:
        """Verify wire frame Shannon byte entropy H(X) >= 7.98 bits/byte across all 4 formats."""
        c_ratchet, _ = SessionRatchet.establish_x25519_pair()
        engine = ChameleonMorphEngine(c_ratchet, min_bucket=512)
        for fmt in WireFormat:
            pkt = engine.seal_message(
                self.sample_mcp_payload,
                goal_context=f"goal:entropy_test_{fmt.name}",
                override_format=fmt,
            )
            self.assertGreaterEqual(
                pkt.shannon_entropy,
                7.98,
                f"Entropy {pkt.shannon_entropy:.4f} < 7.98 for format {fmt.name}",
            )

    def test_11_x25519_session_ratchet_synchronization_and_format_rotation(self) -> None:
        """Verify paired X25519 ratchets derive identical keys and rotate through all 4 formats."""
        r_client, r_server = SessionRatchet.establish_x25519_pair()
        formats_seen = set()
        for turn in range(16):
            goal = f"goal:turn_{turn}"
            e_c, k_c, f_c = r_client.advance(goal)
            e_s, k_s, f_s = r_server.advance(goal)
            self.assertEqual(e_c, turn)
            self.assertEqual(e_c, e_s)
            self.assertEqual(k_c, k_s)
            self.assertEqual(f_c, f_s)
            formats_seen.add(f_c)
            dummy_frame = k_c + bytes([turn])
            r_client.record_frame_digest(dummy_frame)
            r_server.record_frame_digest(dummy_frame)
        self.assertEqual(len(formats_seen), 4)

    def test_12_end_to_end_40_turn_conversation_roundtrip(self) -> None:
        """Verify 40-turn end-to-end multi-agent swarm session with dynamic goals."""
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice, min_bucket=512)
        bob = ChameleonMorphEngine(r_bob, min_bucket=512)

        for turn in range(40):
            goal = f"swarm_goal_step_{turn}"
            msg = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": turn,
                "params": {
                    "tool_name": f"tool_{turn % 5}",
                    "arguments": {"step": turn, "cost_budget": float(turn) * 1.5},
                },
            }
            wire_pkt = alice.seal_message(msg, goal_context=goal)
            self.assertEqual(wire_pkt.headers[HEADER_CONTENT_TYPE], CONTENT_TYPE_HEADER)
            self.assertIn(HEADER_EPOCH_ID, wire_pkt.headers)
            self.assertIn(HEADER_PROOF_NONCE, wire_pkt.headers)
            self.assertIn(HEADER_BUCKET_TIER, wire_pkt.headers)
            self.assertGreaterEqual(wire_pkt.shannon_entropy, 7.98)

            opened = bob.open_message(wire_pkt, goal_context=goal)
            self.assertEqual(opened, msg)

    def test_13_zk_schnorr_fast_m127_prove_and_verify(self) -> None:
        """Verify Stage 3 fast_m127 Schnorr NIZKPoK generation, wire header encoding, and verification."""
        zk = ZKHandshakeEngine()
        frame = os.urandom(512)
        proof = zk.generate_proof(self.sample_key, epoch_id=10, wire_frame=frame, mode="fast_m127")
        hdr = proof.to_wire_header()
        decoded_proof = SchnorrProof.from_wire_header(hdr)
        self.assertEqual(decoded_proof, proof)
        self.assertTrue(zk.verify_proof(decoded_proof, self.sample_key, frame))

    def test_14_zk_schnorr_full256_prove_and_verify(self) -> None:
        """Verify Stage 3 full256 Schnorr NIZKPoK over 2^255 - 19 prime field."""
        zk = ZKHandshakeEngine()
        frame = os.urandom(512)
        proof = zk.generate_proof(self.sample_key, epoch_id=22, wire_frame=frame, mode="full256")
        hdr = proof.to_wire_header()
        decoded_proof = SchnorrProof.from_wire_header(hdr)
        self.assertEqual(decoded_proof, proof)
        self.assertTrue(zk.verify_proof(decoded_proof, self.sample_key, frame))

    def test_15_ed25519_epoch_anchor_signature_verification(self) -> None:
        """Verify RFC 8032 Ed25519 epoch anchor signing and tamper rejection."""
        zk = ZKHandshakeEngine()
        sig = zk.sign_epoch_anchor(self.sample_key, epoch_id=100)
        self.assertTrue(zk.verify_epoch_anchor(sig, self.sample_key, epoch_id=100))
        self.assertFalse(zk.verify_epoch_anchor(sig, self.sample_key, epoch_id=101))

    def test_16_replay_attack_rejection_and_ratchet_resilience(self) -> None:
        """Verify 100% rejection of replayed packets without desynchronizing receiver ratchet."""
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice)
        bob = ChameleonMorphEngine(r_bob)

        pkt0 = alice.seal_message(self.sample_mcp_payload, goal_context="goal:0")
        recovered0 = bob.open_message(pkt0, goal_context="goal:0")
        self.assertEqual(recovered0, self.sample_mcp_payload)

        # Replay pkt0 -> must raise PermissionError
        with self.assertRaises(PermissionError):
            bob.open_message(pkt0, goal_context="goal:0")

        # Subsequent legitimate packet pkt1 must still succeed cleanly
        pkt1 = alice.seal_message(self.sample_mcp_payload, goal_context="goal:1")
        recovered1 = bob.open_message(pkt1, goal_context="goal:1")
        self.assertEqual(recovered1, self.sample_mcp_payload)

    def test_17_mitm_wire_body_tamper_rejection(self) -> None:
        """Verify that flipping a single bit in the wire body fails ZK proof verification."""
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice)
        bob = ChameleonMorphEngine(r_bob)

        pkt = alice.seal_message(self.sample_mcp_payload, goal_context="goal:secure")
        tampered_body = bytearray(pkt.body)
        tampered_body[100] ^= 0x80
        tampered_pkt = ChameleonWirePacket(
            epoch_id=pkt.epoch_id,
            wire_format=pkt.wire_format,
            headers=dict(pkt.headers),
            body=bytes(tampered_body),
            shannon_entropy=pkt.shannon_entropy,
        )

        with self.assertRaises(PermissionError):
            bob.open_message(tampered_pkt, goal_context="goal:secure")

        # Legitimate original packet still opens because Bob rolled back ratchet on tamper failure
        recovered = bob.open_message(pkt, goal_context="goal:secure")
        self.assertEqual(recovered, self.sample_mcp_payload)

    def test_18_mitm_goal_context_mismatch_rejection(self) -> None:
        """Verify that mismatched goal_context fails ratchet key derivation and ZK verification."""
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice)
        bob = ChameleonMorphEngine(r_bob)

        pkt = alice.seal_message(self.sample_mcp_payload, goal_context="goal:authorized_task")
        with self.assertRaises(PermissionError):
            bob.open_message(pkt, goal_context="goal:hijacked_task")

        # Legitimate goal context succeeds
        recovered = bob.open_message(pkt, goal_context="goal:authorized_task")
        self.assertEqual(recovered, self.sample_mcp_payload)

    def test_19_spoofed_epoch_header_rejection(self) -> None:
        """Verify rejection when X-Chameleon-Epoch-ID is modified on the wire."""
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice)
        bob = ChameleonMorphEngine(r_bob)

        pkt = alice.seal_message(self.sample_mcp_payload, goal_context="goal:epoch_check")
        bad_headers = dict(pkt.headers)
        bad_headers[HEADER_EPOCH_ID] = f"{999:016x}"
        bad_pkt = ChameleonWirePacket(
            epoch_id=999,
            wire_format=pkt.wire_format,
            headers=bad_headers,
            body=pkt.body,
            shannon_entropy=pkt.shannon_entropy,
        )
        with self.assertRaises(PermissionError):
            bob.open_message(bad_pkt, goal_context="goal:epoch_check")

    def test_20_heterogeneous_nested_types_across_all_codecs(self) -> None:
        """Verify bytes, negative ints, floats, booleans, None, lists, and dicts across all 4 formats."""
        complex_payload = {
            "jsonrpc": "2.0",
            "id": -999999999,
            "result": {
                "status": "COMPLETED",
                "temperature": 0.00012345,
                "max_tokens": 32768,
                "grounding_chunks": [
                    {"uri": "gs://bucket/doc1", "score": 0.987, "verified": True},
                    {"uri": "gs://bucket/doc2", "score": -1.25, "verified": False},
                    None,
                ],
            },
        }
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice)
        bob = ChameleonMorphEngine(r_bob)
        for fmt in WireFormat:
            pkt = alice.seal_message(complex_payload, goal_context="goal:complex", override_format=fmt)
            rec = bob.open_message(pkt, goal_context="goal:complex")
            self.assertEqual(rec, complex_payload)

    def test_21_large_payload_multi_tier_bucket_scaling(self) -> None:
        """Verify automatic bucket tier scaling (2048..8192) on larger tool outputs."""
        large_payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "data": "A" * 1400,
                "items": list(range(100)),
            },
        }
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice, min_bucket=512)
        bob = ChameleonMorphEngine(r_bob, min_bucket=512)
        pkt = alice.seal_message(large_payload, goal_context="goal:large")
        self.assertIn(len(pkt.body), (2048, 3072, 4096, 6144, 8192))
        self.assertGreaterEqual(pkt.shannon_entropy, 7.98)
        self.assertEqual(bob.open_message(pkt, goal_context="goal:large"), large_payload)

    def test_22_zk_direct_replay_cache_rejection(self) -> None:
        """Verify ZKHandshakeEngine rejects duplicate (epoch_id, ratchet_tag) pairs directly."""
        zk = ZKHandshakeEngine(replay_window_size=64)
        frame = os.urandom(512)
        proof = zk.generate_proof(self.sample_key, epoch_id=5, wire_frame=frame)
        self.assertTrue(zk.verify_proof(proof, self.sample_key, frame, enforce_replay_check=True))
        # Second verification of identical proof must return False
        self.assertFalse(zk.verify_proof(proof, self.sample_key, frame, enforce_replay_check=True))

    def test_23_zk_ancient_epoch_window_eviction(self) -> None:
        """Verify ZKHandshakeEngine rejects epochs older than replay_window_size."""
        zk = ZKHandshakeEngine(replay_window_size=16)
        frame = os.urandom(512)
        proof_new = zk.generate_proof(self.sample_key, epoch_id=100, wire_frame=frame)
        self.assertTrue(zk.verify_proof(proof_new, self.sample_key, frame))

        proof_ancient = zk.generate_proof(self.sample_key, epoch_id=50, wire_frame=frame)
        self.assertFalse(zk.verify_proof(proof_ancient, self.sample_key, frame))

    def test_24_invalid_content_type_rejection(self) -> None:
        """Verify open_message rejects non-Chameleon Content-Type headers."""
        r_alice, r_bob = SessionRatchet.establish_x25519_pair()
        alice = ChameleonMorphEngine(r_alice)
        bob = ChameleonMorphEngine(r_bob)
        pkt = alice.seal_message(self.sample_mcp_payload)
        bad_headers = dict(pkt.headers)
        bad_headers[HEADER_CONTENT_TYPE] = "application/json"
        bad_pkt = ChameleonWirePacket(
            epoch_id=pkt.epoch_id,
            wire_format=pkt.wire_format,
            headers=bad_headers,
            body=pkt.body,
            shannon_entropy=pkt.shannon_entropy,
        )
        with self.assertRaises(ValueError):
            bob.open_message(bad_pkt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
