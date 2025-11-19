import json
import os
from dataclasses import dataclass
from typing import Optional

import web3
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dstack_sdk import DstackClient
from eth_account.messages import encode_defunct
from vllm_router.log import init_logger

log = init_logger(__name__)

ED25519 = "ed25519"
ECDSA = "ecdsa"


@dataclass
class SigningContext:
    method: str
    signing_address: str
    signing_address_bytes: bytes
    _ed_private: Optional[Ed25519PrivateKey] = None
    _raw_account: Optional[web3.Account] = None

    def sign(self, content: str) -> str:
        if self.method == ED25519 and self._ed_private:
            signature = self._ed_private.sign(content.encode("utf-8"))
            return signature.hex()
        if self.method == ECDSA and self._raw_account:
            signed_message = self._raw_account.sign_message(
                encode_defunct(text=content)
            )
            return f"0x{signed_message.signature.hex()}"
        raise ValueError("Signing context is not properly initialised")


def _build_report_data(signing_address_bytes: bytes, nonce: bytes) -> bytes:
    """Build TDX report data: [signing_address (padded to 32 bytes) || nonce (32 bytes)]"""
    if not signing_address_bytes:
        raise ValueError("Signing address must be provided")
    if len(signing_address_bytes) > 32:
        raise ValueError("Signing address exceeds 32 bytes")
    if len(nonce) != 32:
        raise ValueError("Nonce must be 32 bytes")
    return signing_address_bytes.ljust(32, b"\x00") + nonce


def _parse_nonce(nonce: Optional[bytes | str]) -> bytes:
    if nonce is None:
        return os.urandom(32)
    if isinstance(nonce, bytes):
        nonce_bytes = nonce
    else:
        try:
            nonce_bytes = bytes.fromhex(nonce)
        except ValueError as exc:
            raise ValueError("Nonce must be hex-encoded") from exc
    if len(nonce_bytes) != 32:
        raise ValueError("Nonce must be 32 bytes")
    return nonce_bytes


def _create_ed25519_context() -> SigningContext:
    private_key = Ed25519PrivateKey.generate()
    public_key_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signing_address = public_key_bytes.hex()
    return SigningContext(
        method=ED25519,
        signing_address=signing_address,
        signing_address_bytes=public_key_bytes,
        _ed_private=private_key,
    )


def _create_ecdsa_context() -> SigningContext:
    w3 = web3.Web3()
    account = w3.eth.account.create()
    signing_address = account.address
    # Use the 20-byte Ethereum address for attestation (standard verification identifier)
    address_bytes = bytes.fromhex(signing_address[2:])  # Remove '0x' prefix
    return SigningContext(
        method=ECDSA,
        signing_address=signing_address,
        signing_address_bytes=address_bytes,
        _raw_account=account,
    )


ecdsa_context = _create_ecdsa_context()
ed25519_context = _create_ed25519_context()


def sign_message(context: SigningContext, content: str) -> str:
    return context.sign(content)


def generate_attestation(
    context: SigningContext, nonce: Optional[bytes | str] = None
) -> dict:
    request_nonce_bytes = _parse_nonce(nonce)
    request_nonce_hex = request_nonce_bytes.hex()

    # Build TDX report data: signing_address || request_nonce
    report_data = _build_report_data(context.signing_address_bytes, request_nonce_bytes)

    client = DstackClient()
    try:
        quote_result = client.get_quote(report_data)
        # Handle case where event_log might be empty or simpler
        try:
            event_log = (
                json.loads(quote_result.event_log) if quote_result.event_log else {}
            )
        except Exception:
            event_log = {}

        intel_quote = quote_result.quote
    except Exception as e:
        log.warning(
            f"Failed to get TDX quote from DstackClient: {e}. Using mock/empty quote."
        )
        # Return empty quote and event log on failure
        intel_quote = ""
        event_log = {}

    info = {}
    try:
        info = client.info().model_dump()
    except Exception:
        pass

    return dict(
        signing_address=context.signing_address,
        signing_algo=context.method,
        request_nonce=request_nonce_hex,
        intel_quote=intel_quote,
        event_log=event_log,
        info=info,
    )
