"""Field-level authenticated encryption for Protected Health Information (PHI) at rest.

Uses AES-256-GCM (Galois/Counter Mode) authenticated symmetric encryption,
fulfilling DPDP Act 2023 security safeguards for sensitive personal health records.
"""

import base64
import json
import os
from typing import Any
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class PHIEncryptionError(Exception):
    """Raised when encryption or decryption fails."""
    pass


class PHIEncryptor:
    """AES-256-GCM authenticated encryptor for sensitive patient health data at rest."""

    def __init__(self, key: bytes | None = None):
        """Initialize with 256-bit (32 bytes) key or generate/read from environment."""
        if key is None:
            env_key = os.getenv("RETINAGUARD_PHI_ENCRYPTION_KEY")
            if env_key:
                try:
                    self._key = base64.b64decode(env_key)
                except Exception as e:
                    raise PHIEncryptionError(f"Invalid base64 encryption key in env: {e}") from e
            else:
                # Default 32-byte key generated for ephemeral local testing
                self._key = AESGCM.generate_key(bit_length=256)
        else:
            self._key = key

        if len(self._key) != 32:
            raise ValueError(f"AES-256 requires exactly a 32-byte key, got {len(self._key)} bytes.")

        self._aesgcm = AESGCM(self._key)

    @classmethod
    def generate_key_b64(cls) -> str:
        """Generate a random 32-byte AES key and return as base64 string."""
        return base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")

    def encrypt_string(self, plaintext: str, associated_data: str | None = None) -> str:
        """Encrypt plaintext string using a fresh 12-byte nonce.

        Returns base64-encoded string containing [12 bytes nonce + ciphertext + 16 bytes tag].
        """
        try:
            nonce = os.urandom(12)
            ad_bytes = associated_data.encode("utf-8") if associated_data else None
            ciphertext = self._aesgcm.encrypt(nonce, plaintext.encode("utf-8"), ad_bytes)
            combined = nonce + ciphertext
            return base64.b64encode(combined).decode("ascii")
        except Exception as e:
            raise PHIEncryptionError(f"Encryption failed: {e}") from e

    def decrypt_string(self, payload_b64: str, associated_data: str | None = None) -> str:
        """Decrypt base64-encoded [nonce + ciphertext + tag]."""
        try:
            raw = base64.b64decode(payload_b64)
            if len(raw) < 28:  # 12-byte nonce + 16-byte tag min length
                raise PHIEncryptionError("Payload too short to contain valid nonce and tag.")
            nonce = raw[:12]
            ciphertext = raw[12:]
            ad_bytes = associated_data.encode("utf-8") if associated_data else None
            plaintext_bytes = self._aesgcm.decrypt(nonce, ciphertext, ad_bytes)
            return plaintext_bytes.decode("utf-8")
        except Exception as e:
            raise PHIEncryptionError(f"Decryption failed or data tampered: {e}") from e

    def encrypt_json(self, data: dict[str, Any], associated_data: str | None = None) -> str:
        """Serialize dictionary to JSON and encrypt."""
        serialized = json.dumps(data, ensure_ascii=False)
        return self.encrypt_string(serialized, associated_data)

    def decrypt_json(self, payload_b64: str, associated_data: str | None = None) -> dict[str, Any]:
        """Decrypt ciphertext and parse back into JSON dict."""
        decrypted_str = self.decrypt_string(payload_b64, associated_data)
        return json.loads(decrypted_str)

