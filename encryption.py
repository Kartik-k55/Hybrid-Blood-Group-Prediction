"""
AES-256-GCM Encryption and Decryption Utility Module
=====================================================
Provides authenticated encryption for sensitive biometric fingerprint data,
feature vectors, patient records, and biometric image files.

Uses cryptography.hazmat.primitives.ciphers.aead.AESGCM (AES-256 in GCM mode)
which provides confidentiality, integrity, and authenticity assurance.
"""

import os
import json
import base64
import secrets
from typing import Union, Optional, Tuple, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

# Constants
AES_KEY_LENGTH = 32     # 256 bits for AES-256
NONCE_LENGTH = 12       # 96 bits standard nonce for AES-GCM
TAG_LENGTH = 16         # 128 bits authentication tag (handled by AESGCM)


class EncryptionError(Exception):
    """Base exception for encryption/decryption errors."""
    pass


class MissingKeyError(EncryptionError):
    """Raised when AES_KEY environment variable is not found."""
    pass


class InvalidKeyError(EncryptionError):
    """Raised when AES key is invalid in format or length."""
    pass


class DecryptionError(EncryptionError):
    """Raised when decryption or authentication verification fails."""
    pass


def generate_key(encoding: str = 'hex') -> str:
    """
    Generate a cryptographically secure 256-bit (32-byte) key for AES-256.

    Args:
        encoding: 'hex', 'base64', or 'raw'. Defaults to 'hex'.

    Returns:
        Encoded 256-bit key string or bytes.
    """
    raw_key = AESGCM.generate_key(bit_length=256)
    if encoding == 'hex':
        return raw_key.hex()
    elif encoding == 'base64':
        return base64.b64encode(raw_key).decode('utf-8')
    elif encoding == 'raw':
        return raw_key
    else:
        raise ValueError(f"Unsupported encoding '{encoding}'. Choose 'hex', 'base64', or 'raw'.")


def get_aes_key(key: Optional[Union[str, bytes]] = None) -> bytes:
    """
    Retrieve and validate the 32-byte AES-256 key.
    
    If `key` argument is provided, it is validated and used.
    Otherwise, the key is loaded from the `AES_KEY` environment variable.

    Supported key formats:
    - 64-character hexadecimal string
    - 44-character Base64-encoded string
    - 32-byte raw bytes or ASCII string

    Returns:
        32-byte raw key bytes.
    """
    raw_key_val = key if key is not None else os.environ.get('AES_KEY')
    
    if not raw_key_val:
        raise MissingKeyError(
            "AES_KEY is missing. Please set the 'AES_KEY' environment variable "
            "or pass an explicit key. Generate one with `encryption.generate_key()`."
        )

    if isinstance(raw_key_val, bytes):
        if len(raw_key_val) == AES_KEY_LENGTH:
            return raw_key_val
        raw_key_str = raw_key_val.decode('utf-8', errors='ignore').strip()
    else:
        raw_key_str = str(raw_key_val).strip()

    # 1. Check if 64-character hex string
    if len(raw_key_str) == 64:
        try:
            decoded = bytes.fromhex(raw_key_str)
            if len(decoded) == AES_KEY_LENGTH:
                return decoded
        except ValueError:
            pass

    # 2. Check if Base64 string (44 chars with padding)
    if len(raw_key_str) == 44 or raw_key_str.endswith('='):
        try:
            decoded = base64.b64decode(raw_key_str)
            if len(decoded) == AES_KEY_LENGTH:
                return decoded
        except Exception:
            pass

    # 3. Check if raw 32-byte string
    encoded = raw_key_str.encode('utf-8')
    if len(encoded) == AES_KEY_LENGTH:
        return encoded

    raise InvalidKeyError(
        f"Invalid AES_KEY length: got {len(raw_key_str)} characters / {len(encoded)} bytes. "
        f"Expected a 256-bit key (64 hex characters, 44 base64 characters, or 32 raw bytes)."
    )


def encrypt_bytes(
    data: bytes,
    key: Optional[Union[str, bytes]] = None,
    associated_data: Optional[bytes] = None
) -> bytes:
    """
    Encrypt raw bytes using AES-256-GCM with a fresh random 96-bit nonce.

    Args:
        data: Plaintext bytes to encrypt.
        key: Optional 256-bit key (if None, loaded from AES_KEY env var).
        associated_data: Optional authenticated associated data (AAD) for integrity checking.

    Returns:
        Combined binary payload: nonce (12 bytes) + ciphertext & auth tag.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"data must be bytes or bytearray, got {type(data).__name__}")

    raw_key = get_aes_key(key)
    aesgcm = AESGCM(raw_key)
    nonce = secrets.token_bytes(NONCE_LENGTH)
    
    ciphertext = aesgcm.encrypt(nonce, bytes(data), associated_data)
    # Pack nonce + ciphertext (ciphertext already contains 16-byte auth tag at the end)
    return nonce + ciphertext


def decrypt_bytes(
    ciphertext_with_nonce: bytes,
    key: Optional[Union[str, bytes]] = None,
    associated_data: Optional[bytes] = None
) -> bytes:
    """
    Decrypt bytes encrypted with AES-256-GCM, verifying integrity and authentication.

    Args:
        ciphertext_with_nonce: Combined binary payload (12-byte nonce + ciphertext + tag).
        key: Optional 256-bit key (if None, loaded from AES_KEY env var).
        associated_data: Optional authenticated associated data used during encryption.

    Returns:
        Decrypted plaintext bytes.

    Raises:
        DecryptionError: If ciphertext is tampered, corrupted, truncated, or authentication fails.
    """
    if not isinstance(ciphertext_with_nonce, (bytes, bytearray)):
        raise TypeError(f"ciphertext_with_nonce must be bytes, got {type(ciphertext_with_nonce).__name__}")

    if len(ciphertext_with_nonce) < (NONCE_LENGTH + TAG_LENGTH):
        raise DecryptionError(
            f"Ciphertext payload is too short ({len(ciphertext_with_nonce)} bytes). "
            f"Must be at least {NONCE_LENGTH + TAG_LENGTH} bytes."
        )

    raw_key = get_aes_key(key)
    aesgcm = AESGCM(raw_key)

    nonce = ciphertext_with_nonce[:NONCE_LENGTH]
    actual_ciphertext = ciphertext_with_nonce[NONCE_LENGTH:]

    try:
        plaintext = aesgcm.decrypt(nonce, actual_ciphertext, associated_data)
        return plaintext
    except InvalidTag as e:
        raise DecryptionError(
            "Decryption/Authentication failed: Data may have been tampered with, corrupted, "
            "or the wrong AES key/associated data was provided."
        ) from e
    except Exception as e:
        raise DecryptionError(f"Decryption failed: {str(e)}") from e


def encrypt_data(
    data: Union[str, bytes, dict, list],
    key: Optional[Union[str, bytes]] = None,
    associated_data: Optional[Union[str, bytes]] = None
) -> str:
    """
    Encrypt arbitrary data (string, dict, list, or bytes) to a secure Base64-encoded AES-256-GCM string.

    Args:
        data: Plaintext data (string, dict, list, or raw bytes).
        key: Optional 256-bit key (if None, loaded from AES_KEY env var).
        associated_data: Optional associated data for authenticated encryption.

    Returns:
        Base64-encoded string containing (nonce + ciphertext + auth tag).
    """
    if isinstance(data, (dict, list)):
        payload_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
    elif isinstance(data, str):
        payload_bytes = data.encode('utf-8')
    elif isinstance(data, (bytes, bytearray)):
        payload_bytes = bytes(data)
    else:
        raise TypeError(f"Unsupported data type for encryption: {type(data).__name__}")

    aad_bytes = associated_data.encode('utf-8') if isinstance(associated_data, str) else associated_data
    encrypted_blob = encrypt_bytes(payload_bytes, key=key, associated_data=aad_bytes)
    return base64.b64encode(encrypted_blob).decode('ascii')


def decrypt_data(
    encrypted_data: Union[str, bytes],
    key: Optional[Union[str, bytes]] = None,
    associated_data: Optional[Union[str, bytes]] = None,
    return_json: bool = False
) -> Any:
    """
    Decrypt a Base64-encoded AES-256-GCM string or bytes back to original plaintext / JSON object.

    Args:
        encrypted_data: Base64-encoded string or raw encrypted bytes (with nonce).
        key: Optional 256-bit key (if None, loaded from AES_KEY env var).
        associated_data: Optional associated data matching what was used during encryption.
        return_json: If True, parses the decrypted utf-8 string as JSON.

    Returns:
        Decrypted string, parsed JSON dict/list, or raw bytes depending on arguments.
    """
    if isinstance(encrypted_data, str):
        try:
            encrypted_blob = base64.b64decode(encrypted_data.strip())
        except Exception as e:
            raise DecryptionError(f"Invalid base64-encoded ciphertext: {str(e)}") from e
    elif isinstance(encrypted_data, (bytes, bytearray)):
        encrypted_blob = bytes(encrypted_data)
    else:
        raise TypeError(f"encrypted_data must be str or bytes, got {type(encrypted_data).__name__}")

    aad_bytes = associated_data.encode('utf-8') if isinstance(associated_data, str) else associated_data
    decrypted_bytes = decrypt_bytes(encrypted_blob, key=key, associated_data=aad_bytes)

    if return_json:
        try:
            return json.loads(decrypted_bytes.decode('utf-8'))
        except Exception as e:
            raise DecryptionError(f"Failed to parse decrypted data as JSON: {str(e)}") from e

    try:
        return decrypted_bytes.decode('utf-8')
    except UnicodeDecodeError:
        # If it's not valid UTF-8, return raw decrypted bytes
        return decrypted_bytes


def encrypt_file(
    source_path: str,
    destination_path: str,
    key: Optional[Union[str, bytes]] = None,
    associated_data: Optional[Union[str, bytes]] = None
) -> str:
    """
    Encrypt a file from disk (e.g. sensitive fingerprint image or dataset sample) using AES-256-GCM.

    Args:
        source_path: Path to the plaintext source file.
        destination_path: Path to save the encrypted binary file.
        key: Optional 256-bit key.
        associated_data: Optional associated data for authentication.

    Returns:
        Absolute path to the destination encrypted file.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source file not found: {source_path}")

    with open(source_path, 'rb') as f:
        plaintext_bytes = f.read()

    aad_bytes = associated_data.encode('utf-8') if isinstance(associated_data, str) else associated_data
    encrypted_bytes = encrypt_bytes(plaintext_bytes, key=key, associated_data=aad_bytes)

    dest_dir = os.path.dirname(destination_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    with open(destination_path, 'wb') as f:
        f.write(encrypted_bytes)

    return os.path.abspath(destination_path)


def decrypt_file(
    source_path: str,
    destination_path: str,
    key: Optional[Union[str, bytes]] = None,
    associated_data: Optional[Union[str, bytes]] = None
) -> str:
    """
    Decrypt an AES-256-GCM encrypted file back to its original plaintext / image format.

    Args:
        source_path: Path to the encrypted file.
        destination_path: Path to write the decrypted output file.
        key: Optional 256-bit key.
        associated_data: Optional associated data matching what was used during encryption.

    Returns:
        Absolute path to the decrypted file.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Encrypted file not found: {source_path}")

    with open(source_path, 'rb') as f:
        encrypted_bytes = f.read()

    aad_bytes = associated_data.encode('utf-8') if isinstance(associated_data, str) else associated_data
    decrypted_bytes = decrypt_bytes(encrypted_bytes, key=key, associated_data=aad_bytes)

    dest_dir = os.path.dirname(destination_path)
    if dest_dir and not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    with open(destination_path, 'wb') as f:
        f.write(decrypted_bytes)

    return os.path.abspath(destination_path)
