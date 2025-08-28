"""
Encryption Services
==================
Provides envelope encryption, field-level encryption,
and key rotation for data at rest.
"""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class EncryptionConfig:
    """Configuration for encryption services"""
    # KMS settings
    kms_provider: str = "aws"  # aws, azure, gcp, local
    kms_key_id: Optional[str] = None
    
    # Encryption settings
    algorithm: str = "AES-256-GCM"
    enable_envelope: bool = True
    enable_field_level: bool = True
    
    # Key rotation
    enable_rotation: bool = True
    rotation_days: int = 90
    keep_old_keys: int = 3
    
    # Performance
    cache_deks: bool = True
    dek_cache_size: int = 1000
    dek_cache_ttl_minutes: int = 60


@dataclass
class DataEncryptionKey:
    """Data Encryption Key (DEK)"""
    key_id: str
    encrypted_key: bytes  # Encrypted by KEK
    plaintext_key: Optional[bytes] = None  # Cached plaintext
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    def is_expired(self) -> bool:
        """Check if key is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at


@dataclass
class EncryptedData:
    """Encrypted data with metadata"""
    ciphertext: bytes
    nonce: bytes
    key_id: str
    algorithm: str
    encrypted_at: datetime = field(default_factory=datetime.utcnow)
    aad: Optional[bytes] = None  # Additional authenticated data
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for storage"""
        metadata = {
            'key_id': self.key_id,
            'algorithm': self.algorithm,
            'encrypted_at': self.encrypted_at.isoformat(),
            'nonce': base64.b64encode(self.nonce).decode(),
            'aad': base64.b64encode(self.aad).decode() if self.aad else None
        }
        
        # Format: metadata_length(4) + metadata + ciphertext
        metadata_bytes = json.dumps(metadata).encode()
        length_bytes = len(metadata_bytes).to_bytes(4, 'big')
        
        return length_bytes + metadata_bytes + self.ciphertext
        
    @classmethod
    def from_bytes(cls, data: bytes) -> 'EncryptedData':
        """Deserialize from bytes"""
        # Extract metadata length
        metadata_length = int.from_bytes(data[:4], 'big')
        
        # Extract metadata
        metadata_bytes = data[4:4 + metadata_length]
        metadata = json.loads(metadata_bytes.decode())
        
        # Extract ciphertext
        ciphertext = data[4 + metadata_length:]
        
        return cls(
            ciphertext=ciphertext,
            nonce=base64.b64decode(metadata['nonce']),
            key_id=metadata['key_id'],
            algorithm=metadata['algorithm'],
            encrypted_at=datetime.fromisoformat(metadata['encrypted_at']),
            aad=base64.b64decode(metadata['aad']) if metadata.get('aad') else None
        )


class EnvelopeEncryption:
    """
    Envelope encryption using KMS for key encryption keys (KEK)
    and locally generated data encryption keys (DEK).
    """
    
    def __init__(self, config: EncryptionConfig):
        self.config = config
        
        # DEK cache
        self._dek_cache: Dict[str, DataEncryptionKey] = {}
        self._cache_lock = asyncio.Lock()
        
        # Current DEK for encryption
        self._current_dek: Optional[DataEncryptionKey] = None
        
        # KMS client (would be actual KMS client)
        self._kms_client = None
        
    async def initialize(self):
        """Initialize KMS connection"""
        # Would initialize actual KMS client
        # if self.config.kms_provider == "aws":
        #     import boto3
        #     self._kms_client = boto3.client('kms')
        
        # Generate initial DEK
        await self._rotate_dek()
        
        logger.info(f"Envelope encryption initialized with {self.config.kms_provider} KMS")
        
    async def encrypt(self, 
                     plaintext: bytes,
                     aad: Optional[bytes] = None) -> EncryptedData:
        """Encrypt data using envelope encryption"""
        # Get current DEK
        dek = await self._get_current_dek()
        
        # Generate nonce
        nonce = os.urandom(12)  # 96 bits for GCM
        
        # Encrypt with AES-GCM
        aesgcm = AESGCM(dek.plaintext_key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, aad)
        
        return EncryptedData(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=dek.key_id,
            algorithm=self.config.algorithm,
            aad=aad
        )
        
    async def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """Decrypt data"""
        # Get DEK
        dek = await self._get_dek(encrypted_data.key_id)
        
        if not dek:
            raise ValueError(f"DEK not found: {encrypted_data.key_id}")
            
        # Decrypt with AES-GCM
        aesgcm = AESGCM(dek.plaintext_key)
        plaintext = aesgcm.decrypt(
            encrypted_data.nonce,
            encrypted_data.ciphertext,
            encrypted_data.aad
        )
        
        return plaintext
        
    async def _get_current_dek(self) -> DataEncryptionKey:
        """Get current DEK for encryption"""
        async with self._cache_lock:
            if not self._current_dek or self._current_dek.is_expired():
                await self._rotate_dek()
                
            return self._current_dek
            
    async def _get_dek(self, key_id: str) -> Optional[DataEncryptionKey]:
        """Get DEK by ID"""
        # Check cache
        async with self._cache_lock:
            if key_id in self._dek_cache:
                return self._dek_cache[key_id]
                
        # Load from storage and decrypt
        # In production, would load from secure storage
        # For now, generate a deterministic key
        plaintext_key = self._derive_key(key_id.encode())
        
        dek = DataEncryptionKey(
            key_id=key_id,
            encrypted_key=b"encrypted_" + key_id.encode(),
            plaintext_key=plaintext_key
        )
        
        # Cache it
        async with self._cache_lock:
            self._dek_cache[key_id] = dek
            
        return dek
        
    async def _rotate_dek(self):
        """Generate new DEK"""
        # Generate new key
        key_id = f"dek_{datetime.utcnow().timestamp()}"
        plaintext_key = AESGCM.generate_key(bit_length=256)
        
        # Encrypt with KMS (mock)
        encrypted_key = await self._encrypt_with_kms(plaintext_key)
        
        # Create DEK
        dek = DataEncryptionKey(
            key_id=key_id,
            encrypted_key=encrypted_key,
            plaintext_key=plaintext_key,
            expires_at=datetime.utcnow() + timedelta(days=self.config.rotation_days)
        )
        
        # Set as current
        self._current_dek = dek
        
        # Cache it
        async with self._cache_lock:
            self._dek_cache[key_id] = dek
            
            # Cleanup old keys
            if len(self._dek_cache) > self.config.dek_cache_size:
                oldest_key = min(self._dek_cache.keys(), 
                               key=lambda k: self._dek_cache[k].created_at)
                del self._dek_cache[oldest_key]
                
        logger.info(f"Rotated DEK: {key_id}")
        
    async def _encrypt_with_kms(self, plaintext_key: bytes) -> bytes:
        """Encrypt DEK with KMS"""
        # In production, would use actual KMS
        # For now, just prefix it
        return b"kms_encrypted_" + plaintext_key
        
    async def _decrypt_with_kms(self, encrypted_key: bytes) -> bytes:
        """Decrypt DEK with KMS"""
        # In production, would use actual KMS
        # For now, just remove prefix
        if encrypted_key.startswith(b"kms_encrypted_"):
            return encrypted_key[14:]
        return encrypted_key
        
    def _derive_key(self, salt: bytes) -> bytes:
        """Derive key from salt (for demo purposes)"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return kdf.derive(b"demo_password")


class FieldLevelEncryption:
    """
    Field-level encryption for specific sensitive fields.
    Allows querying on non-sensitive fields while protecting PII.
    """
    
    def __init__(self, envelope: EnvelopeEncryption):
        self.envelope = envelope
        
        # Fields to encrypt by default
        self.sensitive_fields = {
            'ssn', 'credit_card', 'bank_account',
            'email', 'phone', 'address',
            'medical_record', 'biometric_data'
        }
        
    async def encrypt_document(self,
                             document: Dict[str, Any],
                             fields_to_encrypt: Optional[List[str]] = None) -> Dict[str, Any]:
        """Encrypt specific fields in a document"""
        encrypted_doc = document.copy()
        
        # Determine fields to encrypt
        if fields_to_encrypt:
            fields = fields_to_encrypt
        else:
            # Auto-detect sensitive fields
            fields = [f for f in document.keys() if f in self.sensitive_fields]
            
        # Encrypt each field
        for field in fields:
            if field in document and document[field] is not None:
                # Convert to bytes
                if isinstance(document[field], str):
                    plaintext = document[field].encode()
                elif isinstance(document[field], (int, float)):
                    plaintext = str(document[field]).encode()
                else:
                    plaintext = json.dumps(document[field]).encode()
                    
                # Encrypt
                encrypted = await self.envelope.encrypt(plaintext)
                
                # Store as base64
                encrypted_doc[field] = {
                    '_encrypted': True,
                    'data': base64.b64encode(encrypted.to_bytes()).decode()
                }
                
        return encrypted_doc
        
    async def decrypt_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt encrypted fields in a document"""
        decrypted_doc = document.copy()
        
        for field, value in document.items():
            if isinstance(value, dict) and value.get('_encrypted'):
                try:
                    # Decode from base64
                    encrypted_bytes = base64.b64decode(value['data'])
                    
                    # Deserialize
                    encrypted_data = EncryptedData.from_bytes(encrypted_bytes)
                    
                    # Decrypt
                    plaintext = await self.envelope.decrypt(encrypted_data)
                    
                    # Restore original type
                    try:
                        # Try JSON first
                        decrypted_doc[field] = json.loads(plaintext.decode())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        # Fall back to string
                        decrypted_doc[field] = plaintext.decode()
                        
                except Exception as e:
                    logger.error(f"Failed to decrypt field {field}: {e}")
                    decrypted_doc[field] = None
                    
        return decrypted_doc


class KeyRotationManager:
    """
    Manages key rotation and re-encryption of data.
    Ensures compliance with key rotation policies.
    """
    
    def __init__(self, config: EncryptionConfig, envelope: EnvelopeEncryption):
        self.config = config
        self.envelope = envelope
        
        # Rotation state
        self._rotation_task: Optional[asyncio.Task] = None
        self._last_rotation: Optional[datetime] = None
        
    async def start(self):
        """Start key rotation scheduler"""
        if self.config.enable_rotation:
            self._rotation_task = asyncio.create_task(self._rotation_loop())
            logger.info(f"Key rotation scheduled every {self.config.rotation_days} days")
            
    async def stop(self):
        """Stop key rotation"""
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass
                
    async def _rotation_loop(self):
        """Background rotation task"""
        while True:
            try:
                # Wait for rotation interval
                await asyncio.sleep(self.config.rotation_days * 24 * 60 * 60)
                
                # Perform rotation
                await self.rotate_keys()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Key rotation error: {e}")
                
    async def rotate_keys(self):
        """Perform key rotation"""
        logger.info("Starting key rotation...")
        
        # Generate new DEK
        await self.envelope._rotate_dek()
        
        # In production, would:
        # 1. Scan all encrypted data
        # 2. Re-encrypt with new key
        # 3. Update key references
        # 4. Mark old keys for deletion after grace period
        
        self._last_rotation = datetime.utcnow()
        logger.info("Key rotation completed")
        
    def get_rotation_status(self) -> Dict[str, Any]:
        """Get key rotation status"""
        return {
            'enabled': self.config.enable_rotation,
            'rotation_days': self.config.rotation_days,
            'last_rotation': self._last_rotation.isoformat() if self._last_rotation else None,
            'next_rotation': (
                (self._last_rotation + timedelta(days=self.config.rotation_days)).isoformat()
                if self._last_rotation else None
            )
        }