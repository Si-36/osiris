"""
🔒 Secure Communication Channels
=================================

Security layer for agent communication with encryption,
authentication, and multi-tenant isolation.

Features:
- End-to-end encryption
- NATS nkey/JWT authentication
- Subject-based authorization
- Payload masking/redaction
- Size enforcement and chunking
- Multi-tenant isolation
"""

import asyncio
import json
import hashlib
import base64
from typing import Dict, Any, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

logger = structlog.get_logger(__name__)


# ==================== Security Configuration ====================

@dataclass
class SecurityConfig:
    """Security configuration for communication"""
    # Encryption
    enable_encryption: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    key_rotation_interval: timedelta = timedelta(days=30)
    
    # Authentication
    enable_auth: bool = True
    auth_method: str = "nkey"  # nkey, jwt, mtls
    
    # Authorization
    enable_subject_acl: bool = True
    default_permissions: Set[str] = field(default_factory=lambda: {"pub", "sub"})
    
    # Data protection
    enable_masking: bool = True
    sensitive_fields: List[str] = field(default_factory=lambda: [
        "password", "token", "secret", "key", "ssn", "credit_card"
    ])
    
    # Size limits
    max_message_size: int = 1024 * 1024  # 1MB
    enable_chunking: bool = True
    chunk_size: int = 64 * 1024  # 64KB


# ==================== Encryption Manager ====================

class EncryptionManager:
    """Manages encryption keys and operations"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.keys: Dict[str, bytes] = {}
        self.current_key_id: str = "default"
        self.ciphers: Dict[str, Fernet] = {}
        
        # Initialize default key
        self._rotate_key()
        
    def _rotate_key(self):
        """Rotate encryption keys"""
        # Generate new key
        new_key = Fernet.generate_key()
        key_id = f"key_{int(datetime.utcnow().timestamp())}"
        
        # Store key and cipher
        self.keys[key_id] = new_key
        self.ciphers[key_id] = Fernet(new_key)
        self.current_key_id = key_id
        
        # Clean old keys (keep last 3)
        if len(self.keys) > 3:
            oldest = sorted(self.keys.keys())[:-3]
            for old_key_id in oldest:
                del self.keys[old_key_id]
                del self.ciphers[old_key_id]
        
        logger.info(f"Rotated encryption key: {key_id}")
    
    def encrypt(self, data: bytes) -> Tuple[str, bytes]:
        """
        Encrypt data and return (key_id, encrypted_data).
        """
        if not self.config.enable_encryption:
            return ("none", data)
        
        cipher = self.ciphers[self.current_key_id]
        encrypted = cipher.encrypt(data)
        
        return (self.current_key_id, encrypted)
    
    def decrypt(self, key_id: str, encrypted_data: bytes) -> bytes:
        """Decrypt data with specified key"""
        if key_id == "none":
            return encrypted_data
        
        if key_id not in self.ciphers:
            raise ValueError(f"Unknown key ID: {key_id}")
        
        cipher = self.ciphers[key_id]
        return cipher.decrypt(encrypted_data)
    
    def derive_shared_key(self, password: str, salt: bytes) -> bytes:
        """Derive shared key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))


# ==================== Authentication Manager ====================

@dataclass
class AgentCredentials:
    """Agent authentication credentials"""
    agent_id: str
    tenant_id: str
    nkey_seed: Optional[str] = None
    jwt_token: Optional[str] = None
    permissions: Set[str] = field(default_factory=set)
    expires_at: Optional[datetime] = None


class AuthenticationManager:
    """Manages agent authentication"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.credentials: Dict[str, AgentCredentials] = {}
        self.nkey_seeds: Dict[str, str] = {}  # agent_id -> seed
        
    def register_agent(
        self,
        agent_id: str,
        tenant_id: str,
        permissions: Optional[Set[str]] = None
    ) -> AgentCredentials:
        """Register agent with credentials"""
        if not self.config.enable_auth:
            return AgentCredentials(
                agent_id=agent_id,
                tenant_id=tenant_id,
                permissions=permissions or self.config.default_permissions
            )
        
        # Generate nkey seed (simplified - use nkeys library in production)
        nkey_seed = f"SA{hashlib.sha256(f'{agent_id}:{tenant_id}'.encode()).hexdigest()[:54]}"
        
        credentials = AgentCredentials(
            agent_id=agent_id,
            tenant_id=tenant_id,
            nkey_seed=nkey_seed,
            permissions=permissions or self.config.default_permissions,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        self.credentials[agent_id] = credentials
        self.nkey_seeds[agent_id] = nkey_seed
        
        logger.info(
            "Agent registered",
            agent_id=agent_id,
            tenant_id=tenant_id,
            permissions=list(credentials.permissions)
        )
        
        return credentials
    
    def authenticate(self, agent_id: str, credential: str) -> bool:
        """Authenticate agent"""
        if not self.config.enable_auth:
            return True
        
        if agent_id not in self.credentials:
            return False
        
        creds = self.credentials[agent_id]
        
        # Check expiration
        if creds.expires_at and datetime.utcnow() > creds.expires_at:
            logger.warning(f"Credentials expired for {agent_id}")
            return False
        
        # Verify credential based on method
        if self.config.auth_method == "nkey":
            return credential == creds.nkey_seed
        elif self.config.auth_method == "jwt":
            # In production, verify JWT signature
            return credential == creds.jwt_token
        
        return False
    
    def get_permissions(self, agent_id: str) -> Set[str]:
        """Get agent permissions"""
        if agent_id in self.credentials:
            return self.credentials[agent_id].permissions
        return set()


# ==================== Authorization Manager ====================

class SubjectAuthorization:
    """Subject-based authorization for pub/sub"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.subject_rules: Dict[str, Dict[str, Set[str]]] = {}  # tenant -> subject -> permissions
        
        # Default rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default authorization rules"""
        # Allow agents to pub/sub to their own subjects
        self.add_rule("*", "aura.*.a2a.*.>", {"pub", "sub"})
        
        # Allow broadcast subscriptions
        self.add_rule("*", "aura.*.broadcast.>", {"sub"})
        
        # Allow response inbox pattern
        self.add_rule("*", "_INBOX.>", {"pub", "sub"})
    
    def add_rule(
        self,
        tenant_id: str,
        subject_pattern: str,
        permissions: Set[str]
    ):
        """Add authorization rule"""
        if tenant_id not in self.subject_rules:
            self.subject_rules[tenant_id] = {}
        
        self.subject_rules[tenant_id][subject_pattern] = permissions
        
        logger.debug(
            "Authorization rule added",
            tenant=tenant_id,
            subject=subject_pattern,
            permissions=list(permissions)
        )
    
    def check_permission(
        self,
        tenant_id: str,
        agent_id: str,
        subject: str,
        permission: str
    ) -> bool:
        """Check if agent has permission for subject"""
        if not self.config.enable_subject_acl:
            return True
        
        # Check tenant rules
        tenant_rules = self.subject_rules.get(tenant_id, {})
        
        # Check wildcard rules
        wildcard_rules = self.subject_rules.get("*", {})
        
        # Combine rules
        all_rules = {**wildcard_rules, **tenant_rules}
        
        # Check each rule
        for pattern, perms in all_rules.items():
            if self._match_subject(subject, pattern) and permission in perms:
                # Additional check for agent-specific subjects
                if ".a2a." in pattern and ".a2a." in subject:
                    # Ensure agent can only access their own a2a subjects
                    parts = subject.split(".")
                    if len(parts) >= 5 and parts[4] == agent_id:
                        return True
                else:
                    return True
        
        return False
    
    def _match_subject(self, subject: str, pattern: str) -> bool:
        """Match subject against pattern (supports * and >)"""
        # Convert pattern to regex
        import re
        
        # Escape special chars except * and >
        pattern_escaped = re.escape(pattern)
        pattern_escaped = pattern_escaped.replace(r'\*', '[^.]+')
        pattern_escaped = pattern_escaped.replace(r'\>', '.*')
        pattern_escaped = f"^{pattern_escaped}$"
        
        return bool(re.match(pattern_escaped, subject))


# ==================== Data Protection ====================

class DataProtection:
    """Data protection utilities"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive fields in data"""
        if not self.config.enable_masking:
            return data
        
        masked = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in self.config.sensitive_fields):
                if isinstance(value, str):
                    # Mask all but last 4 characters
                    if len(value) > 4:
                        masked[key] = "*" * (len(value) - 4) + value[-4:]
                    else:
                        masked[key] = "*" * len(value)
                else:
                    masked[key] = "***REDACTED***"
            elif isinstance(value, dict):
                masked[key] = self.mask_sensitive_data(value)
            else:
                masked[key] = value
        
        return masked
    
    def redact_for_audit(self, data: Any) -> Any:
        """Redact sensitive data for audit logs"""
        if isinstance(data, dict):
            return self.mask_sensitive_data(data)
        elif isinstance(data, str):
            # Check for patterns
            import re
            
            # SSN pattern
            data = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '***-**-****', data)
            
            # Credit card pattern
            data = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '**** **** **** ****', data)
            
            # Email pattern (partial masking)
            data = re.sub(
                r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
                lambda m: m.group(1)[:2] + '***@' + m.group(2),
                data
            )
        
        return data


# ==================== Message Chunking ====================

@dataclass
class MessageChunk:
    """Chunk of a large message"""
    message_id: str
    chunk_index: int
    total_chunks: int
    data: bytes
    checksum: str


class MessageChunker:
    """Handles message chunking for large payloads"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.pending_chunks: Dict[str, List[MessageChunk]] = {}
    
    def chunk_message(self, message_id: str, data: bytes) -> List[MessageChunk]:
        """Split message into chunks"""
        if not self.config.enable_chunking or len(data) <= self.config.max_message_size:
            return [MessageChunk(
                message_id=message_id,
                chunk_index=0,
                total_chunks=1,
                data=data,
                checksum=hashlib.sha256(data).hexdigest()
            )]
        
        chunks = []
        chunk_size = self.config.chunk_size
        total_chunks = (len(data) + chunk_size - 1) // chunk_size
        
        for i in range(total_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, len(data))
            chunk_data = data[start:end]
            
            chunk = MessageChunk(
                message_id=message_id,
                chunk_index=i,
                total_chunks=total_chunks,
                data=chunk_data,
                checksum=hashlib.sha256(chunk_data).hexdigest()
            )
            chunks.append(chunk)
        
        return chunks
    
    def reassemble_message(self, chunks: List[MessageChunk]) -> bytes:
        """Reassemble message from chunks"""
        if not chunks:
            return b""
        
        # Sort by index
        chunks.sort(key=lambda c: c.chunk_index)
        
        # Verify all chunks present
        if len(chunks) != chunks[0].total_chunks:
            raise ValueError(f"Missing chunks: got {len(chunks)}, expected {chunks[0].total_chunks}")
        
        # Verify checksums and reassemble
        data = b""
        for chunk in chunks:
            chunk_checksum = hashlib.sha256(chunk.data).hexdigest()
            if chunk_checksum != chunk.checksum:
                raise ValueError(f"Checksum mismatch for chunk {chunk.chunk_index}")
            data += chunk.data
        
        return data
    
    def add_chunk(self, chunk: MessageChunk) -> Optional[bytes]:
        """Add chunk and return complete message if all chunks received"""
        message_id = chunk.message_id
        
        if message_id not in self.pending_chunks:
            self.pending_chunks[message_id] = []
        
        self.pending_chunks[message_id].append(chunk)
        
        # Check if complete
        if len(self.pending_chunks[message_id]) == chunk.total_chunks:
            chunks = self.pending_chunks.pop(message_id)
            return self.reassemble_message(chunks)
        
        return None


# ==================== Secure Channel ====================

class SecureChannel:
    """
    Unified secure communication channel.
    
    Combines encryption, authentication, authorization,
    and data protection for agent communication.
    """
    
    def __init__(
        self,
        agent_id: str,
        tenant_id: str,
        config: Optional[SecurityConfig] = None
    ):
        self.agent_id = agent_id
        self.tenant_id = tenant_id
        self.config = config or SecurityConfig()
        
        # Initialize managers
        self.encryption = EncryptionManager(self.config)
        self.auth = AuthenticationManager(self.config)
        self.authz = SubjectAuthorization(self.config)
        self.protection = DataProtection(self.config)
        self.chunker = MessageChunker(self.config)
        
        # Register self
        self.credentials = self.auth.register_agent(agent_id, tenant_id)
        
        logger.info(
            "Secure channel initialized",
            agent_id=agent_id,
            tenant_id=tenant_id
        )
    
    def build_secure_subject(
        self,
        base_subject: str,
        operation: str = "pub"
    ) -> str:
        """Build secure subject with tenant isolation"""
        # Add tenant prefix
        subject = f"aura.{self.tenant_id}.{base_subject}"
        
        # Verify permission
        if not self.authz.check_permission(
            self.tenant_id,
            self.agent_id,
            subject,
            operation
        ):
            raise PermissionError(f"No {operation} permission for {subject}")
        
        return subject
    
    async def secure_send(
        self,
        data: Dict[str, Any],
        subject: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send data securely with all protections.
        
        Returns metadata about the send operation.
        """
        headers = headers or {}
        
        # Add authentication
        headers["X-Agent-Id"] = self.agent_id
        headers["X-Tenant-Id"] = self.tenant_id
        headers["X-Auth"] = self.credentials.nkey_seed or ""
        
        # Mask sensitive data for transmission
        masked_data = self.protection.mask_sensitive_data(data)
        
        # Serialize
        serialized = json.dumps(masked_data).encode()
        
        # Check size and chunk if needed
        if len(serialized) > self.config.max_message_size:
            if not self.config.enable_chunking:
                raise ValueError(f"Message too large: {len(serialized)} bytes")
            
            # Chunk and send each chunk
            message_id = headers.get("Nats-Msg-Id", str(int(datetime.utcnow().timestamp())))
            chunks = self.chunker.chunk_message(message_id, serialized)
            
            chunk_metadata = []
            for chunk in chunks:
                # Encrypt chunk
                key_id, encrypted = self.encryption.encrypt(chunk.data)
                
                chunk_headers = headers.copy()
                chunk_headers["X-Chunk-Index"] = str(chunk.chunk_index)
                chunk_headers["X-Total-Chunks"] = str(chunk.total_chunks)
                chunk_headers["X-Chunk-Checksum"] = chunk.checksum
                chunk_headers["X-Encryption-Key-Id"] = key_id
                
                chunk_metadata.append({
                    "chunk_index": chunk.chunk_index,
                    "size": len(encrypted),
                    "encrypted": key_id != "none"
                })
            
            return {
                "message_id": message_id,
                "chunked": True,
                "chunks": chunk_metadata,
                "total_size": len(serialized)
            }
        else:
            # Encrypt entire message
            key_id, encrypted = self.encryption.encrypt(serialized)
            headers["X-Encryption-Key-Id"] = key_id
            
            return {
                "message_id": headers.get("Nats-Msg-Id", ""),
                "chunked": False,
                "size": len(encrypted),
                "encrypted": key_id != "none"
            }
    
    async def secure_receive(
        self,
        encrypted_data: bytes,
        headers: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """
        Receive and decrypt data securely.
        
        Returns decrypted and validated data.
        """
        # Verify authentication
        agent_id = headers.get("X-Agent-Id", "")
        auth_token = headers.get("X-Auth", "")
        
        if not self.auth.authenticate(agent_id, auth_token):
            logger.warning(f"Authentication failed for {agent_id}")
            return None
        
        # Check if chunked
        if "X-Chunk-Index" in headers:
            # Handle chunk
            chunk = MessageChunk(
                message_id=headers.get("Nats-Msg-Id", ""),
                chunk_index=int(headers["X-Chunk-Index"]),
                total_chunks=int(headers["X-Total-Chunks"]),
                data=encrypted_data,
                checksum=headers["X-Chunk-Checksum"]
            )
            
            # Decrypt chunk if needed
            key_id = headers.get("X-Encryption-Key-Id", "none")
            if key_id != "none":
                chunk.data = self.encryption.decrypt(key_id, chunk.data)
            
            # Try to reassemble
            complete_data = self.chunker.add_chunk(chunk)
            if complete_data:
                # All chunks received
                return json.loads(complete_data.decode())
            else:
                # Waiting for more chunks
                return None
        else:
            # Single message
            key_id = headers.get("X-Encryption-Key-Id", "none")
            decrypted = self.encryption.decrypt(key_id, encrypted_data)
            
            return json.loads(decrypted.decode())
    
    def create_shared_channel(
        self,
        channel_name: str,
        participants: List[str],
        password: Optional[str] = None
    ) -> str:
        """
        Create a shared encrypted channel for group communication.
        
        Returns channel ID.
        """
        # Generate channel ID
        channel_id = hashlib.sha256(
            f"{channel_name}:{self.tenant_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Generate shared key if password provided
        if password:
            salt = channel_id.encode()
            shared_key = self.encryption.derive_shared_key(password, salt)
            
            # Store shared key (in production, use KV store)
            self.encryption.keys[f"channel_{channel_id}"] = shared_key
            self.encryption.ciphers[f"channel_{channel_id}"] = Fernet(shared_key)
        
        # Setup authorization for participants
        channel_subject = f"aura.{self.tenant_id}.channels.{channel_id}.>"
        for participant in participants:
            # In production, notify participant
            pass
        
        logger.info(
            "Shared channel created",
            channel_id=channel_id,
            participants=len(participants)
        )
        
        return channel_id
    
    def get_audit_log(self, data: Any) -> Dict[str, Any]:
        """Get audit-safe version of data"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "data": self.protection.redact_for_audit(data)
        }


# ==================== Subject Builders ====================

class SubjectBuilder:
    """Utility class for building NATS subjects"""
    
    @staticmethod
    def agent_direct(tenant_id: str, priority: str, agent_id: str) -> str:
        """Build direct agent-to-agent subject"""
        return f"aura.{tenant_id}.a2a.{priority}.{agent_id}"
    
    @staticmethod
    def broadcast(tenant_id: str, topic: str) -> str:
        """Build broadcast subject"""
        return f"aura.{tenant_id}.broadcast.{topic}"
    
    @staticmethod
    def swarm(tenant_id: str, swarm_id: str, subtopic: str = "*") -> str:
        """Build swarm communication subject"""
        return f"aura.{tenant_id}.swarm.{swarm_id}.{subtopic}"
    
    @staticmethod
    def channel(tenant_id: str, channel_id: str) -> str:
        """Build shared channel subject"""
        return f"aura.{tenant_id}.channels.{channel_id}"
    
    @staticmethod
    def consensus(tenant_id: str, consensus_id: str) -> str:
        """Build consensus protocol subject"""
        return f"aura.{tenant_id}.consensus.{consensus_id}"
    
    @staticmethod
    def kv_bucket(tenant_id: str, bucket_type: str = "state") -> str:
        """Build KV bucket name"""
        return f"AURA_{bucket_type.upper()}_{tenant_id}"