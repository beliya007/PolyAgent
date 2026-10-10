"""
A2A 增强模块 - 高级特性

参考 hello-agents 第十章 3.5-3.7 节,实现 A2A 协议的增强特性:
1. A2A Streaming - 流式消息传输 (SendMessageStream)
2. A2A Auth - 基于 Token 的身份认证
3. A2A Secure - 消息签名与完整性校验
4. A2A MultiModal - 多模态消息(文本+文件+图像)

使用示例:
    from protocols.a2a.enhanced import (
        SecureA2AServer, SecureA2AClient,
        StreamingA2AClient, AuthManager, MessageSigner,
    )

    # 1. 认证
    auth = AuthManager()
    token = auth.generate_token("agent_1", scopes=["execute_skill"])
    auth.verify_token(token)

    # 2. 消息签名
    signer = MessageSigner(secret_key="my-secret")
    signature = signer.sign({"message": "hello"})
    signer.verify({"message": "hello"}, signature)

    # 3. 流式通信
    client = StreamingA2AClient("http://localhost:5000")
    async for chunk in client.stream_skill("long_task", "input"):
        print(chunk)
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

# ============================================================
#  1. 认证模块 (Authentication)
# ============================================================
class AuthScope(str, Enum):
    """认证授权范围"""
    READ_AGENT_CARD = "read_agent_card"
    LIST_SKILLS = "list_skills"
    EXECUTE_SKILL = "execute_skill"
    CREATE_TASK = "create_task"
    ADMIN = "admin"


@dataclass
class AuthToken:
    """认证令牌"""
    token: str
    agent_id: str
    scopes: List[str]
    issued_at: float
    expires_at: float

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes or AuthScope.ADMIN.value in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "agent_id": self.agent_id,
            "scopes": self.scopes,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }


class AuthManager:
    """
    A2A 认证管理器

    提供基于 Token 的身份认证,支持授权范围(Scopes)。

    使用示例:
        auth = AuthManager(secret_key="my-secret")
        token = auth.generate_token("agent_1", scopes=["execute_skill"])
        if auth.verify_token(token, required_scope="execute_skill"):
            # 允许执行
            pass
    """

    def __init__(self, secret_key: str = "a2a-default-secret", default_ttl: int = 3600):
        """
        Args:
            secret_key: 用于签名的密钥
            default_ttl: 默认令牌有效期(秒)
        """
        self.secret_key = secret_key.encode("utf-8")
        self.default_ttl = default_ttl
        self._tokens: Dict[str, AuthToken] = {}
        # 内置已签发的 token 池(便于测试)
        self._seed_tokens()

    def _seed_tokens(self):
        """预置一些演示令牌"""
        demo_token = self.generate_token(
            agent_id="demo_agent",
            scopes=[s.value for s in AuthScope],
        )
        # 不打印输出,避免污染日志

    def generate_token(
        self,
        agent_id: str,
        scopes: Optional[List[str]] = None,
        ttl: Optional[int] = None,
    ) -> AuthToken:
        """
        生成认证令牌

        Args:
            agent_id: 智能体 ID
            scopes: 授权范围列表
            ttl: 有效期(秒)

        Returns:
            AuthToken
        """
        scopes = scopes or [AuthScope.READ_AGENT_CARD.value, AuthScope.LIST_SKILLS.value]
        ttl = ttl or self.default_ttl

        issued_at = time.time()
        expires_at = issued_at + ttl

        # 生成 token: base64(agent_id|issued_at|expires_at|signature)
        payload = f"{agent_id}|{int(issued_at)}|{int(expires_at)}|{','.join(scopes)}"
        signature = hmac.new(
            self.secret_key,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8")[:16]

        token_str = f"{base64.urlsafe_b64encode(payload.encode()).decode()}.{signature_b64}"

        token = AuthToken(
            token=token_str,
            agent_id=agent_id,
            scopes=scopes,
            issued_at=issued_at,
            expires_at=expires_at,
        )
        self._tokens[token_str] = token
        return token

    def verify_token(
        self,
        token_str: str,
        required_scope: Optional[str] = None,
    ) -> Optional[AuthToken]:
        """
        验证令牌

        Args:
            token_str: 令牌字符串
            required_scope: 需要的授权范围

        Returns:
            有效时返回 AuthToken,无效返回 None
        """
        token = self._tokens.get(token_str)
        if token is None:
            return None
        if token.is_expired():
            del self._tokens[token_str]
            return None
        if required_scope and not token.has_scope(required_scope):
            return None
        return token

    def revoke_token(self, token_str: str) -> bool:
        """撤销令牌"""
        return self._tokens.pop(token_str, None) is not None

    def list_active_tokens(self) -> List[AuthToken]:
        """列出所有有效令牌(调试用)"""
        active = [t for t in self._tokens.values() if not t.is_expired()]
        # 清理过期令牌
        expired_keys = [
            k for k, t in self._tokens.items() if t.is_expired()
        ]
        for k in expired_keys:
            del self._tokens[k]
        return active


# ============================================================
#  2. 消息签名模块 (MessageSigner)
# ============================================================
@dataclass
class SignedMessage:
    """已签名消息"""
    payload: Dict[str, Any]
    signature: str
    algorithm: str = "HMAC-SHA256"
    timestamp: float = field(default_factory=time.time)
    nonce: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload": self.payload,
            "signature": self.signature,
            "algorithm": self.algorithm,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }


class MessageSigner:
    """
    A2A 消息签名器

    使用 HMAC-SHA256 对消息进行签名,确保消息完整性和来源可信。

    使用示例:
        signer = MessageSigner(secret_key="my-secret")
        signed = signer.sign_message({"skill": "add", "input": "1+2"})
        if signer.verify_message(signed):
            # 消息未被篡改
            pass
    """

    def __init__(self, secret_key: str = "a2a-default-signing-secret"):
        """
        Args:
            secret_key: 签名密钥
        """
        self.secret_key = secret_key.encode("utf-8")

    def _canonicalize(self, payload: Dict[str, Any]) -> str:
        """规范化消息(排序 key 后序列化)"""
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def sign_message(self, payload: Dict[str, Any]) -> SignedMessage:
        """
        对消息进行签名

        Args:
            payload: 消息内容

        Returns:
            SignedMessage
        """
        nonce = secrets.token_hex(8)
        canonical = self._canonicalize(payload) + nonce
        signature = hmac.new(
            self.secret_key,
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return SignedMessage(
            payload=payload,
            signature=signature,
            nonce=nonce,
        )

    def verify_message(
        self,
        signed_msg: SignedMessage,
        max_age_seconds: float = 300.0,
    ) -> bool:
        """
        验证签名

        Args:
            signed_msg: 已签名消息
            max_age_seconds: 消息最大有效时间(防止重放攻击)

        Returns:
            是否有效
        """
        # 检查时间戳
        if (time.time() - signed_msg.timestamp) > max_age_seconds:
            return False

        canonical = self._canonicalize(signed_msg.payload) + signed_msg.nonce
        expected = hmac.new(
            self.secret_key,
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(expected, signed_msg.signature)


# ============================================================
#  3. 多模态消息模块 (MultiModal)
# ============================================================
class ModalityType(str, Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    JSON = "json"


@dataclass
class MultiModalPart:
    """多模态消息片段"""
    modality: ModalityType
    content: Any
    mime_type: str = "text/plain"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "content": self.content,
            "mime_type": self.mime_type,
            "metadata": self.metadata,
        }


@dataclass
class A2AMessage:
    """A2A 多模态消息"""
    role: str = "user"  # user / agent / system
    parts: List[MultiModalPart] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_text(self, text: str) -> "A2AMessage":
        """添加文本片段"""
        self.parts.append(MultiModalPart(
            modality=ModalityType.TEXT,
            content=text,
            mime_type="text/plain",
        ))
        return self

    def add_image(
        self,
        data: str,
        mime_type: str = "image/png",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "A2AMessage":
        """添加图像片段(base64 编码或 URL)"""
        self.parts.append(MultiModalPart(
            modality=ModalityType.IMAGE,
            content=data,
            mime_type=mime_type,
            metadata=metadata or {},
        ))
        return self

    def add_file(
        self,
        data: str,
        mime_type: str = "application/octet-stream",
        filename: str = "",
    ) -> "A2AMessage":
        """添加文件片段(base64 编码或 URL)"""
        meta = {"filename": filename} if filename else {}
        self.parts.append(MultiModalPart(
            modality=ModalityType.FILE,
            content=data,
            mime_type=mime_type,
            metadata=meta,
        ))
        return self

    def add_json(self, data: Any) -> "A2AMessage":
        """添加 JSON 数据"""
        self.parts.append(MultiModalPart(
            modality=ModalityType.JSON,
            content=data,
            mime_type="application/json",
        ))
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        msg = cls(role=data.get("role", "user"))
        msg.metadata = data.get("metadata", {})
        for p in data.get("parts", []):
            msg.parts.append(MultiModalPart(
                modality=ModalityType(p.get("modality", "text")),
                content=p.get("content"),
                mime_type=p.get("mime_type", "text/plain"),
                metadata=p.get("metadata", {}),
            ))
        return msg


# ============================================================
#  4. 流式通信模块 (Streaming)
# ============================================================
@dataclass
class StreamChunk:
    """流式响应块"""
    chunk_id: int
    content: str
    is_final: bool = False
    artifact: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "is_final": self.is_final,
            "artifact": self.artifact,
            "metadata": self.metadata,
        }


class StreamingA2AClient:
    """
    A2A 流式客户端

    支持 SendMessageStream - 长任务分块返回。

    使用示例:
        client = StreamingA2AClient("http://localhost:5000")

        async for chunk in client.stream_skill("long_task", "input"):
            print(f"[{chunk.chunk_id}] {chunk.content}")
            if chunk.is_final:
                break
    """

    def __init__(self, agent_url: str, timeout: float = 60.0):
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout

    async def stream_skill(
        self,
        skill_name: str,
        input_data: Any,
        chunk_size: int = 10,
    ) -> AsyncIterator[StreamChunk]:
        """
        流式调用技能(模拟分块返回)

        Args:
            skill_name: 技能名
            input_data: 输入数据
            chunk_size: 每次返回的块大小

        Yields:
            StreamChunk
        """
        # 模拟流式响应(在真实场景下会通过 SSE/WebSocket)
        mock_result = f"Streaming result for skill '{skill_name}' with input '{input_data}': 这是一个较长的执行结果..."

        for i in range(0, len(mock_result), chunk_size):
            chunk_content = mock_result[i:i + chunk_size]
            is_final = (i + chunk_size) >= len(mock_result)
            yield StreamChunk(
                chunk_id=i // chunk_size,
                content=chunk_content,
                is_final=is_final,
            )
            await asyncio.sleep(0.05)  # 模拟网络延迟

        # 最终块携带完整工件
        yield StreamChunk(
            chunk_id=-1,
            content="",
            is_final=True,
            artifact={
                "name": f"{skill_name}_result",
                "data": mock_result,
                "mimeType": "text/plain",
            },
        )

    async def stream_message(
        self,
        message: A2AMessage,
        skill_name: str = "",
    ) -> AsyncIterator[StreamChunk]:
        """
        流式发送多模态消息

        Args:
            message: 多模态消息
            skill_name: 目标技能名(从消息 metadata 中推断)

        Yields:
            StreamChunk
        """
        skill = skill_name or message.metadata.get("skill", "default_skill")
        text_parts = [p.content for p in message.parts if p.modality == ModalityType.TEXT]
        input_data = " ".join(text_parts) if text_parts else message.to_dict()

        async for chunk in self.stream_skill(skill, input_data):
            yield chunk


# ============================================================
#  5. 增强型 A2A Server
# ============================================================
class SecureA2AServer:
    """
    安全增强版 A2A Server

    在标准 A2AServer 基础上集成:
    - 认证(AuthManager)
    - 消息签名(MessageSigner)
    - 多模态消息支持

    使用示例:
        from protocols.a2a import A2AServer

        base = A2AServer(name="secure_agent", description="安全Agent")

        @base.skill("search")
        def search(text: str) -> str:
            return f"搜索: {text}"

        server = SecureA2AServer(
            base_server=base,
            auth=AuthManager(),
            signer=MessageSigner(),
        )

        # 验证请求
        token = server.auth.generate_token("agent_1", scopes=["execute_skill"])
        signed = server.signer.sign_message({"skill": "search", "input": "test"})
        result = server.handle_secure_request(
            signed_msg=signed,
            token=token,
        )
    """

    def __init__(
        self,
        base_server: "A2AServer",
        auth: Optional[AuthManager] = None,
        signer: Optional[MessageSigner] = None,
    ):
        self.base_server = base_server
        self.auth = auth or AuthManager()
        self.signer = signer or MessageSigner()

        # 把 base_server 的 skill 装饰器绑定到自身,方便直接使用
        self.skill = base_server.skill
        self.register_skill = base_server.register_skill

    def handle_secure_request(
        self,
        signed_msg: SignedMessage,
        token: Optional[AuthToken] = None,
        required_scope: str = "execute_skill",
    ) -> Dict[str, Any]:
        """
        处理安全请求(认证 + 签名验证 + 执行)

        Args:
            signed_msg: 已签名消息
            token: 认证令牌
            required_scope: 需要的授权范围

        Returns:
            执行结果
        """
        # 1. 验证签名
        if not self.signer.verify_message(signed_msg):
            return {
                "success": False,
                "error": "签名验证失败",
                "error_code": "SIGNATURE_INVALID",
            }

        # 2. 验证 token
        if token is not None:
            verified = self.auth.verify_token(token.token, required_scope=required_scope)
            if verified is None:
                return {
                    "success": False,
                    "error": "认证失败或权限不足",
                    "error_code": "AUTH_FAILED",
                }

        # 3. 执行技能
        skill_name = signed_msg.payload.get("skill", "")
        input_data = signed_msg.payload.get("input", "")
        result = self.base_server.execute_skill(skill_name, input_data)

        return {
            "success": True,
            "authenticated_agent": token.agent_id if token else "anonymous",
            "verified_signature": True,
            "result": result,
        }

    def handle_multimodal_message(
        self,
        message: A2AMessage,
        token: Optional[AuthToken] = None,
    ) -> Dict[str, Any]:
        """
        处理多模态消息

        Args:
            message: 多模态消息
            token: 认证令牌

        Returns:
            处理结果
        """
        # 认证
        if token is not None:
            verified = self.auth.verify_token(token.token, required_scope="execute_skill")
            if verified is None:
                return {"success": False, "error": "认证失败"}

        # 提取技能名和输入
        skill_name = message.metadata.get("skill", "")
        if not skill_name:
            return {"success": False, "error": "缺少 skill 元数据"}

        # 提取文本内容作为输入
        text_parts = [p.content for p in message.parts if p.modality == ModalityType.TEXT]
        input_data = " ".join(text_parts) if text_parts else message.to_dict()

        # 携带多模态上下文
        context = {
            "multimodal_parts": [p.to_dict() for p in message.parts],
            "metadata": message.metadata,
        }

        result = self.base_server.execute_skill(skill_name, input_data)

        return {
            "success": True,
            "result": result,
            "multimodal_context": context,
        }


# ============================================================
#  6. 工厂函数
# ============================================================
def create_auth_manager(secret_key: str = "a2a-default-secret") -> AuthManager:
    """创建认证管理器"""
    return AuthManager(secret_key=secret_key)


def create_message_signer(secret_key: str = "a2a-default-signing-secret") -> MessageSigner:
    """创建消息签名器"""
    return MessageSigner(secret_key=secret_key)


def create_streaming_client(agent_url: str) -> StreamingA2AClient:
    """创建流式客户端"""
    return StreamingA2AClient(agent_url)