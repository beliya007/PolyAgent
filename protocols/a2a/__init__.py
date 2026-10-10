"""
A2A (Agent-to-Agent Protocol) 子模块

参考 polyagent 第十章 3 节设计,实现智能体之间的点对点协作协议。

A2A 协议设计理念:
- 智能体之间直接通信,避免中心化协调器的瓶颈
- 任务(Task)和工件(Artifact)是 A2A 的核心抽象
- 支持任务生命周期管理:创建->协商->执行->完成/失败
- 支持流式通信(SendMessageStream)

A2A 任务生命周期:
- 创建 (created): 任务创建
- 协商 (negotiating): 智能体协商中
- 接受 (accepted): 智能体接受任务
- 执行中 (working): 智能体执行中
- 完成 (completed): 任务完成
- 失败 (failed): 任务失败
- 取消 (canceled): 任务取消

主要组件:
- A2AServer: 智能体服务端,接收并处理其他智能体的请求
- A2AClient: 智能体客户端,向其他智能体发起请求
- A2ATool: 工具包装器,让 Agent 调用其他 A2A 服务
- enhanced: 增强模块(认证/签名/流式/多模态)
"""

from protocols.a2a.server import A2AServer, SkillDefinition, TaskStatus, Task
from protocols.a2a.client import A2AClient
from protocols.a2a.tool import A2ATool

# 增强模块(可选)
try:
    from protocols.a2a.enhanced import (
        AuthManager,
        AuthToken,
        AuthScope,
        MessageSigner,
        SignedMessage,
        A2AMessage,
        MultiModalPart,
        ModalityType,
        StreamChunk,
        StreamingA2AClient,
        SecureA2AServer,
        create_auth_manager,
        create_message_signer,
        create_streaming_client,
    )
    _ENHANCED_AVAILABLE = True
except ImportError:
    _ENHANCED_AVAILABLE = False

__all__ = [
    "A2AServer",
    "A2AClient",
    "A2ATool",
    "SkillDefinition",
    "TaskStatus",
    "Task",
    # 增强
    "AuthManager",
    "AuthToken",
    "AuthScope",
    "MessageSigner",
    "SignedMessage",
    "A2AMessage",
    "MultiModalPart",
    "ModalityType",
    "StreamChunk",
    "StreamingA2AClient",
    "SecureA2AServer",
    "create_auth_manager",
    "create_message_signer",
    "create_streaming_client",
]