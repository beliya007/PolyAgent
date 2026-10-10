"""
A2A 和 ANP 模块测试脚本

测试 A2A Server/Client/Tool 和 ANP Discovery/Router/Tool 功能。

运行方式：
    cd agent
    python -m protocols.test_all_protocols
"""

import sys


def test_a2a_server():
    """测试 A2A Server"""
    print("\n" + "=" * 60)
    print("测试 1: A2A Server (技能注册与执行)")
    print("=" * 60)

    from protocols.a2a import A2AServer

    server = A2AServer(
        name="calculator_agent",
        description="数学计算智能体",
        capabilities={"math": ["add", "multiply", "factorial"]}
    )

    @server.skill("add")
    def add_numbers(text: str) -> str:
        """加法计算"""
        import re
        # 简单解析 "add 5 3" 格式
        parts = re.findall(r'-?\d+\.?\d*', text)
        if len(parts) >= 2:
            result = float(parts[0]) + float(parts[1])
            return f"{parts[0]} + {parts[1]} = {result}"
        return "请提供两个数字，例如: add 5 3"

    @server.skill("multiply")
    def multiply_numbers(text: str) -> str:
        """乘法计算"""
        import re
        parts = re.findall(r'-?\d+\.?\d*', text)
        if len(parts) >= 2:
            result = float(parts[0]) * float(parts[1])
            return f"{parts[0]} × {parts[1]} = {result}"
        return "请提供两个数字"

    @server.skill("info")
    def get_info(text: str) -> str:
        """获取智能体信息"""
        return f"我是 {server.name},支持技能: {list(server._skills.keys())}"

    # 列出技能
    print("\n服务器注册的技能:")
    for skill in server.list_skills():
        print(f"  - {skill['name']}: {skill['description']}")

    # 获取智能体卡片
    print("\n智能体卡片:")
    card = server.get_agent_card()
    print(f"  名称: {card['name']}")
    print(f"  描述: {card['description']}")
    print(f"  能力: {card['capabilities']}")

    # 同步执行技能
    print("\n执行技能:")
    result = server.execute_skill("add", "add 10 5")
    print(f"  add: {result.get('result')}")

    result = server.execute_skill("multiply", "multiply 6 7")
    print(f"  multiply: {result.get('result')}")

    result = server.execute_skill("info", "info")
    print(f"  info: {result.get('result')}")

    return server


def test_a2a_client_local():
    """测试 A2A Client (本地内存模式)"""
    print("\n" + "=" * 60)
    print("测试 2: A2A Client (本地内存模式)")
    print("=" * 60)

    from protocols.a2a import A2AServer, A2AClient

    # 创建服务端
    server = A2AServer(name="local_server", description="本地服务")

    @server.skill("greet")
    def greet(text: str) -> str:
        return f"你好,{text}!"

    @server.skill("echo")
    def echo(text: str) -> str:
        return f"Echo: {text}"

    # 创建客户端
    client = A2AClient("http://localhost:5000", use_http=False)

    print("\n客户端信息:")
    print(f"  URL: {client.agent_url}")

    # 获取技能列表
    skills = client.list_skills()
    print(f"\n发现技能 ({len(skills)} 个):")
    for s in skills:
        print(f"  - {s['name']}: {s['description']}")

    # 注意：use_http=False 时是模拟模式
    result = client.execute_skill("greet", "World")
    print(f"\n执行结果: {result}")


def test_a2a_tool():
    """测试 A2ATool"""
    print("\n" + "=" * 60)
    print("测试 3: A2ATool (工具包装器)")
    print("=" * 60)

    from protocols.a2a import A2AServer, A2ATool

    # 创建服务端
    server = A2AServer(name="weather_agent", description="天气查询智能体")

    @server.skill("query_weather")
    def query_weather(text: str) -> str:
        """查询天气"""
        return f"查询地点: {text}\n天气: 晴\n温度: 25°C\n湿度: 60%"

    @server.skill("forecast")
    def forecast(text: str) -> str:
        """天气预报"""
        return f"{text} 未来7天天气: 晴转多云"

    # 从 server 创建 A2ATool
    tool = A2ATool.from_server(server, name="weather")
    print(f"\nA2ATool 创建: {tool.name}")

    # 列出展开的技能
    expanded = tool.get_expanded_skills()
    print(f"\n展开技能 ({len(expanded)} 个):")
    for skill in expanded:
        print(f"  - {skill.prefixed_name}: {skill.description}")

    # 调用展开的技能
    print("\n调用展开的技能:")
    result = tool.call_expanded_skill("weather_query_weather", "上海")
    print(f"  weather_query_weather: {result}")

    result = tool.call_expanded_skill("weather_forecast", "北京")
    print(f"  weather_forecast: {result}")


def test_anp_discovery():
    """测试 ANPDiscovery"""
    print("\n" + "=" * 60)
    print("测试 4: ANPDiscovery (服务注册与发现)")
    print("=" * 60)

    from protocols.anp import ANPDiscovery, register_service

    discovery = ANPDiscovery()

    # 注册多个服务
    print("\n注册服务:")
    register_service(
        discovery=discovery,
        service_id="nlp_agent_1",
        service_name="NLP处理专家A",
        service_type="nlp",
        capabilities=["text_analysis", "sentiment_analysis"],
        endpoint="http://localhost:8001",
        metadata={"load": 0.3, "price": 0.01}
    )

    register_service(
        discovery=discovery,
        service_id="nlp_agent_2",
        service_name="NLP处理专家B",
        service_type="nlp",
        capabilities=["text_analysis", "translation"],
        endpoint="http://localhost:8002",
        metadata={"load": 0.7, "price": 0.005}
    )

    register_service(
        discovery=discovery,
        service_id="vision_agent_1",
        service_name="视觉识别专家",
        service_type="vision",
        capabilities=["object_detection", "face_recognition"],
        endpoint="http://localhost:8003",
        metadata={"load": 0.2, "price": 0.02}
    )

    # 发现服务
    print("\n按类型发现 'nlp' 服务:")
    nlp_services = discovery.discover_by_type("nlp")
    for s in nlp_services:
        print(f"  - {s.service_name} (负载: {s.load:.1%})")

    print("\n按能力发现 'text_analysis' 服务:")
    text_services = discovery.discover_by_capability("text_analysis")
    for s in text_services:
        print(f"  - {s.service_name} ({s.service_type})")

    # 统计信息
    print("\n统计信息:")
    print(json.dumps(discovery.stats(), ensure_ascii=False, indent=2))


def test_anp_router():
    """测试 ANPRouter"""
    print("\n" + "=" * 60)
    print("测试 5: ANPRouter (智能路由)")
    print("=" * 60)

    from protocols.anp import ANPDiscovery, ANPRouter, register_service

    discovery = ANPDiscovery()

    # 注册多个相同能力的服务（不同负载/价格）
    register_service(
        discovery=discovery,
        service_id="agent_1",
        service_name="Agent 1",
        service_type="compute",
        capabilities=["calc"],
        endpoint="http://localhost:8001",
        metadata={"load": 0.8, "price": 0.01}
    )

    register_service(
        discovery=discovery,
        service_id="agent_2",
        service_name="Agent 2",
        service_type="compute",
        capabilities=["calc"],
        endpoint="http://localhost:8002",
        metadata={"load": 0.2, "price": 0.02}
    )

    register_service(
        discovery=discovery,
        service_id="agent_3",
        service_name="Agent 3",
        service_type="compute",
        capabilities=["calc"],
        endpoint="http://localhost:8003",
        metadata={"load": 0.5, "price": 0.005}
    )

    router = ANPRouter(discovery)

    # 测试不同路由策略
    strategies = ["least_loaded", "lowest_price", "round_robin", "random"]

    for strategy in strategies:
        agent = router.route_by_capability("calc", strategy=strategy)
        print(f"  策略 '{strategy}': {agent.service_name} (负载: {agent.load:.1%})")


def test_anp_tool():
    """测试 ANPTool"""
    print("\n" + "=" * 60)
    print("测试 6: ANPTool (工具包装器)")
    print("=" * 60)

    from protocols.anp.tool import ANPTool

    tool = ANPTool(name="network")
    print(f"\nANPTool 创建: {tool.name}")

    # 注册服务（通过 execute 接口）
    print("\n注册服务 (通过 execute):")
    result = tool.execute(
        action="register_service",
        service_id="my_agent_1",
        service_name="我的智能体",
        service_type="math",
        capabilities=["calc", "analyze"],
        endpoint="http://localhost:9001",
        metadata={"load": 0.3}
    )
    print(f"  {result}")

    # 列出所有服务
    print("\n列出所有服务:")
    print(tool.execute(action="list_services"))

    # 发现服务
    print("\n发现 'math' 类型服务:")
    print(tool.execute(action="discover_services", service_type="math"))

    # 智能路由
    print("\n智能路由 (按能力 calc):")
    print(tool.execute(action="route_task", capability="calc", strategy="least_loaded"))

    # 统计信息
    print("\n统计信息:")
    print(tool.execute(action="stats"))

    # 帮助信息
    print("\n帮助信息:")
    print(tool.execute()[:200] + "...")


def test_a2a_enhanced_auth():
    """测试 A2A 增强:认证管理器"""
    print("\n" + "=" * 60)
    print("测试 7: A2A 增强 - AuthManager (认证)")
    print("=" * 60)

    from protocols.a2a import AuthManager, AuthScope

    auth = AuthManager(secret_key="test-secret")

    # 生成 token
    token = auth.generate_token(
        agent_id="test_agent",
        scopes=[AuthScope.EXECUTE_SKILL.value, AuthScope.LIST_SKILLS.value],
    )
    print(f"\n生成 token:")
    print(f"  agent_id: {token.agent_id}")
    print(f"  scopes: {token.scopes}")
    print(f"  有效期: {token.expires_at - token.issued_at:.0f} 秒")

    # 验证 token
    verified = auth.verify_token(token.token, required_scope="execute_skill")
    print(f"\n验证 token (需要 execute_skill): {'通过' if verified else '失败'}")

    # 测试无效 scope
    verified = auth.verify_token(token.token, required_scope="admin")
    print(f"验证 token (需要 admin): {'通过' if verified else '拒绝'}")

    # 撤销
    auth.revoke_token(token.token)
    verified = auth.verify_token(token.token)
    print(f"撤销后验证: {'通过' if verified else '失败'}")


def test_a2a_enhanced_signer():
    """测试 A2A 增强:消息签名"""
    print("\n" + "=" * 60)
    print("测试 8: A2A 增强 - MessageSigner (签名)")
    print("=" * 60)

    from protocols.a2a import MessageSigner

    signer = MessageSigner(secret_key="test-signing-secret")

    payload = {"skill": "add", "input": "1 2"}
    signed = signer.sign_message(payload)

    print(f"\n签名消息:")
    print(f"  signature (前 16 字符): {signed.signature[:16]}...")
    print(f"  nonce: {signed.nonce}")

    # 验证签名
    is_valid = signer.verify_message(signed)
    print(f"\n签名验证: {'通过' if is_valid else '失败'}")

    # 篡改 payload
    tampered = signed
    tampered.payload["input"] = "恶意输入"
    is_valid = signer.verify_message(tampered)
    print(f"篡改后验证: {'通过' if is_valid else '失败(防篡改)'}")


def test_a2a_enhanced_multimodal():
    """测试 A2A 增强:多模态消息"""
    print("\n" + "=" * 60)
    print("测试 9: A2A 增强 - 多模态消息")
    print("=" * 60)

    from protocols.a2a import A2AMessage, ModalityType

    msg = A2AMessage(role="user")
    msg.add_text("请分析这张图片")
    msg.add_image(
        data="data:image/png;base64,iVBORw0KG...",
        mime_type="image/png",
        metadata={"width": 800, "height": 600},
    )
    msg.add_json({"case_id": "CASE-001", "priority": "high"})

    print(f"\n多模态消息:")
    print(f"  角色: {msg.role}")
    print(f"  片段数: {len(msg.parts)}")
    for i, part in enumerate(msg.parts, 1):
        print(f"  [{i}] {part.modality.value}: {str(part.content)[:50]}")


def test_a2a_secure_server():
    """测试 A2A 增强:SecureA2AServer"""
    print("\n" + "=" * 60)
    print("测试 10: A2A 增强 - SecureA2AServer (安全请求)")
    print("=" * 60)

    from protocols.a2a import A2AServer, SecureA2AServer, AuthManager, MessageSigner

    # 基础 server
    base = A2AServer(name="secure_test_agent", description="安全测试Agent")

    @base.skill("add")
    def add(text: str) -> str:
        import re
        parts = re.findall(r'\d+', text)
        if len(parts) >= 2:
            return str(int(parts[0]) + int(parts[1]))
        return "无效输入"

    # 增强为 secure
    secure = SecureA2AServer(
        base_server=base,
        auth=AuthManager(secret_key="secure-test"),
        signer=MessageSigner(secret_key="signing-test"),
    )

    # 生成 token + 签名
    token = secure.auth.generate_token(
        "client_agent",
        scopes=["execute_skill"],
    )
    signed = secure.signer.sign_message({"skill": "add", "input": "5 7"})

    # 安全请求
    result = secure.handle_secure_request(signed_msg=signed, token=token)
    print(f"\n安全请求结果:")
    print(f"  success: {result.get('success')}")
    print(f"  authenticated_agent: {result.get('authenticated_agent')}")
    print(f"  verified_signature: {result.get('verified_signature')}")
    if result.get('success'):
        print(f"  task_result: {result['result']['result']}")


def test_anp_identity():
    """测试 ANP 增强:DID 身份"""
    print("\n" + "=" * 60)
    print("测试 11: ANP 增强 - ANPIdentity (DID)")
    print("=" * 60)

    from protocols.anp import ANPIdentity

    identity = ANPIdentity(
        agent_id="analysis_agent_01",
        service_type="analysis",
        service_name="通用分析 Agent",
    )

    print(f"\n智能体身份:")
    print(f"  agent_id: {identity.agent_id}")
    print(f"  service_type: {identity.service_type}")
    print(f"  DID: {identity.did}")
    print(f"  public_key (前 16 字符): {identity.public_key[:16]}...")

    # 签名/验证
    data = "important configuration data"
    signature = identity.sign(data)
    is_valid = identity.verify(data, signature)
    print(f"\n签名: {signature[:16]}...")
    print(f"验证: {'通过' if is_valid else '失败'}")


def test_anp_secure_bus():
    """测试 ANP 增强:安全消息总线"""
    print("\n" + "=" * 60)
    print("测试 12: ANP 增强 - SecureMessageBus")
    print("=" * 60)

    from protocols.anp import ANPIdentity, SecureMessageBus

    bus = SecureMessageBus(secret_key="bus-secret")

    # 注册身份
    id_1 = ANPIdentity("agent_a", "analysis")
    id_2 = ANPIdentity("agent_b", "summary")
    bus.register_identity(id_1)
    bus.register_identity(id_2)

    # 发送
    msg_id = bus.send(
        from_agent="agent_a",
        to_agent="agent_b",
        payload={"action": "summarize", "task_id": "TASK-001"},
    )
    print(f"\n消息已发送: {msg_id}")

    # 接收
    msg = bus.receive("agent_b")
    if msg:
        print(f"\nagent_b 收到消息:")
        print(f"  from: {msg.from_agent}")
        print(f"  payload: {msg.payload}")
        print(f"  签名验证: 通过")

    print(f"\n总线统计: {bus.stats()}")


def test_anp_health_monitor():
    """测试 ANP 增强:健康监控"""
    print("\n" + "=" * 60)
    print("测试 13: ANP 增强 - HealthMonitor")
    print("=" * 60)

    from protocols.anp import ANPDiscovery, HealthMonitor, register_service

    discovery = ANPDiscovery()
    register_service(
        discovery=discovery,
        service_id="healthy_agent",
        service_name="健康Agent",
        service_type="compute",
        capabilities=["calc"],
        endpoint="http://localhost:9001",
        metadata={"load": 0.3},
    )

    monitor = HealthMonitor(discovery)
    # 注册检查函数(返回 True = 健康)
    monitor.add_check("healthy_agent", lambda: True)

    result = monitor.check_agent("healthy_agent")
    print(f"\n健康检查结果:")
    print(f"  agent_id: {result.agent_id}")
    print(f"  healthy: {result.healthy}")
    print(f"  latency_ms: {result.latency_ms:.2f}")


def test_anp_mesh():
    """测试 ANP 增强:服务网格"""
    print("\n" + "=" * 60)
    print("测试 14: ANP 增强 - ANPMesh (服务网格)")
    print("=" * 60)

    from protocols.anp import ANPIdentity, ANPMesh

    mesh = ANPMesh()

    # 添加节点
    id_a = ANPIdentity("node_a", "analysis")
    id_b = ANPIdentity("node_b", "summary")
    id_c = ANPIdentity("node_c", "router")
    mesh.add_agent(id_a)
    mesh.add_agent(id_b)
    mesh.add_agent(id_c)

    # 建立连接(三角网络)
    mesh.connect("node_a", "node_b")
    mesh.connect("node_b", "node_c")
    mesh.connect("node_a", "node_c")

    # 路径查找
    path = mesh.find_path("node_a", "node_c")
    print(f"\nnode_a -> node_c 路径: {' -> '.join(path) if path else '不可达'}")

    # 路由消息
    print(f"\n路由消息:")
    mesh.route_message(
        source="node_a",
        target="node_c",
        capability="data_analysis",
        payload={"data": "sample"},
    )

    # 拓扑
    topo = mesh.topology()
    print(f"\n网络拓扑:")
    print(f"  节点数: {len(topo['nodes'])}")
    print(f"  边数: {len(topo['edges'])}")


def main():
    print("=" * 60)
    print("A2A 和 ANP 协议模块测试")
    print("=" * 60)

    # 测试 A2A 基础
    test_a2a_server()
    test_a2a_client_local()
    test_a2a_tool()

    # 测试 ANP 基础
    test_anp_discovery()
    test_anp_router()
    test_anp_tool()

    # 测试 A2A 增强
    test_a2a_enhanced_auth()
    test_a2a_enhanced_signer()
    test_a2a_enhanced_multimodal()
    test_a2a_secure_server()

    # 测试 ANP 增强
    test_anp_identity()
    test_anp_secure_bus()
    test_anp_health_monitor()
    test_anp_mesh()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    import json
    main()