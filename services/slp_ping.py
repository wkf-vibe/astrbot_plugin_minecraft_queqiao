"""Minecraft 服务端状态协议（Server List Ping）的最小实现（零第三方依赖）。

用于测量**直连服务器**的网络延迟（RTT, ms）——与玩家客户端加入服务器时
看到的 ping 同一测量口径，而不经过鹊桥中转：握手 + 状态请求的往返耗时。

```text
客户端                                 服务端 (TCP, 默认 25565)
  |-- Handshake(next state=1) + Status request -->|
  |<------------------- Status response (JSON) ---|
```

状态响应 JSON 同时带回版本名（含服务端品牌，如 ``forge 1.20.1`` /
``Paper 1.20.4``）、在线/最大人数与 MOTD，供监控侧缓存服务端类型
（TPS 指令 auto 选择）与在线人数。

仅使用标准库（asyncio/struct/zlib），不引入第三方 MC 库依赖。
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import struct
import time
import zlib
from dataclasses import dataclass, field

from ..core.constants import DEFAULT_SLP_PORT

# 握手协议版本：-1 表示"未知"，服务端按当前版本兼容处理（免维护版本表）
_PROTOCOL_VERSION = -1
# 握手 next state=1：进入 status 阶段
_NEXT_STATE_STATUS = 1
_STATUS_PACKET_ID = 0x00

# PROXY protocol v2（Paper `proxies.proxy-protocol: true` 时必须先发）：
# 12 字节签名 + version/command（0x21 = v2 + PROXY）
_PROXY_V2_SIGNATURE = b"\x0d\x0a\x0d\x0a\x00\x0d\x0a\x51\x55\x49\x54\x0a"
_PROXY_V2_CMD_PROXY = b"\x21"

# § 后跟一位格式/颜色代码（如 §a、§6、§r）。SLP players.sample 的名字在
# 部分服务端会被塞入这类彩色/广告文本，直连查询展示前剥离成可读纯文本。
_SECTION_RE = re.compile(r"§.")


@dataclass
class McPingResult:
    """一次 SLP ping 的结果（rtt 为往返耗时毫秒）。

    `player_names` 取自状态响应的 ``players.sample``（在线玩家名样本，
    已剥离 § 格式码）：可能不全或被服务端伪造/留空，与鹊桥 `get_status`
    的 sample 同源同语义。
    """

    rtt_ms: float
    version_name: str = ""
    online: int = 0
    max_players: int = 0
    description: str = ""
    player_names: list[str] = field(default_factory=list)


def _strip_format_codes(value: str) -> str:
    """去掉 Minecraft 文本格式码（§ 后跟一位代码）与首尾空白。"""
    if not value:
        return ""
    return _SECTION_RE.sub("", value).strip()


def _encode_varint(value: int) -> bytes:
    out = bytearray()
    value &= 0xFFFFFFFF
    while True:
        if value & ~0x7F == 0:
            out.append(value)
            return bytes(out)
        out.append((value & 0x7F) | 0x80)
        value >>= 7


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
    """解析 VarInt，返回 (值, 新偏移)。"""
    num = 0
    shift = 0
    for i in range(5):
        if offset + i >= len(data):
            raise ValueError("VarInt 数据不完整")
        byte = data[offset + i]
        num |= (byte & 0x7F) << shift
        shift += 7
        if byte & 0x80 == 0:
            return num, offset + i + 1
    raise ValueError("VarInt 过长")


async def _read_exact(
    reader: asyncio.StreamReader, size: int, timeout: float
) -> bytes:
    data = b""
    while len(data) < size:
        chunk = await asyncio.wait_for(reader.read(size - len(data)), timeout)
        if not chunk:
            raise ConnectionError("连接被服务端关闭")
        data += chunk
    return data


async def _read_varint(reader: asyncio.StreamReader, timeout: float) -> int:
    num = 0
    shift = 0
    for _ in range(5):
        byte = (await _read_exact(reader, 1, timeout))[0]
        num |= (byte & 0x7F) << shift
        shift += 7
        if byte & 0x80 == 0:
            return num
    raise ValueError("VarInt 过长")


async def _read_packet(reader: asyncio.StreamReader, timeout: float) -> bytes:
    """读取一个长度前缀的包；容忍服务端强制压缩（zlib 解压）。"""
    length = await _read_varint(reader, timeout)
    if length < 0:
        compressed = await _read_exact(reader, -length, timeout)
        try:
            return zlib.decompress(compressed)
        except zlib.error:
            raise ConnectionError("压缩包解压失败") from None
    return await _read_exact(reader, length, timeout)


def _component_text(node: object) -> str:
    """把 MOTD（str / {'text': ...} / 组件列表）抽取为纯文本。"""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return _component_text(node.get("text", ""))
    if isinstance(node, list):
        return "".join(_component_text(item) for item in node)
    return str(node)


async def mc_ping(
    host: str, port: int = 25565, timeout: float = 5.0
) -> McPingResult | None:
    """直连服务器执行一次状态查询，返回 RTT 与响应摘要；失败返回 None。

    失败原因（连接拒绝 / 超时 / 非 MC 端口 / 响应非法）需要展示给用户时，
    改用 ``mc_ping_ex`` 取回原因文本。
    """
    result, _ = await mc_ping_ex(host, port, timeout=timeout)
    return result


def _proxy_v2_header(
    sockname: tuple[str, int] | None, peername: tuple[str, int] | None
) -> bytes | None:
    """按 PROXY protocol v2 生成 IPv4/IPv6 的 PROXY 头（非 PROXY 命令）。

    Paper 开启 ``proxies.proxy-protocol`` 后，**任何**连到游戏端口的连接都要
    先收到 PROXY 头（FRP 隧道就是这么做的），否则服务端一直等头、不回状态
    响应——表现为「TCP 能连上但 SLP 永远超时」。
    """
    if not sockname or not peername:
        return None
    try:
        src_ip, src_port = sockname[0], int(sockname[1])
        dst_ip, dst_port = peername[0], int(peername[1])
        if ":" in src_ip:
            src = socket.inet_pton(socket.AF_INET6, src_ip)
            dst = socket.inet_pton(socket.AF_INET6, dst_ip)
            family_proto = b"\x21"  # AF_INET6 + STREAM
        else:
            src = socket.inet_aton(src_ip)
            dst = socket.inet_aton(dst_ip)
            family_proto = b"\x11"  # AF_INET + STREAM
        body = src + dst + struct.pack(">HH", src_port, dst_port)
        return (
            _PROXY_V2_SIGNATURE
            + _PROXY_V2_CMD_PROXY
            + family_proto
            + struct.pack(">H", len(body))
            + body
        )
    except (OSError, ValueError, IndexError):
        return None


async def _probe_once(
    host: str, port: int, timeout: float, use_proxy_protocol: bool
) -> tuple[McPingResult | None, str, bool]:
    """单次 SLP 探测，返回 ``(结果, 失败原因, TCP 是否连上)``。"""
    started = time.monotonic()
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout
        )
    except asyncio.TimeoutError:
        return None, f"连接 {host}:{port} 超时（地址/端口不可达或被防火墙丢弃）", False
    except ConnectionRefusedError:
        return None, f"连接 {host}:{port} 被拒绝（端口未监听）", False
    except OSError as exc:
        return None, f"连接 {host}:{port} 失败：{exc}", False

    try:
        if use_proxy_protocol:
            header = _proxy_v2_header(
                writer.get_extra_info("sockname"), writer.get_extra_info("peername")
            )
            if header is None:
                return None, f"{host}:{port} 无法生成 PROXY v2 头", True
            writer.write(header)
        host_bytes = host.encode()
        handshake = (
            b"\x00"
            + _encode_varint(_PROTOCOL_VERSION)
            + _encode_varint(len(host_bytes))
            + host_bytes
            + struct.pack(">H", port)
            + bytes([_NEXT_STATE_STATUS])
        )
        writer.write(_encode_varint(len(handshake)) + handshake)
        writer.write(b"\x01\x00")  # 状态请求包（len=1, id=0）
        await asyncio.wait_for(writer.drain(), timeout)

        data = await asyncio.wait_for(_read_packet(reader, timeout), timeout)
    except asyncio.TimeoutError:
        return (
            None,
            f"{host}:{port} 端口可连接但未返回状态响应（超时）",
            True,
        )
    except (OSError, ConnectionError, ValueError) as exc:
        return None, f"{host}:{port} 状态响应异常：{exc}", True
    finally:
        writer.close()
        try:
            await asyncio.wait_for(writer.wait_closed(), 1.0)
        except (OSError, asyncio.TimeoutError):
            pass

    rtt_ms = round((time.monotonic() - started) * 1000, 1)
    try:
        _, offset = _decode_varint(data, 0)
        json_len, offset = _decode_varint(data, offset)
        raw = data[offset : offset + json_len]
        payload = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return None, f"{host}:{port} 响应不是 Minecraft 状态协议", True
    # 严格校验：必须是合法 MC 状态响应（含 version / players 等结构）。
    # 避免把 HTTP 等非 MC 服务（如鹊桥 Web 端口）的响应误判为成功——
    # 那会让延迟显示一个并非游戏端口的假低值
    if not isinstance(payload, dict):
        return None, f"{host}:{port} 响应不是 Minecraft 状态协议", True
    if not isinstance(payload.get("version"), dict) and not isinstance(
        payload.get("players"), dict
    ):
        return None, f"{host}:{port} 响应不是 Minecraft 状态协议", True

    version = payload.get("version") or {}
    players = payload.get("players") or {}
    # players.sample：在线玩家名样本（可能不全/被伪造/留空）；name 可能是
    # 文本组件（str/dict/list），与 MOTD 同源处理
    player_names: list[str] = []
    sample_raw = players.get("sample")
    if isinstance(sample_raw, list):
        for item in sample_raw:
            if not isinstance(item, dict):
                continue
            name = _strip_format_codes(_component_text(item.get("name")))
            if name:
                player_names.append(name)
    return (
        McPingResult(
            rtt_ms=rtt_ms,
            version_name=str(version.get("name") or ""),
            online=int(players.get("online") or 0),
            max_players=int(players.get("max") or 0),
            description=_component_text(payload.get("description", "")),
            player_names=player_names,
        ),
        "",
        True,
    )


async def mc_ping_ex(
    host: str, port: int = 25565, timeout: float = 5.0
) -> tuple[McPingResult | None, str]:
    """SLP 探测并返回 ``(结果, 失败原因)``；成功时失败原因为空串。

    先按普通 SLP 握手探测；若「TCP 连上了但服务端不回状态响应」，再带一个
    PROXY protocol v2 头重试一次——服务端开了 ``proxy-protocol``（FRP 隧道
    直连的常见形态）时，必须先收到 PROXY 头才会处理 SLP。两种都不通才判定
    失败，并把最后一次的具体原因交给调用方展示。
    """
    if not host:
        return None, "未配置探测地址"
    result, reason, connected = await _probe_once(host, port, timeout, False)
    if result is not None:
        return result, ""
    # 明确不可达（拒绝/连接超时/响应非法）时，PROXY 头也救不回来
    if not connected or "未返回状态响应" not in reason:
        return None, reason

    retried, retry_reason, _ = await _probe_once(host, port, timeout, True)
    if retried is not None:
        return retried, ""
    return None, f"{retry_reason}（已尝试 PROXY v2 兼容头）"


def brand_from_slp(version_name: str, description: str = "") -> str | None:
    """从状态响应的版本名 / MOTD 中识别服务端品牌，供 TPS 指令 auto 选择。

    返回小写品牌关键字（forge / fabric / quilt / paper / spigot / bukkit /
    purpur / pufferfish / leaves / sponge / folia）；识别不到返回 None。
    """
    text = f"{version_name} {description}".lower()
    for brand in (
        "pufferfish", "purpur", "paper", "spigot", "bukkit", "folia",
        "sponge", "leaves", "fabric", "quilt", "forge",
    ):
        if brand in text:
            return brand
    return None


def parse_host_port(
    text: str, default_port: int = DEFAULT_SLP_PORT
) -> tuple[str, int] | None:
    """把用户填写的「host[:port]」解析为 (host, port)；空值或非法返回 None。

    容忍复制粘贴带上的 scheme 与路径（``http://mc.example.com:25565/``），
    IPv6 支持 ``[::1]:25565`` 与裸 ``::1`` 两种写法：裸 IPv6 的冒号无法与
    端口分隔符区分，故只在整个字符串恰好一个冒号时才把后段当端口。
    """
    value = (text or "").strip()
    if not value:
        return None
    if "//" in value:
        value = value.split("//", 1)[1]
    value = value.split("/", 1)[0].strip()
    if not value:
        return None
    if value.startswith("["):
        host, _, rest = value[1:].partition("]")
        if not host:
            return None
        port_text = rest[1:] if rest.startswith(":") else ""
        return host, int(port_text) if port_text.isdigit() else default_port
    if value.count(":") == 1:
        host, _, port_text = value.partition(":")
        if not host:
            return None
        return host, int(port_text) if port_text.isdigit() else default_port
    return value, default_port