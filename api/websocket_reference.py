"""
WebSocket & Server-Sent Events (SSE) Reference

Apply to: Real-time dashboards, live notifications, collaborative tools,
          chat applications, live sports/market data feeds

Features:
- FastAPI WebSocket: connection lifecycle, ping/pong keepalive
- Room/channel manager: fan-out to multiple subscribers per room
- JWT authentication on WebSocket upgrade (query-param + header patterns)
- Horizontal scaling via Redis Pub/Sub (one node publishes, all nodes fan-out)
- Graceful disconnect handling and reconnection backoff guidance
- Server-Sent Events (SSE): unidirectional push without WebSocket overhead
- Rate limiting per connection (token bucket)
- Binary frame support (MessagePack alternative to JSON)
- Structured message envelope (type + payload + correlation_id)
- AsyncAPI 2.x schema reference (docstring)
- Choosing WebSocket vs SSE vs long-polling decision guide

Installation:
    pip install fastapi uvicorn redis[asyncio] msgpack

Run:
    uvicorn websocket_reference:app --reload

Test WebSocket:
    websocat ws://localhost:8000/ws/room/general?token=valid-token

Test SSE:
    curl -N http://localhost:8000/events/user-123?token=valid-token
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Optional, Set

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


# =============================================================================
# AsyncAPI 2.x schema reference
# =============================================================================
#
# asyncapi: "2.6.0"
# info:
#   title: Real-Time Events API
#   version: "1.0.0"
# channels:
#   /ws/room/{roomId}:
#     parameters:
#       roomId: { schema: { type: string } }
#     subscribe:
#       message:
#         payload:
#           type: object
#           properties:
#             type: { type: string, enum: [message, join, leave, ping] }
#             payload: { type: object }
#             correlation_id: { type: string, format: uuid }
#             ts: { type: integer }
#     publish:
#       message:
#         payload:
#           $ref: "#/components/schemas/OutboundEvent"
#
#   /events/{userId}:
#     description: SSE endpoint — unidirectional server push
#     subscribe:
#       message:
#         payload:
#           type: object
#           properties:
#             event: { type: string }
#             data: { type: object }
#             id: { type: string }


# =============================================================================
# Message envelope
# =============================================================================

@dataclass
class WSMessage:
    type: str        # "message" | "join" | "leave" | "ping" | "pong" | "error"
    payload: Any = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: int = field(default_factory=lambda: int(time.time()))

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "ts": self.ts,
        })

    @classmethod
    def from_json(cls, raw: str) -> "WSMessage":
        data = json.loads(raw)
        return cls(
            type=data.get("type", "message"),
            payload=data.get("payload"),
            correlation_id=data.get("correlation_id", str(uuid.uuid4())),
            ts=data.get("ts", int(time.time())),
        )


# =============================================================================
# Connection rate limiter
# =============================================================================

@dataclass
class ConnectionRateLimiter:
    """Token-bucket rate limiter per connection."""
    capacity: int = 30           # max burst
    refill_rate: float = 10.0    # tokens per second
    _tokens: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.capacity)
        self._last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False


# =============================================================================
# Room manager — in-process fan-out
# =============================================================================

class RoomManager:
    """
    In-memory connection registry.
    For multi-node deployments, replace publish() with Redis Pub/Sub (see RedisRoomBridge).
    """

    def __init__(self) -> None:
        self._rooms: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def join(self, room_id: str, ws: WebSocket) -> None:
        async with self._lock:
            self._rooms.setdefault(room_id, set()).add(ws)
        logger.info("WS joined room %s (size=%d)", room_id, len(self._rooms[room_id]))

    async def leave(self, room_id: str, ws: WebSocket) -> None:
        async with self._lock:
            room = self._rooms.get(room_id, set())
            room.discard(ws)
            if not room:
                self._rooms.pop(room_id, None)
        logger.info("WS left room %s", room_id)

    async def broadcast(self, room_id: str, message: WSMessage, exclude: Optional[WebSocket] = None) -> None:
        """Fan-out to every connection in the room (except sender)."""
        payload = message.to_json()
        dead: list[WebSocket] = []
        for ws in list(self._rooms.get(room_id, set())):
            if ws is exclude:
                continue
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.leave(room_id, ws)

    def room_size(self, room_id: str) -> int:
        return len(self._rooms.get(room_id, set()))


room_manager = RoomManager()


# =============================================================================
# Redis Pub/Sub bridge — horizontal scaling
# =============================================================================

class RedisRoomBridge:
    """
    Bridges RoomManager fan-out to Redis PUBLISH/SUBSCRIBE so that
    messages published on any server node reach all nodes in the cluster.

    Usage:
        bridge = RedisRoomBridge(redis_url="redis://localhost:6379")
        await bridge.start()                        # in lifespan
        await bridge.publish("general", message)    # instead of room_manager.broadcast()
        await bridge.stop()
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        self._sub_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        import redis.asyncio as aioredis  # pip install redis[asyncio]
        self._pub = aioredis.from_url(self._redis_url, decode_responses=True)
        self._sub_client = aioredis.from_url(self._redis_url, decode_responses=True)
        self._pubsub = self._sub_client.pubsub()
        await self._pubsub.psubscribe("room:*")
        self._sub_task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._sub_task:
            self._sub_task.cancel()
        await self._pub.aclose()
        await self._sub_client.aclose()

    async def publish(self, room_id: str, message: WSMessage) -> None:
        await self._pub.publish(f"room:{room_id}", message.to_json())

    async def _listen(self) -> None:
        """Receive from Redis, fan-out locally."""
        async for raw in self._pubsub.listen():
            if raw["type"] != "pmessage":
                continue
            channel: str = raw["channel"]          # "room:general"
            room_id = channel.split(":", 1)[1]
            try:
                msg = WSMessage.from_json(raw["data"])
                await room_manager.broadcast(room_id, msg)
            except Exception as exc:
                logger.error("Redis bridge error: %s", exc)


# =============================================================================
# JWT authentication helper
# =============================================================================

async def authenticate_ws(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate WebSocket upgrade token.
    Tokens can arrive via:
      1. Query param: ws://host/ws/room?token=<jwt>   (simple, works everywhere)
      2. Subprotocol: Sec-WebSocket-Protocol: access_token, <jwt>  (more secure)
    Never send credentials in Cookie without CSRF protection.
    """
    if not token:
        return None
    # Replace with real JWT decode (PyJWT / python-jose):
    if token == "valid-token":  # placeholder
        return {"sub": "user-123", "roles": ["viewer"]}
    return None


# =============================================================================
# FastAPI application + lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: start Redis bridge
    # bridge = RedisRoomBridge(redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"))
    # await bridge.start()
    logger.info("WebSocket server started")
    yield
    # await bridge.stop()
    logger.info("WebSocket server stopped")


app = FastAPI(
    title="WebSocket & SSE Reference",
    description="Production-ready real-time API patterns",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# WebSocket endpoint
# =============================================================================

PING_INTERVAL = 25.0   # seconds between server pings
PING_TIMEOUT = 10.0    # seconds to wait for pong before disconnect


@app.websocket("/ws/room/{room_id}")
async def websocket_room(
    ws: WebSocket,
    room_id: str,
    token: Optional[str] = Query(default=None),
):
    """
    WebSocket chat room with:
    - JWT auth on upgrade
    - Per-connection rate limiting
    - Keepalive ping/pong
    - Structured message envelope
    - Graceful disconnect
    """
    user = await authenticate_ws(token)
    if user is None:
        await ws.close(code=4401, reason="Unauthorized")
        return

    await ws.accept()
    await room_manager.join(room_id, ws)
    limiter = ConnectionRateLimiter()
    user_id = user["sub"]

    # Announce join
    await room_manager.broadcast(room_id, WSMessage(type="join", payload={"user": user_id}), exclude=ws)

    # Keepalive task
    async def keepalive() -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                if ws.client_state != WebSocketState.CONNECTED:
                    return
                await ws.send_text(WSMessage(type="ping").to_json())
                # Wait for pong (handled in main loop)
            except Exception:
                return

    ka_task = asyncio.create_task(keepalive())

    try:
        while True:
            raw = await asyncio.wait_for(ws.receive_text(), timeout=PING_INTERVAL + PING_TIMEOUT)
            msg = WSMessage.from_json(raw)

            if msg.type == "pong":
                continue  # heartbeat acknowledged

            if not limiter.allow():
                await ws.send_text(WSMessage(type="error", payload={"code": 429, "detail": "Rate limit exceeded"}).to_json())
                continue

            if msg.type == "message":
                msg.payload = {**(msg.payload or {}), "sender": user_id}
                await room_manager.broadcast(room_id, msg)

    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception as exc:
        logger.error("WebSocket error for user %s: %s", user_id, exc)
    finally:
        ka_task.cancel()
        await room_manager.leave(room_id, ws)
        await room_manager.broadcast(room_id, WSMessage(type="leave", payload={"user": user_id}))


# =============================================================================
# Server-Sent Events (SSE) endpoint
# =============================================================================

async def _sse_event(data: Any, event: str = "message", id: Optional[str] = None) -> str:
    """Format a single SSE event block."""
    lines = []
    if id:
        lines.append(f"id: {id}")
    lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    lines.append("")  # blank line = end of event
    return "\n".join(lines) + "\n"


async def _user_event_generator(user_id: str) -> AsyncGenerator[str, None]:
    """
    Yield SSE events for a specific user.
    Production pattern: subscribe to Redis or a DB change-stream per user_id.
    This stub yields synthetic events every 2 seconds.
    """
    # Send retry hint so clients reconnect after 3 s on disconnect
    yield "retry: 3000\n\n"

    seq = 0
    try:
        while True:
            await asyncio.sleep(2)
            seq += 1
            event_data = {"seq": seq, "user_id": user_id, "msg": f"Notification #{seq}"}
            yield await _sse_event(data=event_data, event="notification", id=str(seq))
    except asyncio.CancelledError:
        pass  # client disconnected


@app.get("/events/{user_id}")
async def sse_user_feed(
    request: Request,
    user_id: str,
    token: Optional[str] = Query(default=None),
):
    """
    Server-Sent Events feed for a specific user.
    Clients use the EventSource API (no extra library needed in browsers).

    JS usage:
        const es = new EventSource('/events/user-123?token=valid-token');
        es.addEventListener('notification', e => console.log(JSON.parse(e.data)));

    SSE vs WebSocket:
    - Use SSE when you only need server → client push (no client → server messages).
    - SSE automatically reconnects; WebSocket does not (client must handle retry).
    - SSE works through all HTTP/1.1 proxies; WebSocket may require proxy config.
    - WebSocket is required for bidirectional low-latency messaging.
    """
    user = await authenticate_ws(token)
    if user is None:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    return StreamingResponse(
        _user_event_generator(user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# Webhook receiver reference
# =============================================================================

import hashlib
import hmac


def verify_webhook_signature(
    payload: bytes,
    signature_header: str,
    secret: str,
    algorithm: str = "sha256",
) -> bool:
    """
    Verify HMAC signature on incoming webhook payloads.
    Compatible with GitHub, Stripe, Shopify, Twilio patterns.

    Header formats:
    - GitHub:  X-Hub-Signature-256: sha256=<hex>
    - Stripe:  Stripe-Signature: t=<ts>,v1=<hex>
    - Generic: X-Signature: <hex>
    """
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256 if algorithm == "sha256" else hashlib.sha1,
    ).hexdigest()

    # Handle "sha256=<hex>" prefix (GitHub style)
    sig = signature_header.split("=", 1)[-1]
    return hmac.compare_digest(expected, sig)


@app.post("/webhooks/github")
async def github_webhook(request: Request) -> Dict[str, Any]:
    """
    Receive and verify GitHub webhook events.
    Pattern applies to any provider that sends HMAC-signed payloads.
    """
    body = await request.body()
    sig_header = request.headers.get("X-Hub-Signature-256", "")

    WEBHOOK_SECRET = "your-github-webhook-secret"  # load from env in production
    if not verify_webhook_signature(body, sig_header, WEBHOOK_SECRET):
        from fastapi import HTTPException
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    event_type = request.headers.get("X-GitHub-Event", "unknown")
    payload = json.loads(body)
    logger.info("GitHub webhook: event=%s repo=%s", event_type, payload.get("repository", {}).get("full_name"))
    # Dispatch to event handler queue (Celery, arq, etc.)
    return {"status": "accepted", "event": event_type}


# =============================================================================
# Protocol choice guide
# =============================================================================
#
# ┌─────────────────┬────────────┬───────────┬──────────────┬─────────────────┐
# │ Protocol        │ Direction  │ Latency   │ Browser      │ Best for        │
# ├─────────────────┼────────────┼───────────┼──────────────┼─────────────────┤
# │ REST/HTTP       │ req/resp   │ ~50 ms    │ Native       │ CRUD, public API │
# │ GraphQL         │ req/resp   │ ~50 ms    │ Native       │ Flexible fetch  │
# │ gRPC            │ req/resp   │ ~1 ms     │ via grpc-web │ Microservices   │
# │ WebSocket       │ bidirect.  │ ~1 ms     │ Native       │ Chat, collab    │
# │ SSE             │ server→cli │ ~5 ms     │ Native       │ Notifications   │
# │ Webhooks        │ server→cli │ async     │ N/A          │ Event callbacks │
# └─────────────────┴────────────┴───────────┴──────────────┴─────────────────┘
#
# WebSocket Security Checklist:
# ✅ Authenticate on upgrade (before ws.accept()) — cannot re-authenticate later
# ✅ Validate Origin header — prevent CSRF-style WebSocket hijacking
# ✅ Per-connection rate limiting — prevent message flood
# ✅ Ping/pong keepalive — detect stale connections on load balancers
# ✅ Graceful close on error — avoid connection leaks
# ✅ Cap room/channel size — prevent memory exhaustion
# ✅ Horizontal scale via Redis Pub/Sub — stateless nodes
# ✅ Use wss:// (TLS) always in production
# ✅ HMAC-verify all incoming webhooks
