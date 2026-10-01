"""Slow subscribers and blocked sockets must not retain unbounded updates."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from sse_starlette.sse import SendTimeoutError

from ml_exp_server.api.routes import stream as stream_route
from ml_exp_server.api.sse import EventBroker, IndexEventResponse


def test_capacity_validation_and_independent_slow_client_resync():
    with pytest.raises(ValueError):
        EventBroker(0)

    async def scenario():
        broker = EventBroker(2)
        fast, slow = broker.stream(), broker.stream()
        tasks = [asyncio.create_task(anext(source)) for source in (fast, slow)]
        await asyncio.sleep(0)
        broker._publish({'n': 0})
        assert await asyncio.gather(*tasks) == [{'data': '{"n": 0}'}] * 2
        for value in range(1, 10):
            broker._publish({'n': value})
            assert json.loads((await anext(fast))['data']) == {'n': value}
        assert len(broker._subscribers) == 1
        assert json.loads((await anext(slow))['data']) == {'type': 'resync_required'}
        with pytest.raises(StopAsyncIteration):
            await anext(slow)
        await fast.aclose()
        assert not broker._subscribers
    asyncio.run(scenario())


def test_thread_handoff_burst_is_bounded_and_requests_resync():
    async def scenario():
        broker = EventBroker(4)
        broker.bind_loop(asyncio.get_running_loop())
        source = broker.stream()
        pending = asyncio.create_task(anext(source))
        await asyncio.sleep(0)
        def burst():
            for value in range(10000):
                broker.publish_threadsafe({'n': value})
        thread = threading.Thread(target=burst)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert len(broker._pending) <= 4
        assert json.loads((await pending)['data']) == {'type': 'resync_required'}
        await source.aclose()
        assert not broker._subscribers
    asyncio.run(scenario())


@pytest.mark.parametrize('mode', ['data', 'ping', 'blocked_disconnect', 'blocked_timeout'])
def test_real_response_wire_heartbeat_timeout_and_disconnect_cleanup(mode):
    async def scenario():
        broker = EventBroker()
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(broker=broker)))
        response = await stream_route(request)
        assert isinstance(response, IndexEventResponse)
        assert response.ping_interval == 15 and response.send_timeout == 15
        response.ping_interval = 0.01 if mode == 'ping' else 0
        response.send_timeout = 0.03
        disconnect = asyncio.Event()
        messages = []
        async def receive():
            await disconnect.wait()
            return {'type': 'http.disconnect'}
        async def send(message):
            messages.append(message)
            if message['type'] == 'http.response.start':
                asyncio.get_running_loop().call_soon(broker._publish, {'type': 'index_updated'})
            elif message.get('body'):
                if mode == 'ping' and not message['body'].startswith(b':'):
                    return
                if mode != 'blocked_timeout':
                    disconnect.set()
                if mode.startswith('blocked'):
                    await asyncio.Event().wait()
        if mode == 'blocked_timeout':
            with pytest.raises(SendTimeoutError):
                await asyncio.wait_for(response({'type': 'http'}, receive, send), 1)
        else:
            await asyncio.wait_for(response({'type': 'http'}, receive, send), 1)
        bodies = b''.join(message.get('body', b'') for message in messages)
        assert b'data: {"type": "index_updated"}\n\n' in bodies
        if mode == 'ping':
            assert b': ping' in bodies
        assert not broker._subscribers
    asyncio.run(scenario())
