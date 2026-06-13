import asyncio
import threading

from assistant.bus import Event, EventBus


def test_publish_subscribe_round_trip() -> None:
    async def main() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        sub = bus.subscribe("speech_heard")
        bus.publish(Event("speech_heard", {"text": "hello"}))
        event = await asyncio.wait_for(sub.get(), timeout=1)
        assert event.type == "speech_heard"
        assert event.payload == {"text": "hello"}
        assert event.ts > 0

    asyncio.run(main())


def test_multiple_subscribers_each_receive() -> None:
    async def main() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        first = bus.subscribe("idle_tick")
        second = bus.subscribe("idle_tick")
        catch_all = bus.subscribe()
        bus.publish(Event("idle_tick"))
        for sub in (first, second, catch_all):
            event = await asyncio.wait_for(sub.get(), timeout=1)
            assert event.type == "idle_tick"

    asyncio.run(main())


def test_publish_from_plain_thread() -> None:
    async def main() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        sub = bus.subscribe("gaze")
        thread = threading.Thread(
            target=bus.publish, args=(Event("gaze", {"x": 0.5, "y": 0.25}),)
        )
        thread.start()
        event = await asyncio.wait_for(sub.get(), timeout=2)
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert event.type == "gaze"
        assert event.payload == {"x": 0.5, "y": 0.25}

    asyncio.run(main())


def test_inbox_filters_and_drains_for_sync_consumer() -> None:
    async def main() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        inbox = bus.open_inbox("face_state", "say")
        bus.publish(Event("face_state", {"state": "happy"}))
        bus.publish(Event("idle_tick"))
        bus.publish(Event("say", {"text": "hi"}))
        drained = inbox.drain()
        assert [event.type for event in drained] == ["face_state", "say"]
        assert inbox.drain() == []

    asyncio.run(main())


def test_inbox_blocking_get_returns_event() -> None:
    async def main() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        inbox = bus.open_inbox("say")
        bus.publish(Event("say", {"text": "hi"}))
        event = inbox.get(timeout=1)
        assert event is not None
        assert event.type == "say"

    asyncio.run(main())


def test_inbox_blocking_get_times_out_to_none() -> None:
    async def main() -> None:
        bus = EventBus()
        bus.attach_loop(asyncio.get_running_loop())
        inbox = bus.open_inbox("say")
        assert inbox.get(timeout=0.01) is None

    asyncio.run(main())
