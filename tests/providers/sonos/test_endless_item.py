"""
Tests for a radio station or live source that a Sonos speaker plays as an item of its cloud queue.

Such a stream never ends by itself. A speaker that ran out of it (it stops reading an
uncompressed stream after 4 GB) reads its cloud queue to say it played the queue out and
reports the stream as lost: the queue restarts the item, once.
"""

import logging
from collections import deque
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from music_assistant_models.enums import MediaType
from music_assistant_models.player import PlayerMedia
from music_assistant_models.player_queue import PlayerQueue

from music_assistant.controllers.player_queues import PlayerQueuesController
from music_assistant.controllers.player_queues.state import PlayerQueueData
from music_assistant.providers.sonos.player import SonosPlayer
from tests.providers.sonos.test_cloud_queue import (
    QUEUE_ID,
    _error_report,
    _make_cloud_queue,
    _make_media_queue_item,
)
from tests.providers.sonos.test_stream_route import BASE_URL, SINGLE_URL, _make_speaker, _media

SESSION_ID = "session-2"


async def _play_item(
    monkeypatch: pytest.MonkeyPatch, *media_types: MediaType
) -> tuple[SonosPlayer, PlayerQueueData]:
    """
    Create a speaker that plays the first item of a queue held by the real queue controller.

    :param monkeypatch: The fixture that sets the flow mode of the speaker.
    :param media_types: The media type of each item of the queue, in order.
    :return: The speaker and the record of its queue.
    """
    player, group = _make_speaker(monkeypatch, live_sources_uncompressed=True)
    queue = PlayerQueue(
        queue_id=QUEUE_ID,
        active=True,
        display_name="Party",
        available=True,
        items=len(media_types),
    )
    queue_data = PlayerQueueData(queue=queue)
    queue_data.session_id = SESSION_ID
    queue_data.items = [
        _make_media_queue_item(f"item{i}", media_type) for i, media_type in enumerate(media_types)
    ]
    queue.current_index = 0
    queue.current_item = queue_data.items[0]
    queues = cast("Any", PlayerQueuesController.__new__(PlayerQueuesController))
    queues.logger = logging.getLogger("test.sonos.player_queues")
    queues.mass = player.mass
    queues._queue_data = {QUEUE_ID: queue_data}
    # what a restart does is the queue controller's: only that one is asked for counts here
    queues._restart_lost_flow_stream = MagicMock()
    queues.release_failed_item_source = MagicMock()
    mass = cast("MagicMock", player.mass)
    mass.player_queues = queues
    mass.players.get_player.return_value.offline_by_power_control = False
    await player.play_media(_media(media_types[0]))
    # a lossless stream without an end is an item of the cloud queue, as a track is
    group.play_cloud_queue.assert_awaited_once()
    player._attr_current_media = PlayerMedia(uri=f"{SINGLE_URL}.wav", queue_item_id="item0")
    player.reported_playback_errors = deque(maxlen=16)
    return player, queue_data


async def _report_lost_stream(player: SonosPlayer, report: str) -> None:
    """
    Let the speaker report that it lost the stream of the item it plays.

    :param player: The speaker.
    :param report: How it reports: over its websocket ("playback_error") or in its final
        report of the time it played ("time_played").
    """
    wire_id = cast("str", player.wire_item_id("item0"))
    if report == "time_played":
        request = MagicMock()
        error = {"type": "transport", "status": "ERROR_LOST_CONNECTION"}
        request.json = AsyncMock(
            return_value={"items": [{**_error_report("final"), "id": wire_id, "error": error}]}
        )
        await _make_cloud_queue()._handle_sonos_queue_time_played(player, request)
    else:
        cast("MagicMock", player.mass).streams.base_url = BASE_URL
        error = {
            "errorCode": "ERROR_PLAYBACK_FAILED",
            "reason": "ERROR_LOST_CONNECTION",
            "itemId": wire_id,
        }
        player._on_playback_error(MagicMock(data=error))


def _play_out(player: SonosPlayer) -> None:
    """Let the speaker read its cloud queue as it does once it played the queue out."""
    player.on_cloud_queue_read("queueCompleted", player.wire_item_id("item0"))


def _queues(player: SonosPlayer) -> Any:
    """Return the queue controller of the speaker."""
    return cast("Any", player.mass.player_queues)


@pytest.mark.parametrize("media_type", [MediaType.RADIO, MediaType.AUDIO_SOURCE])
async def test_a_station_the_speaker_played_out_restarts(
    monkeypatch: pytest.MonkeyPatch, media_type: MediaType
) -> None:
    """A speaker that ran out of a stream without an end does not ask for it again."""
    player, _ = await _play_item(monkeypatch, media_type)

    _play_out(player)

    _queues(player)._restart_lost_flow_stream.assert_called_once_with(
        QUEUE_ID, SESSION_ID, only_while_playing=False, queue_item_id="item0"
    )


async def test_the_end_of_a_track_restarts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A speaker that played the last track of its queue reached the end of the queue."""
    player, _ = await _play_item(monkeypatch, MediaType.TRACK)

    _play_out(player)

    _queues(player)._restart_lost_flow_stream.assert_not_called()


async def test_a_station_the_queue_moved_on_from_restarts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A station that is no longer the current item of the queue stays where it stopped."""
    player, queue_data = await _play_item(monkeypatch, MediaType.RADIO, MediaType.TRACK)
    queue_data.queue.current_item = queue_data.items[1]

    _play_out(player)

    _queues(player)._restart_lost_flow_stream.assert_not_called()


@pytest.mark.parametrize("report", ["playback_error", "time_played"])
async def test_a_lost_station_that_restarts_keeps_its_source(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    """The source a restart plays from is not released by a report of the lost stream."""
    player, _ = await _play_item(monkeypatch, MediaType.RADIO)
    _play_out(player)

    await _report_lost_stream(player, report)

    _queues(player).release_failed_item_source.assert_not_called()
    _queues(player)._restart_lost_flow_stream.assert_called_once()


async def test_the_signals_of_one_lost_station_restart_it_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The speaker reports the lost stream, plays its queue out, then sends its final report."""
    player, _ = await _play_item(monkeypatch, MediaType.RADIO)

    await _report_lost_stream(player, "playback_error")
    _play_out(player)
    await _report_lost_stream(player, "time_played")

    _queues(player)._restart_lost_flow_stream.assert_called_once_with(
        QUEUE_ID, SESSION_ID, only_while_playing=False, queue_item_id="item0"
    )
    # only the lost stream's own source, released before the restart was asked for
    _queues(player).release_failed_item_source.assert_called_once_with(QUEUE_ID, "item0")
