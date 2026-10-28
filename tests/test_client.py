import io
import time

import pytest

from csm_inference.client import pump_response


def test_pcm_is_delivered_before_next_network_read():
    sink = io.BytesIO()

    class Response:
        reads = 0

        def read1(self, size):
            self.reads += 1
            if self.reads == 1:
                return b"\x01\x00" * 320
            # The first chunk is already sent, before EOF or the next chunk.
            assert len(sink.getvalue()) == 640
            return b""

    metrics = pump_response(Response(), sink, time.perf_counter())
    assert metrics["audio_seconds"] == 320 / 24000


def test_truncated_http_body_is_not_treated_as_success():
    import http.client

    class Response:
        def read1(self, size):
            raise http.client.IncompleteRead(b"partial")

    with pytest.raises(http.client.IncompleteRead):
        pump_response(Response(), io.BytesIO(), time.perf_counter())


def test_partial_pcm_sample_is_rejected():
    class Response:
        parts = iter([b"\x00", b""])

        def read1(self, size):
            return next(self.parts)

    with pytest.raises(RuntimeError, match="incomplete PCM"):
        pump_response(Response(), io.BytesIO(), time.perf_counter())
