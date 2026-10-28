import json

import pytest

from csm_inference import cli


def test_file_command_preserves_transcript_and_measurements(tmp_path, monkeypatch, capsys):
    args = cli.parser().parse_args(["say", "Hello from CSM.", "-o", str(tmp_path / "hello.wav")])

    def synthesize(text, args):
        args.output.write_bytes(b"test audio")
        return {"text": text, "audio_seconds": 1}

    monkeypatch.setattr(cli, "synthesize", synthesize)
    assert cli.run(args) == 0
    assert capsys.readouterr().out == "Hello from CSM.\n"
    assert (tmp_path / "hello.txt").read_text() == "Hello from CSM.\n"
    assert json.loads((tmp_path / "hello.json").read_text())["audio_seconds"] == 1


@pytest.mark.parametrize("options", [[], ["--reference-audio", "missing.wav"]])
def test_invalid_input_rejected_before_inference(options, monkeypatch):
    args = cli.parser().parse_args(["say", "" if not options else "Hello", *options])
    monkeypatch.setattr(cli, "synthesize", lambda *a: pytest.fail("Inference should not run"))
    with pytest.raises(ValueError):
        cli.run(args)


@pytest.mark.parametrize("limit", ["0", "1501"])
def test_frame_limits(limit):
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["say", "Hello", "--max-frames", limit])


@pytest.mark.parametrize("command", ["text", "respond"])
def test_only_speech_commands_are_exposed(command):
    with pytest.raises(SystemExit):
        cli.parser().parse_args([command, "Hello"])


def test_stream_command_dispatches_to_incremental_client(monkeypatch):
    from csm_inference import client

    calls = []
    monkeypatch.setattr(client, "stream_text", lambda text, **kw: calls.append((text, kw)) or {})
    args = cli.parser().parse_args(["stream", "Hello", "--max-frames", "100"])
    assert cli.run(args) == 0
    assert calls == [("Hello", {"port": 7861, "seed": 42, "max_frames": 100})]


def test_compatibility_engine_flag():
    args = cli.parser().parse_args(["say", "Hello", "--fast-depth"])
    assert args.engine == "direct-depth"
