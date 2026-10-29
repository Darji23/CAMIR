"""Unit tests for the CAMIR CLI interface."""

from unittest.mock import patch
import pytest
from camir.cli import build_parser, main


def test_cli_version(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--version"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "camir 0.1.0" in captured.out or "camir 0.1.0" in captured.err


def test_cli_calibrate(capsys):
    ret = main(["calibrate", "--split", "validation"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Calibrating conformal risk threshold" in captured.out
    assert "tau" in captured.out


def test_cli_shadow(capsys):
    ret = main(["shadow", "--policy", "P4", "--samples", "10"])
    assert ret == 0
    captured = capsys.readouterr()
    assert "Shadow replay finished" in captured.out
    assert "Escalation tax" in captured.out


def test_cli_serve(capsys):
    with patch("uvicorn.run") as mock_run:
        ret = main(["serve", "--port", "9000"])
        assert ret == 0
        captured = capsys.readouterr()
        assert "Starting live CAMIR server" in captured.out
        mock_run.assert_called_once()


def test_cli_help_no_command(capsys):
    ret = main([])
    assert ret == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out or "camir" in captured.out


def test_cli_replay_command_parsing():
    parser = build_parser()
    args = parser.parse_args(["replay"])
    assert args.command == "replay"
