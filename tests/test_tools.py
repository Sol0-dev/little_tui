from pathlib import Path

import pytest

from little_tui.tools import ToolError, build_tools


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def tools(workspace: Path):
    return {t.name: t for t in build_tools(workspace)}


def test_path_escape_rejected(workspace: Path, tools) -> None:
    with pytest.raises(ToolError, match="escapes workspace"):
        tools["read_file"].handler({"path": "../secret.txt"})


def test_write_and_read_roundtrip(workspace: Path, tools) -> None:
    result = tools["write_file"].handler({"path": "a/b.txt", "content": "line1\nline2\nline3"})
    assert result["created"] is True
    assert (workspace / "a" / "b.txt").is_file()

    read = tools["read_file"].handler({"path": "a/b.txt"})
    assert read["total_lines"] == 3
    assert read["lines"] == ["1:line1", "2:line2", "3:line3"]


def test_read_offset_limit(workspace: Path, tools) -> None:
    tools["write_file"].handler({"path": "f.txt", "content": "\n".join(f"l{i}" for i in range(10))})
    read = tools["read_file"].handler({"path": "f.txt", "offset": 3, "limit": 3})
    assert read["lines"] == ["4:l3", "5:l4", "6:l5"]
    assert read["truncated"] is True


def test_edit_exactly_once(workspace: Path, tools) -> None:
    tools["write_file"].handler({"path": "f.txt", "content": "aaa bbb aaa"})
    with pytest.raises(ToolError, match="matches 2 times"):
        tools["edit_file"].handler({"path": "f.txt", "old_string": "aaa", "new_string": "x"})
    with pytest.raises(ToolError, match="not found"):
        tools["edit_file"].handler({"path": "f.txt", "old_string": "zzz", "new_string": "x"})
    ok = tools["edit_file"].handler({"path": "f.txt", "old_string": "bbb", "new_string": "CCC"})
    assert ok["replaced"] == 1
    assert (workspace / "f.txt").read_text() == "aaa CCC aaa"


def test_glob_skips_vendor_dirs(workspace: Path, tools) -> None:
    (workspace / "src").mkdir()
    (workspace / "src" / "main.py").write_text("")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "dep.js").write_text("")
    result = tools["glob"].handler({"pattern": "**/*"})
    assert "src/main.py" in result["matches"]
    assert not any("node_modules" in m for m in result["matches"])


def test_grep_finds_line_numbers(workspace: Path, tools) -> None:
    tools["write_file"].handler({"path": "f.py", "content": "def foo():\n    return 1\n"})
    result = tools["grep"].handler({"pattern": "return"})
    assert result["results"] == [{"file": "f.py", "line": 2, "text": "    return 1"}]


def test_shell_runs_in_workspace(workspace: Path, tools) -> None:
    result = tools["shell"].handler({"command": "pwd"})
    assert result["exit_code"] == 0
    assert str(workspace) in result["stdout"]


def test_shell_timeout(workspace: Path, tools) -> None:
    with pytest.raises(ToolError, match="timed out"):
        tools["shell"].handler({"command": "sleep 10", "timeout": 1})


def test_dangerous_flag(workspace: Path) -> None:
    specs = {t.name: t for t in build_tools(workspace)}
    assert specs["shell"].dangerous is True
    assert specs["read_file"].dangerous is False


def test_list_dir_sorts_dirs_first(workspace: Path, tools) -> None:
    (workspace / "zzz.txt").write_text("x")
    (workspace / "aaa").mkdir()
    result = tools["list_dir"].handler({})
    assert result["entries"][0] == "aaa/"
    assert result["entries"][1].startswith("zzz.txt")
    assert result["count"] == 2


def test_current_datetime(workspace: Path, tools) -> None:
    result = tools["current_datetime"].handler({})
    assert result["utc_iso8601"].endswith("Z") or "+00:00" in result["utc_iso8601"]
    assert isinstance(result["unix_seconds"], int)
    assert result["unix_seconds"] > 1_700_000_000
