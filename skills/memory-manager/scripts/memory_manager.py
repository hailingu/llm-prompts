#!/usr/bin/env python3
"""Memory manager aligned to the repository's current two-level standard."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


TOOLS_JSON_HELP = "Tools used (JSON array)"
SESSION_ID_HELP = "Session identifier"
THEME_NAME_HELP = "Theme name provided by the calling agent"
ENTRY_TYPE_L1_HELP = "Entry type for the L1 raw log"
SOURCE_LABEL_HELP = "Optional source label recorded in the CSV manifest"
THEME_REQUIRED_ERROR = "theme is required"
TEMPLATE_CHOICES = ["decision", "error", "task"]
EXCLUDED_THEME_DIRS = {"sessions", "data"}
DEFAULT_GLOBAL_MEMORY = "# Global Memory\n\n## Active Mission\n- None\n"


def _parse_json_arg(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _normalize_theme(theme: Optional[str], *, required: bool = False) -> Optional[str]:
    if theme is None:
        if required:
            raise ValueError(THEME_REQUIRED_ERROR)
        return None
    normalized = theme.strip().lower()
    if not normalized:
        if required:
            raise ValueError(THEME_REQUIRED_ERROR)
        return None
    if normalized == "auto":
        raise ValueError("auto theme detection has been removed; the calling agent must pass --theme explicitly")
    return normalized


def _normalize_csv_name(name: str) -> str:
    normalized = Path(name).name.strip()
    if not normalized:
        raise ValueError("CSV file name is required")
    if normalized != name.strip():
        raise ValueError("CSV file name must not contain directory components")
    if not normalized.lower().endswith(".csv"):
        normalized = f"{normalized}.csv"
    return normalized


def _read_csv_source(csv_content: Optional[str], source_file: Optional[str]) -> str:
    if bool(csv_content) == bool(source_file):
        raise ValueError("provide exactly one of --csv-content or --source-file")
    if csv_content is not None:
        return csv_content
    if source_file is None:
        raise ValueError("source file is required")
    return Path(source_file).read_text(encoding="utf-8")


def _preview_text_lines(content: str, line_count: int) -> str:
    if line_count <= 0:
        return ""
    lines = content.splitlines()
    return "\n".join(lines[:line_count])


class MemoryManager:
    """Minimal memory manager for repo-local L1/L2/L3 persistence."""

    def __init__(self, workspace_root: Optional[str] = None) -> None:
        self.workspace_root = Path(workspace_root or os.getcwd()).resolve()
        self.memory_dir = self.workspace_root / "memory"
        self.sessions_dir = self.memory_dir / "sessions"
        self.data_dir = self.memory_dir / "data"
        self.data_manifest_file = self.data_dir / "manifest.json"
        self.global_memory_file = self.memory_dir / "global.md"
        self._ensure_layout()

    def _ensure_layout(self) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if not self.data_manifest_file.exists():
            _write_text(self.data_manifest_file, "{}\n")
        if not self.global_memory_file.exists():
            _write_text(self.global_memory_file, DEFAULT_GLOBAL_MEMORY)

    def _now(self) -> dt.datetime:
        return dt.datetime.now()

    def _session_log_path(self, when: Optional[dt.datetime] = None) -> Path:
        when = when or self._now()
        return self.sessions_dir / f"{when.strftime('%Y-%m-%d')}.md"

    def _theme_dir(self, theme: str) -> Path:
        return self.memory_dir / theme.strip().lower()

    def _theme_path(self, theme: str, when: Optional[dt.datetime] = None) -> Path:
        when = when or self._now()
        return self._theme_dir(theme) / f"{when.strftime('%Y-%m-%d_%H')}.md"

    def _data_path(self, name: str) -> Path:
        return self.data_dir / _normalize_csv_name(name)

    def _read_data_manifest(self) -> Dict[str, Any]:
        if not self.data_manifest_file.exists():
            return {}
        content = _read_text(self.data_manifest_file).strip()
        if not content:
            return {}
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("CSV manifest must be a JSON object")
        return parsed

    def _write_data_manifest(self, manifest: Dict[str, Any]) -> None:
        _write_text(self.data_manifest_file, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    def _csv_summary(self, csv_content: str) -> Dict[str, Any]:
        rows = list(csv.reader(csv_content.splitlines()))
        if not rows:
            return {"row_count": 0, "column_count": 0, "columns": []}
        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else []
        return {
            "row_count": len(data_rows),
            "column_count": len(header),
            "columns": header,
        }

    def _upsert_data_manifest_entry(
        self,
        name: str,
        csv_content: str,
        source_label: Optional[str] = None,
        description: Optional[str] = None,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        path = self._data_path(name)
        manifest = self._read_data_manifest()
        summary = self._csv_summary(csv_content)
        manifest[path.name] = {
            "name": path.name,
            "path": str(path),
            "updated_at": self._now().isoformat(timespec="seconds"),
            "source_label": source_label,
            "description": description,
            "declared_columns": columns or None,
            "detected_columns": summary["columns"],
            "row_count": summary["row_count"],
            "column_count": summary["column_count"],
            "size_bytes": len(csv_content.encode("utf-8")),
        }
        self._write_data_manifest(manifest)
        return manifest[path.name]

    def _is_global_worthy(self, content: str, theme: str) -> bool:
        if theme != "preferences":
            return False
        lowered = content.lower()
        durable_markers = [
            "remember",
            "default",
            "always",
            "preference",
            "constraint",
            "记住",
            "不要忘记",
            "默认",
            "长期",
            "以后",
            "约束",
            "规则",
        ]
        return any(marker in lowered or marker in content for marker in durable_markers)

    def append_session_log(
        self,
        entry_type: str,
        content: str,
        tools_used: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        when: Optional[dt.datetime] = None,
    ) -> str:
        when = when or self._now()
        path = self._session_log_path(when)
        header = f"# Session Log - {when.strftime('%Y-%m-%d')}\n\n"
        entry_lines = [f"### [{when.strftime('%Y-%m-%d %H:%M:%S')}] {entry_type.upper()}"]
        if session_id:
            entry_lines.append(f"**Session:** {session_id}")
        if tools_used:
            entry_lines.append(f"**Tools:** {', '.join(tools_used)}")
        entry_lines.append("")
        entry_lines.append(content.strip())
        entry_lines.append("")
        entry_lines.append("---")
        entry = "\n".join(entry_lines)

        if path.exists():
            _append_text(path, f"\n\n{entry}")
        else:
            _write_text(path, header + entry)
        return str(path)

    def read_session_logs(self, days_back: int = 7, limit: int = 20) -> List[Dict[str, Any]]:
        cutoff = self._now() - dt.timedelta(days=days_back)
        records: List[Dict[str, Any]] = []
        for path in sorted(self.sessions_dir.glob("*.md"), reverse=True):
            try:
                file_date = dt.datetime.strptime(path.stem, "%Y-%m-%d")
            except ValueError:
                continue
            if file_date < cutoff:
                continue
            content = _read_text(path)
            records.append(
                {
                    "date": path.stem,
                    "path": str(path),
                    "size": len(content),
                    "content": content,
                }
            )
            if len(records) >= limit:
                break
        return records

    def _format_theme_entry(
        self,
        content: str,
        template: Optional[str] = None,
        title: Optional[str] = None,
        when: Optional[dt.datetime] = None,
    ) -> str:
        when = when or self._now()
        timestamp = when.strftime("%Y-%m-%d %H:%M:%S")
        template_titles = {
            "decision": "Decision Record",
            "error": "Error Record",
            "task": "Task Record",
        }
        if title and template:
            heading = f"{template_titles[template]}: {title}"
        elif title:
            heading = title
        elif template:
            heading = f"{template_titles[template]} - {timestamp}"
        else:
            heading = timestamp
        return f"## {heading}\n\n{content.strip()}\n"

    def write_theme_memory(
        self,
        theme: str,
        content: str,
        template: Optional[str] = None,
        title: Optional[str] = None,
        when: Optional[dt.datetime] = None,
    ) -> str:
        resolved_theme = _normalize_theme(theme, required=True)
        if resolved_theme is None:
            raise ValueError(THEME_REQUIRED_ERROR)
        when = when or self._now()
        path = self._theme_path(resolved_theme, when)
        entry = self._format_theme_entry(content, template=template, title=title, when=when)
        if path.exists():
            _append_text(path, f"\n\n{entry}")
        else:
            _write_text(path, entry)
        return str(path)

    def write_data_memory(self, name: str, csv_content: str, replace: bool = False) -> str:
        path = self._data_path(name)
        if path.exists() and not replace:
            raise ValueError(f"CSV data file already exists: {path.name}. Use --replace to overwrite it")
        _write_text(path, csv_content)
        return str(path)

    def read_data_memory(self, name: str, head: Optional[int] = None) -> Dict[str, Any]:
        path = self._data_path(name)
        if not path.exists():
            raise ValueError(f"CSV data file does not exist: {path.name}")
        content = _read_text(path)
        manifest = self._read_data_manifest()
        return {
            "name": path.name,
            "path": str(path),
            "size": path.stat().st_size,
            "metadata": manifest.get(path.name),
            "content": _preview_text_lines(content, head) if head is not None else content,
            "truncated": head is not None,
        }

    def list_data_files(self) -> List[Dict[str, Any]]:
        manifest = self._read_data_manifest()
        records: List[Dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("*.csv")):
            record = {
                "name": path.name,
                "path": str(path),
                "size": path.stat().st_size,
            }
            metadata = manifest.get(path.name)
            if metadata:
                record["metadata"] = metadata
            records.append(record)
        return records

    def read_theme_memory(self, theme: str, hours_back: int = 24, limit: int = 20) -> List[Dict[str, Any]]:
        directory = self._theme_dir(theme)
        if not directory.exists():
            return []
        cutoff = self._now() - dt.timedelta(hours=hours_back)
        records: List[Dict[str, Any]] = []
        for path in sorted(directory.glob("*.md"), reverse=True):
            try:
                file_time = dt.datetime.strptime(path.stem, "%Y-%m-%d_%H")
            except ValueError:
                continue
            if file_time < cutoff:
                continue
            records.append(
                {
                    "timestamp": file_time.isoformat(),
                    "path": str(path),
                    "content": _read_text(path),
                }
            )
            if len(records) >= limit:
                break
        return records

    def list_themes(self) -> List[str]:
        return sorted(
            path.name
            for path in self.memory_dir.iterdir()
            if path.is_dir() and path.name not in EXCLUDED_THEME_DIRS and not path.name.startswith(".")
        )

    def read_global_memory(self) -> str:
        return _read_text(self.global_memory_file) if self.global_memory_file.exists() else DEFAULT_GLOBAL_MEMORY

    def write_global_memory(self, content: str, append: bool = True) -> str:
        normalized = content.strip()
        if append and self.global_memory_file.exists():
            _append_text(self.global_memory_file, f"\n\n{normalized}")
        else:
            _write_text(self.global_memory_file, normalized + "\n")
        return str(self.global_memory_file)

    def _recent_theme_index(self, hours_back: int, limit: int) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for theme in self.list_themes():
            memories = self.read_theme_memory(theme, hours_back=hours_back, limit=1)
            if not memories:
                continue
            latest = memories[0]
            records.append(
                {
                    "theme": theme,
                    "timestamp": latest["timestamp"],
                    "path": latest["path"],
                    "preview": latest["content"].strip().replace("\n", " ")[:160],
                }
            )
        records.sort(key=lambda item: item["timestamp"], reverse=True)
        return records[:limit]

    def session_init(
        self,
        session_id: Optional[str] = None,
        recent_days: int = 7,
        theme_limit: int = 8,
        write_log: bool = True,
    ) -> Dict[str, Any]:
        result = {
            "status": "success",
            "global_path": str(self.global_memory_file),
            "global_excerpt": self.read_global_memory()[:2000],
            "recent_logs": [
                {"date": item["date"], "path": item["path"], "size": item["size"]}
                for item in self.read_session_logs(days_back=recent_days, limit=3)
            ],
            "recent_themes": self._recent_theme_index(hours_back=recent_days * 24, limit=theme_limit),
            "log_written": False,
        }
        if write_log:
            result["log_path"] = self.append_session_log(
                entry_type="session_init",
                content="Session initialized and memory context loaded.",
                session_id=session_id,
            )
            result["log_written"] = True
        return result

    def persist_turn(
        self,
        raw_content: str,
        entry_type: str = "turn",
        tools_used: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        extracted_content: Optional[str] = None,
        theme: str = "auto",
        template: Optional[str] = None,
        title: Optional[str] = None,
        promote_global: bool = False,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "success",
            "l1_written": True,
            "l1_path": self.append_session_log(
                entry_type=entry_type,
                content=raw_content,
                tools_used=tools_used,
                session_id=session_id,
            ),
            "l2_written": False,
            "l2_path": None,
            "theme": None,
            "global_updated": False,
            "global_path": None,
        }
        if not extracted_content:
            return result

        resolved_theme = _normalize_theme(theme, required=True)
        if resolved_theme is None:
            raise ValueError(THEME_REQUIRED_ERROR)
        result["l2_path"] = self.write_theme_memory(
            theme=resolved_theme,
            content=extracted_content,
            template=template,
            title=title,
        )
        result["l2_written"] = True
        result["theme"] = resolved_theme

        if promote_global or self._is_global_worthy(extracted_content, resolved_theme):
            timestamp = self._now().strftime("%Y-%m-%d %H:%M")
            result["global_path"] = self.write_global_memory(
                f"## Persisted Turn Promotion - {timestamp}\n\n{extracted_content.strip()}",
                append=True,
            )
            result["global_updated"] = True

        return result


def _handle_session_init(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    return manager.session_init(
        session_id=args.session_id,
        recent_days=args.recent_days,
        theme_limit=args.theme_limit,
        write_log=not args.no_log,
    )


def _handle_log_turn(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "success",
        "path": manager.append_session_log(
            entry_type=args.entry_type,
            content=args.content,
            tools_used=_parse_json_arg(args.tools, []),
            session_id=args.session_id,
        ),
    }


def _handle_read_logs(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "success",
        "logs": manager.read_session_logs(days_back=args.days_back, limit=args.limit),
    }


def _handle_persist_turn(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    return manager.persist_turn(
        raw_content=args.raw_content,
        entry_type=args.entry_type,
        tools_used=_parse_json_arg(args.tools, []),
        session_id=args.session_id,
        extracted_content=args.extracted_content,
        theme=args.theme,
        template=args.template,
        title=args.title,
        promote_global=args.promote_global,
    )


def _handle_write_theme(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    l1_path = manager.append_session_log(
        entry_type=args.entry_type,
        content=args.raw_content or args.content,
        tools_used=_parse_json_arg(args.tools, []),
        session_id=args.session_id,
    )
    resolved_theme = _normalize_theme(args.theme, required=True)
    if resolved_theme is None:
        raise ValueError(THEME_REQUIRED_ERROR)
    result = {
        "status": "success",
        "l1_written": True,
        "l1_path": l1_path,
        "path": manager.write_theme_memory(
            theme=resolved_theme,
            content=args.content,
            template=args.template,
            title=args.title,
        ),
        "theme": resolved_theme,
        "global_updated": False,
        "global_path": None,
    }
    if args.promote_global or manager._is_global_worthy(args.content, resolved_theme):
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        result["global_path"] = manager.write_global_memory(
            f"## Theme Memory Promotion - {timestamp}\n\n{args.content.strip()}",
            append=True,
        )
        result["global_updated"] = True
    return result


def _handle_read_theme(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "success",
        "memories": manager.read_theme_memory(
            theme=args.theme,
            hours_back=args.hours_back,
            limit=args.limit,
        ),
    }


def _handle_read_global(manager: MemoryManager, _args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "success",
        "path": str(manager.global_memory_file),
        "content": manager.read_global_memory(),
    }


def _handle_write_global(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    l1_path = manager.append_session_log(
        entry_type=args.entry_type,
        content=args.raw_content or args.content,
        tools_used=_parse_json_arg(args.tools, []),
        session_id=args.session_id,
    )
    return {
        "status": "success",
        "l1_written": True,
        "l1_path": l1_path,
        "path": manager.write_global_memory(args.content, append=args.append),
    }


def _handle_list_themes(manager: MemoryManager, _args: argparse.Namespace) -> Dict[str, Any]:
    return {"status": "success", "themes": manager.list_themes()}


def _handle_write_data(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    csv_content = _read_csv_source(args.csv_content, args.source_file)
    raw_log = args.raw_content or f"Stored CSV memory data file {_normalize_csv_name(args.name)}"
    data_path = manager.write_data_memory(
        name=args.name,
        csv_content=csv_content,
        replace=args.replace,
    )
    manifest_entry = manager._upsert_data_manifest_entry(
        name=args.name,
        csv_content=csv_content,
        source_label=args.source_label,
        description=args.description,
        columns=_parse_json_arg(args.columns_json, None),
    )
    return {
        "status": "success",
        "l1_written": True,
        "l1_path": manager.append_session_log(
            entry_type=args.entry_type,
            content=raw_log,
            tools_used=_parse_json_arg(args.tools, []),
            session_id=args.session_id,
        ),
        "data_path": data_path,
        "manifest_entry": manifest_entry,
    }


def _handle_list_data(manager: MemoryManager, _args: argparse.Namespace) -> Dict[str, Any]:
    return {"status": "success", "data_files": manager.list_data_files()}


def _handle_read_data(manager: MemoryManager, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "status": "success",
        "data": manager.read_data_memory(name=args.name, head=args.head),
    }


COMMAND_HANDLERS = {
    "session-init": _handle_session_init,
    "log-turn": _handle_log_turn,
    "read-logs": _handle_read_logs,
    "persist-turn": _handle_persist_turn,
    "write-theme": _handle_write_theme,
    "read-theme": _handle_read_theme,
    "read-global": _handle_read_global,
    "write-global": _handle_write_global,
    "list-themes": _handle_list_themes,
    "write-data": _handle_write_data,
    "list-data": _handle_list_data,
    "read-data": _handle_read_data,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory Manager for repo-local L1/L2/L3 persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s session-init --session-id abc
  %(prog)s log-turn --entry-type user --content \"User asked for a plan\"
  %(prog)s persist-turn --entry-type assistant --raw-content \"Full raw turn\" \\
    --extracted-content \"Decision: standardize on persist-turn\" --theme decision
    %(prog)s write-theme --theme preferences --content \"User prefers concise conclusions first\" --promote-global
  %(prog)s read-theme --theme research --hours-back 72
  %(prog)s read-global
    %(prog)s write-data --name metrics.csv --csv-content \"date,value\\n2026-03-07,1\" --description \"Sample metric"
    %(prog)s read-data --name metrics.csv --head 5
  %(prog)s list-themes
        """,
    )
    parser.add_argument(
        "--workspace",
        help="Workspace root path (defaults to current working directory)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    session_init_parser = subparsers.add_parser(
        "session-init",
        help="Load global memory and recent memory index for a new session",
    )
    session_init_parser.add_argument("--session-id", help=SESSION_ID_HELP)
    session_init_parser.add_argument("--recent-days", type=int, default=7, help="Days of recent context to inspect")
    session_init_parser.add_argument("--theme-limit", type=int, default=8, help="Maximum recent themes to return")
    session_init_parser.add_argument("--no-log", action="store_true", help="Do not write a session_init entry to L1")

    log_turn_parser = subparsers.add_parser("log-turn", help="Write raw content to L1 only")
    log_turn_parser.add_argument("--entry-type", default="turn", help="Entry type")
    log_turn_parser.add_argument("--content", required=True, help="Content to log")
    log_turn_parser.add_argument("--tools", help=TOOLS_JSON_HELP)
    log_turn_parser.add_argument("--session-id", help=SESSION_ID_HELP)

    read_logs_parser = subparsers.add_parser("read-logs", help="Read recent L1 session logs")
    read_logs_parser.add_argument("--days-back", type=int, default=7, help="Days to look back")
    read_logs_parser.add_argument("--limit", type=int, default=20, help="Maximum log files to return")

    persist_turn_parser = subparsers.add_parser(
        "persist-turn",
        help="Write raw L1 content first, then optional extracted L2 content",
    )
    persist_turn_parser.add_argument("--entry-type", default="turn", help=ENTRY_TYPE_L1_HELP)
    persist_turn_parser.add_argument("--raw-content", required=True, help="Raw interaction content for L1")
    persist_turn_parser.add_argument("--extracted-content", help="Optional reusable content for L2")
    persist_turn_parser.add_argument("--theme", help=THEME_NAME_HELP)
    persist_turn_parser.add_argument("--template", choices=TEMPLATE_CHOICES, help="Optional L2 template")
    persist_turn_parser.add_argument("--title", help="Optional L2 heading title")
    persist_turn_parser.add_argument("--tools", help=TOOLS_JSON_HELP)
    persist_turn_parser.add_argument("--session-id", help=SESSION_ID_HELP)
    persist_turn_parser.add_argument("--promote-global", action="store_true", help="Also append extracted content to global memory")

    write_theme_parser = subparsers.add_parser("write-theme", help="Write extracted content to L2")
    write_theme_parser.add_argument("--theme", required=True, help=THEME_NAME_HELP)
    write_theme_parser.add_argument("--content", required=True, help="Content to write")
    write_theme_parser.add_argument("--template", choices=TEMPLATE_CHOICES, help="Optional L2 template")
    write_theme_parser.add_argument("--title", help="Optional heading title")
    write_theme_parser.add_argument("--promote-global", action="store_true", help="Also append content to global memory")
    write_theme_parser.add_argument("--raw-content", help="Optional L1 raw content; defaults to --content")
    write_theme_parser.add_argument("--entry-type", default="turn", help=ENTRY_TYPE_L1_HELP)
    write_theme_parser.add_argument("--tools", help=TOOLS_JSON_HELP)
    write_theme_parser.add_argument("--session-id", help=SESSION_ID_HELP)

    read_theme_parser = subparsers.add_parser("read-theme", help="Read recent L2 theme memory")
    read_theme_parser.add_argument("--theme", required=True, help=THEME_NAME_HELP)
    read_theme_parser.add_argument("--hours-back", type=int, default=24, help="Hours to look back")
    read_theme_parser.add_argument("--limit", type=int, default=20, help="Maximum files to return")

    subparsers.add_parser("read-global", help="Read global memory")

    write_global_parser = subparsers.add_parser("write-global", help="Write or append to global memory")
    write_global_parser.add_argument("--content", required=True, help="Content to write")
    write_global_parser.add_argument("--append", action="store_true", help="Append instead of replace")
    write_global_parser.add_argument("--raw-content", help="Optional L1 raw content; defaults to --content")
    write_global_parser.add_argument("--entry-type", default="turn", help=ENTRY_TYPE_L1_HELP)
    write_global_parser.add_argument("--tools", help=TOOLS_JSON_HELP)
    write_global_parser.add_argument("--session-id", help=SESSION_ID_HELP)

    subparsers.add_parser("list-themes", help="List all L2 themes")
    write_data_parser = subparsers.add_parser("write-data", help="Write a CSV data file to memory/data and log the action to L1")
    write_data_parser.add_argument("--name", required=True, help="CSV file name")
    write_data_parser.add_argument("--csv-content", help="CSV content to store")
    write_data_parser.add_argument("--source-file", help="Existing CSV file to copy into memory/data")
    write_data_parser.add_argument("--replace", action="store_true", help="Overwrite an existing CSV data file")
    write_data_parser.add_argument("--description", help="Optional manifest description for the CSV dataset")
    write_data_parser.add_argument("--source-label", help=SOURCE_LABEL_HELP)
    write_data_parser.add_argument("--columns-json", help="Optional JSON array describing expected columns")
    write_data_parser.add_argument("--raw-content", help="Optional L1 raw content for the write action")
    write_data_parser.add_argument("--entry-type", default="turn", help=ENTRY_TYPE_L1_HELP)
    write_data_parser.add_argument("--tools", help=TOOLS_JSON_HELP)
    write_data_parser.add_argument("--session-id", help=SESSION_ID_HELP)

    read_data_parser = subparsers.add_parser("read-data", help="Read a CSV data file stored under memory/data")
    read_data_parser.add_argument("--name", required=True, help="CSV file name")
    read_data_parser.add_argument("--head", type=int, help="Return only the first N lines of the CSV file")

    subparsers.add_parser("list-data", help="List CSV data files stored under memory/data")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = MemoryManager(args.workspace)
    try:
        result = COMMAND_HANDLERS[args.command](manager, args)
    except Exception as error:
        print(json.dumps({"status": "error", "message": str(error)}, ensure_ascii=False, indent=2))
        sys.exit(1)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
