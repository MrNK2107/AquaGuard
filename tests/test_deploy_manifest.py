"""The HF Space upload list must contain every app module the Space UI imports.

Parsed with ast, so huggingface_hub is not needed (and nothing is uploaded).
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _uploaded():
    tree = ast.parse((ROOT / "scripts" / "deploy_to_hf.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "files_to_upload" for t in node.targets):
            return {elt.elts[1].value for elt in node.value.elts}
    raise AssertionError("files_to_upload not found")


def _app_imports(path: Path, seen=None):
    seen = seen if seen is not None else set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.ImportFrom):
            if node.level and path.parent.name == "app":
                mods = [f"app.{node.module}"] if node.module else []
            elif node.module:
                mods = [node.module]
        elif isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        for m in mods:
            if m.startswith("app.") and m not in seen:
                seen.add(m)
                _app_imports(ROOT / (m.replace(".", "/") + ".py"), seen)
    return seen


def test_space_upload_covers_ui_imports():
    uploaded = _uploaded()
    needed = {m.replace(".", "/") + ".py" for m in _app_imports(ROOT / "app_ui.py")} | {"app/__init__.py", "reasoning.py"}
    missing = needed - uploaded
    assert not missing, f"deploy_to_hf.py would not upload: {sorted(missing)}"
