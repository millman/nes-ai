from __future__ import annotations

import argparse
import ast
import importlib.util
import importlib.machinery
import types
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Set, List

import tomli
import tomli_w

from jepa_world_model.metadata import write_run_metadata


def _run_git(args: list[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(cwd) if cwd else None,
    )
    output = result.stdout.strip() or result.stderr.strip()
    if result.returncode != 0:
        raise RuntimeError(f"git command failed ({result.returncode}): {' '.join(args)} :: {output}")
    return output


def _extract_commit_from_metadata(metadata_git_text: str) -> str:
    lines = metadata_git_text.splitlines()
    seen_header = False
    for line in lines:
        stripped = line.strip()
        if not stripped and not seen_header:
            continue
        if not seen_header:
            if stripped.lower().startswith("git commit"):
                seen_header = True
            continue
        if stripped:
            return stripped
    raise ValueError("Unable to extract git commit from metadata_git.txt")


def _collect_imports(tree: ast.AST) -> List[Tuple[str, Optional[str]]]:
    imports: List[Tuple[str, Optional[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    module = alias.name
                    imports.append((module, None))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base = node.module
                for alias in node.names:
                    imports.append((base, alias.name))
    return imports


def _module_path_from_import(module: str, repo_root: Path) -> Optional[Path]:
    candidate = repo_root / (module.replace(".", "/") + ".py")
    if candidate.exists():
        return candidate
    return None


def _is_stdlib_module(module_root: str) -> bool:
    if hasattr(sys, "stdlib_module_names"):
        return module_root in sys.stdlib_module_names  # type: ignore[attr-defined]
    spec = importlib.util.find_spec(module_root)
    if spec is None or spec.origin is None:
        return False
    return "site-packages" not in spec.origin and "dist-packages" not in spec.origin


def _make_stub_module(name: str, is_package: bool = True) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    mod.__spec__ = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=is_package)
    return mod


def _defined_names(node: ast.AST) -> Set[str]:
    names: Set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.asname:
                names.add(alias.asname)
            elif alias.name:
                names.add(alias.name.split(".")[0])
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.asname:
                names.add(alias.asname)
            elif alias.name:
                names.add(alias.name)
    return names


def _names_in_node(node: ast.AST) -> Set[str]:
    builtins = set(dir(__builtins__))
    refs: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id not in builtins:
                refs.add(child.id)
        elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            if child.value.id not in builtins:
                refs.add(child.value.id)
        elif isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                if func.id not in builtins:
                    refs.add(func.id)
            elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id not in builtins:
                    refs.add(func.value.id)
    return refs


def _extract_config_ast(trainer_cache_path: Path) -> ast.Module:
    source = trainer_cache_path.read_text()
    tree = ast.parse(source)
    future_imports: List[ast.stmt] = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    definitions: Dict[str, ast.AST] = {}
    for node in tree.body:
        for name in _defined_names(node):
            definitions.setdefault(name, node)

    required: Set[str] = {"TrainConfig", "ModelConfig"}
    kept_nodes: Set[ast.AST] = set()
    changed = True
    while changed:
        changed = False
        for name in list(required):
            node = definitions.get(name)
            if node is None or node in kept_nodes:
                continue
            kept_nodes.add(node)
            refs = _names_in_node(node)
            for ref in refs:
                if ref not in required and ref in definitions:
                    required.add(ref)
                    changed = True

    filtered_body: List[ast.stmt] = []
    for node in tree.body:
        if node in kept_nodes:
            filtered_body.append(node)
    module_body = future_imports + filtered_body
    return ast.Module(body=module_body, type_ignores=[])


def _stage_dependencies(trainer_cache_path: Path, repo_root: Path) -> Dict[str, Set[Optional[str]]]:
    config_ast = _extract_config_ast(trainer_cache_path)
    stub_map: Dict[str, Set[Optional[str]]] = {}
    for module, attr in _collect_imports(config_ast):
        module_root = module.split(".")[0]
        if _is_stdlib_module(module_root):
            continue
        local_path = _module_path_from_import(module, repo_root)
        if local_path is None:
            stub_map.setdefault(module, set()).add(attr)
    return stub_map


def _install_stub_chain(module: str, attrs: Set[Optional[str]], inserted: List[str]) -> None:
    parts = module.split(".")
    parent = None
    for idx in range(len(parts)):
        name = ".".join(parts[: idx + 1])
        if name in sys.modules:
            parent = sys.modules[name]
            continue
        stub_mod = _make_stub_module(name, is_package=True)
        sys.modules[name] = stub_mod
        inserted.append(name)
        if parent is not None:
            setattr(parent, parts[idx], stub_mod)
        parent = stub_mod
    target = sys.modules[module]
    for attr in attrs:
        if not attr:
            continue
        if hasattr(target, attr):
            continue
        sub_name = f"{module}.{attr}"
        sub_mod = _make_stub_module(sub_name, is_package=True)
        setattr(target, attr, sub_mod)
        sys.modules[sub_name] = sub_mod
        inserted.append(sub_name)


def _load_configs_from_source(module_path: Path, repo_root: Path, stubs: Dict[str, Set[Optional[str]]]) -> Tuple[Any, Any]:
    sys_path_added = False
    inserted_modules: List[str] = []
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
        sys_path_added = True
    # Install stub modules for missing imports
    for mod, attrs in stubs.items():
        if mod in sys.modules:
            continue
        _install_stub_chain(mod, attrs, inserted_modules)
    # Parse only the config-relevant portion
    config_ast = _extract_config_ast(module_path)
    compiled = compile(config_ast, str(module_path), "exec")
    module = types.ModuleType("trainer_checked_in")
    module.__file__ = str(module_path)
    module_inserted = False
    if module.__name__ not in sys.modules:
        sys.modules[module.__name__] = module
        module_inserted = True
    try:
        exec(compiled, module.__dict__)
    finally:
        if sys_path_added:
            try:
                sys.path.remove(str(repo_root))
            except ValueError:
                pass
        for mod in inserted_modules:
            sys.modules.pop(mod, None)
        if module_inserted:
            sys.modules.pop(module.__name__, None)
    if not hasattr(module, "TrainConfig") or not hasattr(module, "ModelConfig"):
        raise RuntimeError("Trainer module missing TrainConfig or ModelConfig")
    return module.TrainConfig(), module.ModelConfig()


def _write_base_metadata(
    commit: str,
    repo_root: Path,
    trainer_path: str,
    cache_root: Path,
    exp_id: str,
) -> Path:
    base_root = cache_root / exp_id
    target_dir = base_root / commit
    metadata_path = base_root / "metadata.txt"
    if metadata_path.exists():
        return base_root
    source = _run_git(["git", "show", f"{commit}:{trainer_path}"], cwd=repo_root)
    if not source:
        raise RuntimeError(f"git show returned empty content for {commit}:{trainer_path}")
    trainer_cache_path = target_dir / trainer_path
    trainer_cache_path.parent.mkdir(parents=True, exist_ok=True)
    trainer_cache_path.write_text(source)
    # Collect stub mapping after staging dependencies
    stub_map = _stage_dependencies(trainer_cache_path, repo_root)
    train_cfg, model_cfg = _load_configs_from_source(trainer_cache_path, repo_root, stub_map)
    base_root.mkdir(parents=True, exist_ok=True)
    write_run_metadata(base_root, train_cfg, model_cfg, exclude_fields={"title"})
    if not metadata_path.exists():
        raise RuntimeError("metadata.txt missing after write_run_metadata")
    return base_root


def _read_configs_from_metadata(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = tomli.loads(path.read_text())
    train_cfg = data.get("train_config")
    model_cfg = data.get("model_config")
    if train_cfg is None or model_cfg is None:
        raise ValueError(f"{path} missing train_config or model_config")
    return train_cfg, model_cfg


def _collect_changed_leaves(current: Any, base: Any, path: List[str], out: List[Tuple[str, Any]]) -> None:
    # Recurse through dictionaries; treat other mismatches as leaf differences.
    if isinstance(current, dict) and isinstance(base, dict):
        keys = set(current.keys()) | set(base.keys())
        for key in sorted(keys):
            _collect_changed_leaves(current.get(key), base.get(key), path + [key], out)
        return
    if current != base:
        leaf = path[-1] if path else ""
        out.append((leaf, current))


def generate_model_diff(exp_dir: Path, trainer_path: str = "jepa_world_model_trainer.py", repo_root: Optional[Path] = None) -> Path:
    # --- Cache setup ---
    cache_dir = exp_dir / "server_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "model_diff.txt"
    metadata_path = exp_dir / "metadata.txt"
    metadata_git_path = exp_dir / "metadata_git.txt"

    if cache_path.exists():
        return cache_path

    # --- Read current metadata ---
    train_current, model_current = _read_configs_from_metadata(metadata_path)
    base_commit = _extract_commit_from_metadata(metadata_git_text=metadata_git_path.read_text())

    # --- Resolve git roots and cache ---
    git_root_text = _run_git(["git", "rev-parse", "--show-toplevel"], cwd=repo_root)
    git_root = Path(git_root_text)
    git_cache_root = Path(__file__).resolve().parent / "out.git_cache"
    git_cache_root.mkdir(parents=True, exist_ok=True)
    trainer_rel = str((git_root / trainer_path).relative_to(git_root))

    # --- Prepare base metadata ---
    base_dir = _write_base_metadata(base_commit, git_root, trainer_rel, git_cache_root, exp_dir.name)
    base_metadata_path = base_dir / "metadata.txt"
    train_base, model_base = _read_configs_from_metadata(base_metadata_path)

    # --- Compute diff as single-line CSV of leaf changes ---
    changed: List[Tuple[str, Any]] = []
    _collect_changed_leaves(train_current, train_base, ["train_config"], changed)
    _collect_changed_leaves(model_current, model_base, ["model_config"], changed)
    entries: List[str] = []
    for leaf, value in changed:
        if leaf:
            entries.append(f"{leaf}={value}")
    if not entries:
        entries.append("status=no_diff")
    cache_path.write_text("\n".join(entries) + "\n")
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model diff for an experiment directory.")
    parser.add_argument("--experiment", type=Path, required=True, help="Path to experiment directory.")
    parser.add_argument("--trainer-path", type=str, default="jepa_world_model_trainer.py", help="Trainer path relative to repo root.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Override git repo root.")
    args = parser.parse_args()
    cache_path = generate_model_diff(args.experiment, trainer_path=args.trainer_path, repo_root=args.repo_root)
    print(f"Wrote model diff to {cache_path}")


if __name__ == "__main__":
    main()
