import ast
from typing import Any, Dict, List
import networkx as nx

TRACKED_ALLOCATORS = {"open", "socket", "connect", "malloc"}
TRACKED_DEALLOCATORS = {"close", "disconnect", "free"}


class LeakDetector:

    def __init__(self, cfg: nx.DiGraph):
        self.cfg = cfg

    def detect_leaks(self) -> List[Dict[str, Any]]:
        leaks = []
        entry_nodes = [
            n
            for n, d in self.cfg.in_degree()
            if d == 0 and "Entry" in self.cfg.nodes[n]["label"]
        ]
        exit_nodes = [
            n for n in self.cfg.nodes if "Exit" in self.cfg.nodes[n]["label"]
        ]

        if not entry_nodes or not exit_nodes:
            return leaks

        entry = entry_nodes[0]
        exit_target = exit_nodes[0]

        # Find all reachable paths from Entry to Exit
        all_paths = list(nx.all_simple_paths(self.cfg, entry, exit_target))

        for path in all_paths:
            # Map tracking variable -> allocation metadata
            active_handles: Dict[str, Dict[str, Any]] = {}

            for node_id in path:
                block = self.cfg.nodes[node_id]["block"]
                for stmt in block.statements:
                    # Check allocation: var = open(...)
                    if isinstance(stmt, ast.Assign):
                        for target in stmt.targets:
                            if (
                                isinstance(target, ast.Name)
                                and isinstance(stmt.value, ast.Call)
                                and isinstance(stmt.value.func, ast.Name)
                            ):
                                if stmt.value.func.id in TRACKED_ALLOCATORS:
                                    active_handles[target.id] = {
                                        "var": target.id,
                                        "allocator": stmt.value.func.id,
                                        "line": stmt.lineno,
                                        "path": path,
                                    }

                    # Check deallocation: var.close() or free(var)
                    elif isinstance(stmt, ast.Expr) and isinstance(
                        stmt.value, ast.Call
                    ):
                        # Method call: handle.close()
                        if (
                            isinstance(stmt.value.func, ast.Attribute)
                            and stmt.value.func.attr in TRACKED_DEALLOCATORS
                        ):
                            caller = stmt.value.func.value
                            if (
                                isinstance(caller, ast.Name)
                                and caller.id in active_handles
                            ):
                                del active_handles[caller.id]

            # Path reached exit: any handle still open is a verified leak
            for var_name, meta in active_handles.items():
                leaks.append(
                    {
                        "variable": var_name,
                        "resource_type": meta["allocator"],
                        "allocated_line": meta["line"],
                        "execution_path": meta["path"],
                        "message": f"Resource handle '{var_name}' allocated at line {meta['line']} is never closed on execution path {meta['path']}.",
                    }
                )

        # De-duplicate leaks across multiple paths
        unique_leaks = {
            (l["variable"], l["allocated_line"]): l for l in leaks
        }.values()
        return list(unique_leaks)