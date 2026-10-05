import ast
import os

import networkx as nx
import matplotlib.pyplot as plt

from git import Repo
from pydriller import Repository
from matplotlib.lines import Line2D

import hashlib
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import Optional

CHILD = "child"
NEXT_SIB = "next_sib"
NEXT_USE = "next_use"
CLASS_METHOD = "class_method"
CALL = "call"
DATA_DEP = "data_dep"

SCOPE_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

NODE_COLORS = {
    "FunctionDef": "#6fa8dc",
    "AsyncFunctionDef": "#6fa8dc",
    "ClassDef": "#93c47d",
    "Import": "#f6b26b",
    "ImportFrom": "#f6b26b",
    "alias": "#f6b26b",
    "Assign": "#e06666",
    "AugAssign": "#e06666",
    "Name": "#ffd966",
    "Call": "#c27ba0",
}

DEFAULT_COLOR = "#cccccc"

EDGE_STYLES = {
    CHILD: dict(edge_color="#999999", style="solid", width=1.0),
    NEXT_SIB: dict(edge_color="#cccccc", style="dotted", width=0.6),
    NEXT_USE: dict(edge_color="#3d85c6", style="dashed", width=1.2),
    CLASS_METHOD: dict(edge_color="#38761d", style="solid", width=1.8),
    CALL: dict(edge_color="#a64d79", style="solid", width=1.8),
    DATA_DEP: dict(edge_color="#cc0000", style="dashed", width=1.4),
}

def node_token(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.arg):
        return n.arg
    if isinstance(n, ast.Attribute):
        return n.attr
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return n.name
    if isinstance(n, ast.alias):
        return n.asname or n.name
    if isinstance(n, ast.ImportFrom):
        return n.module or ""
    if isinstance(n, ast.Constant):
        v = n.value
        if isinstance(v, str):
            return "STR:" + v[:30]
        if isinstance(v, (int, float, complex)) and not isinstance(v, bool):
            return "NUM:" + repr(v)
        return repr(v)
    return ""

def sanitize_label(text):
    if not text:
        return text
    return text.replace("$", r"\$").replace("\n", "\\n")

def iter_ast_children(node):
    for field, value in ast.iter_fields(node):
        if isinstance(value, ast.AST):
            yield field, value
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, ast.AST):
                    yield field, v

class Graph:
    def __init__(self, path):
        self.path = path
        self.G = nx.MultiDiGraph()

    def add_node(self, node_path, node_type, token, lineno):
        self.G.add_node(node_path, type=node_type, token=token, lineno=lineno)

    def add_edge(self, source_path, target_path, edge_type):
        if source_path in self.G and target_path in self.G:
            self.G.add_edge(source_path, target_path, edge_type=edge_type)

    def edge_counts(self):
        counts = {}
        for _, _, d in self.G.edges(data=True):
            counts[d["edge_type"]] = counts.get(d["edge_type"], 0) + 1
        return counts

    def visualize(self, figsize=(12, 8), layout="dot", edge_types=None):
        if self.G.number_of_nodes() == 0:
            print(f"Graph kosong untuk {self.path}")
            return

        node_colors = [
            NODE_COLORS.get(self.G.nodes[n]["type"], DEFAULT_COLOR)
            for n in self.G.nodes
        ]
        labels = {
            n: sanitize_label(self.G.nodes[n]["token"] or self.G.nodes[n]["type"])
            for n in self.G.nodes
        }

        pos = self._layout(layout)

        plt.figure(figsize=figsize)
        plt.title(f"Graph for {self.path}")

        active_types = edge_types or list(EDGE_STYLES.keys())
        for etype in active_types:
            style = EDGE_STYLES[etype]
            edges = [
                (u, v) for u, v, d in self.G.edges(data=True)
                if d.get("edge_type") == etype
            ]
            if edges:
                show_arrows = (etype != NEXT_SIB)
                extra = dict(arrowsize=8, connectionstyle="arc3,rad=0.05") if show_arrows else {}
                nx.draw_networkx_edges(
                    self.G, pos, edgelist=edges,
                    edge_color=style["edge_color"],
                    style=style["style"],
                    width=style["width"],
                    arrows=show_arrows,
                    **extra,
                )

        nx.draw_networkx_nodes(
            self.G, pos, node_color=node_colors, node_size=300,
            linewidths=0.5, edgecolors="black",
        )
        nx.draw_networkx_labels(self.G, pos, labels, font_size=6)

        legend_handles = [
            Line2D([0], [0], marker="o", color="w", label=t,
                   markerfacecolor=c, markersize=8)
            for t, c in NODE_COLORS.items()
        ] + [
            Line2D([0], [0], color=EDGE_STYLES[e]["edge_color"], lw=2,
                   linestyle=EDGE_STYLES[e]["style"], label=e)
            for e in active_types
        ]
        plt.legend(handles=legend_handles, loc="upper left",
                   bbox_to_anchor=(1.02, 1), fontsize=7)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def _layout(self, layout):
        if layout == "dot":
            try:
                return nx.nx_agraph.graphviz_layout(self.G, prog="dot")
            except Exception:
                try:
                    return nx.nx_pydot.graphviz_layout(self.G, prog="dot")
                except Exception:
                    pass
        return nx.spring_layout(self.G, k=0.6, iterations=100, seed=42)

class Builder:
    def __init__(self, folder_path, project_name, commit_hash):
        self.folder_path = folder_path
        self.project_name = project_name
        self.commit_hash = commit_hash

        self.repo_path = os.path.join(folder_path, project_name)
        self.repo = Repo(self.repo_path)
        self.commit = self.repo.commit(commit_hash)

        self.sources = None
        self.graphs = {}
        self.failed_files = {}

    @classmethod
    def from_sources(cls, commit_hash, sources):
        """Builder tanpa Repo lokal. sources: {path: source_code} (mis. dari load_commit)."""
        self = cls.__new__(cls)
        self.folder_path = self.project_name = self.repo_path = None
        self.repo = self.commit = None
        self.commit_hash = commit_hash
        self.sources = sources
        self.graphs = {}
        self.failed_files = {}
        return self

    def build(self, paths=None):
        """paths: kalau diisi, hanya file dengan path itu yang diparse."""
        if self.sources is not None:
            for path, source_code in self.sources.items():
                if paths is None or path in paths:
                    self._build_one(path, lambda s=source_code: s)
            return self.graphs

        for item in self.commit.tree.traverse():
            if item.type != "blob" or not item.path.endswith(".py"):
                continue
            if paths is not None and item.path not in paths:
                continue
            self._build_one(item.path, lambda i=item: i.data_stream.read().decode("utf-8"))

        return self.graphs

    def _build_one(self, path, read_source):
        print(f"Processing file: {path}")

        try:
            source_code = read_source()
            tree = ast.parse(source_code)
        except (SyntaxError, UnicodeDecodeError, ValueError) as e:
            print(f"  Skip {path}: {type(e).__name__}: {e}")
            self.failed_files[path] = str(e)
            return

        self.graphs[path] = self.build_graph(tree, path)

    def build_graph(self, tree, path):
        graph = Graph(path)

        root_path = ("root",)
        graph.add_node(root_path, "Root", "", 0)

        last_sib = {}
        last_use = {}
        field_counter = {}
        id_to_path = {}

        def_registry = {}
        call_records = []

        stack = [(tree, root_path, "root", root_path, None, None)]

        while stack:
            ast_node, parent_path, field, scope_path, func_path, parent_type = stack.pop()

            fc_key = (parent_path, field)
            idx = field_counter.get(fc_key, 0)
            field_counter[fc_key] = idx + 1

            node_path = parent_path + (f"{field}[{idx}]:{type(ast_node).__name__}",)
            id_to_path[id(ast_node)] = node_path

            token = node_token(ast_node)
            lineno = getattr(ast_node, "lineno", 0) or 0
            node_type = type(ast_node).__name__

            graph.add_node(node_path, node_type, token, lineno)
            graph.add_edge(parent_path, node_path, CHILD)

            if fc_key in last_sib:
                graph.add_edge(last_sib[fc_key], node_path, NEXT_SIB)
            last_sib[fc_key] = node_path

            if isinstance(ast_node, (ast.Name, ast.arg, ast.alias)) and token:
                use_key = (scope_path, token)
                if use_key in last_use:
                    graph.add_edge(last_use[use_key], node_path, NEXT_USE)
                last_use[use_key] = node_path

            if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                def_registry.setdefault(ast_node.name, node_path)

            if (parent_type == "ClassDef" and field == "body"
                    and isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef))):
                graph.add_edge(parent_path, node_path, CLASS_METHOD)

            if isinstance(ast_node, ast.Call) and func_path is not None:
                callee = None
                if isinstance(ast_node.func, ast.Name):
                    callee = ast_node.func.id
                elif isinstance(ast_node.func, ast.Attribute):
                    callee = ast_node.func.attr
                if callee:
                    call_records.append((func_path, callee))

            child_scope = node_path if isinstance(ast_node, SCOPE_NODES) else scope_path
            child_func_path = (
                node_path if isinstance(ast_node, (ast.FunctionDef, ast.AsyncFunctionDef))
                else func_path
            )

            for f, c in reversed(list(iter_ast_children(ast_node))):
                stack.append((c, node_path, f, child_scope, child_func_path, node_type))

        for caller_path, callee_name in call_records:
            target_path = def_registry.get(callee_name)
            if target_path and target_path != caller_path:
                graph.add_edge(caller_path, target_path, CALL)

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                target_paths = [
                    id_to_path[id(t)] for t in node.targets
                    if isinstance(t, ast.Name) and id(t) in id_to_path
                ]
                source_paths = [
                    id_to_path[id(n)] for n in ast.walk(node.value)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                    and id(n) in id_to_path
                ]
                for sp in source_paths:
                    for tp in target_paths:
                        if sp != tp:
                            graph.add_edge(sp, tp, DATA_DEP)

            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and id(node.target) in id_to_path:
                    tp = id_to_path[id(node.target)]
                    graph.add_edge(tp, tp, DATA_DEP)
                    for n in ast.walk(node.value):
                        if (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                                and id(n) in id_to_path):
                            graph.add_edge(id_to_path[id(n)], tp, DATA_DEP)

        return graph


# ---------------------------------------------------------------------------
# Unified id: alignment top-down antara graph before dan current
#
# Prinsip: node dianggap "sama" hanya kalau ditemukan di bawah parent yang
# sudah cocok DAN di field yang sama. Hash isi (subtree_hash) hanya dipakai
# untuk membandingkan saudara-saudara di bawah parent yang sama, bukan untuk
# mencari "apakah hash ini ada di mana pun di file".
# ---------------------------------------------------------------------------

def _field_of(path):
    return path[-1].split("[")[0]


def compute_subtree_hashes(graph, root_path=("root",)):
    """Hash isi tiap subtree (tipe + token + hash anak, urut) dan daftar anak berurutan."""
    G = graph.G
    order = {n: i for i, n in enumerate(G.nodes)}   # urutan insert = urutan AST
    children = defaultdict(list)
    for u, v, d in G.edges(data=True):
        if d["edge_type"] == CHILD:
            children[u].append(v)
    for u in children:
        children[u].sort(key=order.get)

    hashes = {}
    stack = [(root_path, False)]
    while stack:
        n, visited = stack.pop()
        if not visited:
            stack.append((n, True))
            for c in children.get(n, []):
                stack.append((c, False))
            continue
        d = G.nodes[n]
        h = hashlib.sha1(f"{d['type']}|{d['token']}".encode("utf-8", "ignore"))
        for c in children.get(n, []):
            h.update(f"|{_field_of(c)}:{hashes[c]}".encode())
        hashes[n] = h.hexdigest()[:16]

    nx.set_node_attributes(G, hashes, "subtree_hash")
    graph.children = dict(children)
    return hashes


OPERATOR_TYPES = {
    "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow", "FloorDiv",
    "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
    "And", "Or", "Not", "Invert", "UAdd", "USub",
    "Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn",
}
CTX_TYPES = {"Load", "Store", "Del"}
SCOPE_TYPE_NAMES = {"FunctionDef", "AsyncFunctionDef", "ClassDef", "Module"}


def _kind(t):
    """Semua operator dianggap satu jenis, supaya Add <-> Sub bisa dipasangkan."""
    return "operator" if t in OPERATOR_TYPES else t


def _preorder(children, n):
    out, stack = [], [n]
    while stack:
        x = stack.pop()
        out.append(x)
        stack.extend(reversed(children.get(x, [])))
    return out


def _similarity(gb, gc, nb, nc):
    """Skor kemiripan 2 node (untuk memasangkan node yang berubah). -1 = tidak boleh dipasangkan."""
    db, dc = gb.G.nodes[nb], gc.G.nodes[nc]
    if _kind(db["type"]) != _kind(dc["type"]):
        return -1.0
    if db["subtree_hash"] == dc["subtree_hash"]:
        return 2.0
    tb, tc = db["token"] or "", dc["token"] or ""
    tok = SequenceMatcher(None, tb, tc).ratio() if (tb or tc) else 1.0
    kb = {gb.G.nodes[c]["subtree_hash"] for c in gb.children.get(nb, [])}
    kc = {gc.G.nodes[c]["subtree_hash"] for c in gc.children.get(nc, [])}
    struct = len(kb & kc) / len(kb | kc) if (kb or kc) else 1.0
    return 0.5 * tok + 0.5 * struct


def _pair_block(gb, gc, blk_b, blk_c):
    """Pasangkan node di blok 'replace': tipe harus sama, skor tertinggi dulu, seri -> posisi terdekat."""
    cands = []
    for i, nb in enumerate(blk_b):
        for j, nc in enumerate(blk_c):
            s = _similarity(gb, gc, nb, nc)
            if s >= 0:
                cands.append((-s, abs(i - j), i, j))
    cands.sort()
    used_b, used_c, pairs = set(), set(), []
    for _, _, i, j in cands:
        if i in used_b or j in used_c:
            continue
        used_b.add(i)
        used_c.add(j)
        pairs.append((blk_b[i], blk_c[j]))
    return pairs


def detect_moves(gb, gc, status_b, status_c, pairs, min_size=3):
    """
    Pasangkan subtree `removed` di before dengan subtree `added` di current yang
    isinya identik (subtree_hash sama) -> status 'moved' (akar) dan 'unchanged'
    (isi di bawahnya). Dua aturan, urut dari yang paling ketat:
      1. anchor : parent tempat subtree itu lepas/menempel adalah pasangan yang
                  sama (swap argumen, bungkus/lepas UnaryOp) -> ukuran berapa pun
      2. scope  : fungsi/class pembungkusnya pasangan yang sama, dan subtree
                  cukup besar (>= min_size) supaya leaf kecil tidak salah cocok
    """
    par_b = {c: p for p, cs in gb.children.items() for c in cs}
    par_c = {c: p for p, cs in gc.children.items() for c in cs}

    def outer(par, status, n, st):          # ancestor-or-self teratas ber-status st
        while n in par and status.get(par[n]) == st:
            n = par[n]
        return n

    def scope(g, par, n):
        while n in par:
            n = par[n]
            if g.G.nodes[n]["type"] in SCOPE_TYPE_NAMES:
                return n
        return n

    def size(g, n):
        return len(_preorder(g.children, n))

    rem = [n for n, s in status_b.items()
           if s == "removed" and gb.G.nodes[n]["type"] not in CTX_TYPES]
    rem.sort(key=lambda n: -size(gb, n))     # subtree besar dulu
    add_by_hash = defaultdict(list)
    for n, s in status_c.items():
        if s == "added" and gc.G.nodes[n]["type"] not in CTX_TYPES:
            add_by_hash[gc.G.nodes[n]["subtree_hash"]].append(n)

    used_b, used_c = set(), set()
    for rule in ("anchor", "scope"):
        for x in rem:
            if x in used_b:
                continue
            for y in add_by_hash.get(gb.G.nodes[x]["subtree_hash"], []):
                if y in used_c:
                    continue
                if rule == "anchor":
                    ax = par_b.get(outer(par_b, status_b, x, "removed"))
                    ay = par_c.get(outer(par_c, status_c, y, "added"))
                    ok = ax is not None and pairs.get(ax) == ay
                else:
                    ok = (size(gb, x) >= min_size
                          and pairs.get(scope(gb, par_b, x)) == scope(gc, par_c, y))
                if not ok:
                    continue
                for a, b in zip(_preorder(gb.children, x), _preorder(gc.children, y)):
                    pairs[a] = b
                    status_b[a] = status_c[b] = "unchanged"
                    used_b.add(a)
                    used_c.add(b)
                status_b[x] = status_c[y] = "moved"
                break


def unify_graphs(gb, gc, root_path=("root",), uid_prefix=""):
    """
    Cocokkan node graph before (gb) dan current (gc), beri unified id (`uid`)
    dan `status` di kedua graph:
      unchanged | modified (tipe/token node itu sendiri berubah) |
      modified_context (node sama, tapi ada perubahan di bawahnya) | added | removed
    Return: (status_before, status_current, modified_pairs)
    """
    compute_subtree_hashes(gb)
    compute_subtree_hashes(gc)

    status_b, status_c, pairs = {}, {}, {}

    def mark(g, n, status_map, st):
        for x in _preorder(g.children, n):
            status_map[x] = st

    def pair_identical(nb, nc):
        for x, y in zip(_preorder(gb.children, nb), _preorder(gc.children, nc)):
            pairs[x] = y
            status_b[x] = "unchanged"
            status_c[y] = "unchanged"

    def by_field(g, p):
        groups = {}
        for c in g.children.get(p, []):
            groups.setdefault(_field_of(c), []).append(c)
        return groups

    pairs[root_path] = root_path
    same_root = gb.G.nodes[root_path]["subtree_hash"] == gc.G.nodes[root_path]["subtree_hash"]
    status_b[root_path] = status_c[root_path] = "unchanged" if same_root else "modified_context"

    work = [(root_path, root_path)]
    while work:
        pb, pc = work.pop()
        fb, fc = by_field(gb, pb), by_field(gc, pc)
        for field in list(fb) + [f for f in fc if f not in fb]:
            lb, lc = fb.get(field, []), fc.get(field, [])
            hb = [gb.G.nodes[c]["subtree_hash"] for c in lb]
            hc = [gc.G.nodes[c]["subtree_hash"] for c in lc]
            for tag, i1, i2, j1, j2 in SequenceMatcher(None, hb, hc, autojunk=False).get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        pair_identical(lb[i1 + k], lc[j1 + k])
                    continue
                blk_b, blk_c = lb[i1:i2], lc[j1:j2]
                matched = _pair_block(gb, gc, blk_b, blk_c)
                done_b = {b for b, _ in matched}
                done_c = {c for _, c in matched}
                for b in blk_b:
                    if b not in done_b:
                        mark(gb, b, status_b, "removed")
                for c in blk_c:
                    if c not in done_c:
                        mark(gc, c, status_c, "added")
                for b, c in matched:
                    db, dc = gb.G.nodes[b], gc.G.nodes[c]
                    if db["subtree_hash"] == dc["subtree_hash"]:   # dipindah dalam parent yg sama
                        pair_identical(b, c)
                        continue
                    pairs[b] = c
                    st = "modified_context" if (db["type"], db["token"]) == (dc["type"], dc["token"]) else "modified"
                    status_b[b] = status_c[c] = st
                    work.append((b, c))

    detect_moves(gb, gc, status_b, status_c, pairs)

    # unified id: node yang cocok berbagi uid yang sama di kedua graph
    rev = {c: b for b, c in pairs.items()}
    uid_b, uid_c, k = {}, {}, 0
    for n in gb.G.nodes:
        uid_b[n] = f"{uid_prefix}u{k}"
        if n in pairs:
            uid_c[pairs[n]] = f"{uid_prefix}u{k}"
        k += 1
    for n in gc.G.nodes:
        if n not in uid_c:
            uid_c[n] = f"{uid_prefix}u{k}"
            k += 1

    nx.set_node_attributes(gb.G, uid_b, "uid")
    nx.set_node_attributes(gc.G, uid_c, "uid")
    nx.set_node_attributes(gb.G, status_b, "status")
    nx.set_node_attributes(gc.G, status_c, "status")

    modified_pairs = [(b, c) for b, c in pairs.items() if status_b[b] == "modified"]
    return status_b, status_c, modified_pairs


# ---------------------------------------------------------------------------
# Level commit: banyak file, termasuk file yang hanya ada di satu sisi
# ---------------------------------------------------------------------------

@dataclass
class FileDiff:
    path_before: Optional[str]
    path_current: Optional[str]
    kind: str                      # modified | added | removed | renamed
    gb: Optional["Graph"] = None   # None kalau file baru
    gc: Optional["Graph"] = None   # None kalau file dihapus
    modified_pairs: list = field(default_factory=list)


def changed_py_files(before_commit, current_commit):
    """[(path_before | None, path_current | None)] untuk file .py yang berubah.
    Rename terdeteksi oleh git (a_path != b_path)."""
    out = []
    for d in before_commit.diff(current_commit):
        a = None if d.new_file else d.a_path
        b = None if d.deleted_file else d.b_path
        a = a if a and a.endswith(".py") else None   # bukan .py -> anggap tidak ada
        b = b if b and b.endswith(".py") else None
        if a or b:
            out.append((a, b))
    return out


def mark_one_sided(g, status, uid_prefix):
    """File yang hanya ada di satu sisi: semua node added/removed, tetap diberi uid."""
    compute_subtree_hashes(g)
    nx.set_node_attributes(g.G, {n: status for n in g.G.nodes}, "status")
    nx.set_node_attributes(
        g.G, {n: f"{uid_prefix}u{i}" for i, n in enumerate(g.G.nodes)}, "uid")


def unify_commit(builder_before, builder_current, files=None, pairs=None):
    """
    Bandingkan dua commit pada level file. Hanya file .py yang berubah yang diparse.
    files: opsional, batasi ke path tertentu (mis. kolom file_path dataset).
    pairs: opsional, [(path_before | None, path_current | None)]; default dari git diff.
    Return: (list[FileDiff], skipped: {path: alasan})
    """
    if pairs is None:
        pairs = changed_py_files(builder_before.commit, builder_current.commit)
    if files is not None:
        keep = set(files)
        pairs = [(a, b) for a, b in pairs if a in keep or b in keep]

    builder_before.build(paths={a for a, _ in pairs if a})
    builder_current.build(paths={b for _, b in pairs if b})

    results, skipped = [], {}
    for i, (a, b) in enumerate(pairs):
        pre = f"f{i}:"
        gb = builder_before.graphs.get(a) if a else None
        gc = builder_current.graphs.get(b) if b else None
        if (a and gb is None) or (b and gc is None):        # gagal parse (mis. syntax py2)
            why = (builder_before.failed_files.get(a) if a and gb is None
                   else builder_current.failed_files.get(b))
            skipped[b or a] = why or "tidak bisa diparse"
            continue

        if gb is not None and gc is not None:
            _, _, mp = unify_graphs(gb, gc, uid_prefix=pre)
            kind = "modified" if a == b else "renamed"
            results.append(FileDiff(a, b, kind, gb, gc, mp))
        elif gc is not None:
            mark_one_sided(gc, "added", pre)
            results.append(FileDiff(None, b, "added", None, gc))
        else:
            mark_one_sided(gb, "removed", pre)
            results.append(FileDiff(a, None, "removed", gb, None))
    return results, skipped


# ---------------------------------------------------------------------------
# PyDriller: ambil data commit langsung dari URL (clone otomatis ke cache_dir)
# ---------------------------------------------------------------------------

@dataclass
class CommitData:
    hash: str
    parent_hash: Optional[str]
    msg: str
    files: list   # [(path_before | None, path_current | None, source_before, source_current)]


def load_commit(repo_url, commit_hash, cache_dir="repository"):
    """Commit + file .py yang berubah via PyDriller. Repo di-clone sekali ke cache_dir
    lalu dipakai ulang di run berikutnya."""
    os.makedirs(cache_dir, exist_ok=True)
    commit = next(Repository(repo_url, single=commit_hash, clone_repo_to=cache_dir).traverse_commits())

    files = []
    for m in commit.modified_files:
        # PyDriller memakai separator OS (\ di Windows); samakan dengan path git (/)
        a = m.old_path.replace(os.sep, "/") if m.old_path else None
        b = m.new_path.replace(os.sep, "/") if m.new_path else None
        a = a if a and a.endswith(".py") else None   # bukan .py -> anggap tidak ada
        b = b if b and b.endswith(".py") else None
        if a or b:
            files.append((a, b, m.source_code_before if a else None, m.source_code if b else None))

    parent = commit.parents[0] if commit.parents else None
    if parent and not files and commit.merge:
        print(f"Warning: {commit.hash} adalah merge commit; PyDriller tidak memberi daftar file yang berubah.")
    return CommitData(commit.hash, parent, commit.msg, files)


def unify_commit_from_url(repo_url, commit_hash, files=None, cache_dir="repository"):
    """Seperti unify_commit, tapi cukup URL + hash.
    Return: (builder_before, builder_current, list[FileDiff], skipped)"""
    data = load_commit(repo_url, commit_hash, cache_dir)
    builder_before = Builder.from_sources(data.parent_hash, {a: sb for a, _, sb, _ in data.files if a})
    builder_current = Builder.from_sources(data.hash, {b: sc for _, b, _, sc in data.files if b})
    pairs = [(a, b) for a, b, _, _ in data.files]
    file_diffs, skipped = unify_commit(builder_before, builder_current, files=files, pairs=pairs)
    return builder_before, builder_current, file_diffs, skipped



# ---------------------------------------------------------------------------
# Union graph (X): satu node per uid, fitur before/after digabung
# ---------------------------------------------------------------------------

def _token_sim(tb, tc):
    """1.0 kalau token sama; rasio kemiripan kalau berubah; 0.0 kalau node hanya ada di satu sisi."""
    if tb is None or tc is None:
        return 0.0
    return 1.0 if tb == tc else SequenceMatcher(None, tb, tc).ratio()


def _edge_counter(g):
    """Counter[(uid_src, uid_dst, edge_type)] -> jumlah edge paralel."""
    if g is None:
        return Counter()
    return Counter((g.G.nodes[u]["uid"], g.G.nodes[v]["uid"], d["edge_type"])
                   for u, v, d in g.G.edges(data=True))


def build_union_graph(file_diffs):
    """
    FileDiff hasil unify_commit -> satu nx.MultiDiGraph (X).

    Node (kunci = uid):
      status, changed, in_before, in_after,
      type_before/type_after, token_before/token_after, token_sim,
      lineno_before/lineno_after, file
    Edge (kunci = edge_type; satu edge per (uid_src, uid_dst, edge_type)):
      edge_type, edge_status (kept|removed|added), in_before, in_after, n_before, n_after

    Tidak ada root virtual: tiap file jadi komponen sendiri (atribut `file`).
    Edge balik child->parent sengaja TIDAK ditambah di sini; itu urusan tahap tensor.
    """
    U = nx.MultiDiGraph()
    for fd in file_diffs:
        gb, gc = fd.gb, fd.gc
        file = fd.path_current or fd.path_before
        nb = {d["uid"]: d for _, d in gb.G.nodes(data=True)} if gb else {}
        nc = {d["uid"]: d for _, d in gc.G.nodes(data=True)} if gc else {}

        for uid in list(nb) + [u for u in nc if u not in nb]:   # urutan deterministik
            b, c = nb.get(uid), nc.get(uid)
            status = (c or b)["status"]
            tb = b["token"] if b else None
            tc = c["token"] if c else None
            U.add_node(
                uid, file=file, status=status, changed=(status != "unchanged"),
                in_before=b is not None, in_after=c is not None,
                type_before=b["type"] if b else None, type_after=c["type"] if c else None,
                token_before=tb, token_after=tc, token_sim=_token_sim(tb, tc),
                lineno_before=b["lineno"] if b else None,
                lineno_after=c["lineno"] if c else None,
            )

        eb, ec = _edge_counter(gb), _edge_counter(gc)
        for key in list(eb) + [k for k in ec if k not in eb]:
            u, v, et = key
            n_b, n_c = eb.get(key, 0), ec.get(key, 0)
            U.add_edge(
                u, v, key=et, edge_type=et,
                edge_status="kept" if (n_b and n_c) else ("removed" if n_b else "added"),
                in_before=n_b > 0, in_after=n_c > 0, n_before=n_b, n_after=n_c,
            )
    return U


def summarize_union(U):
    """Ringkasan indikator perubahan (untuk pengecekan cepat di notebook)."""
    nodes = Counter(d["status"] for _, d in U.nodes(data=True))
    edges = Counter((d["edge_type"], d["edge_status"])
                    for _, _, d in U.edges(data=True) if d["edge_status"] != "kept")
    return {"nodes": dict(nodes), "edges_changed": dict(edges),
            "n_nodes": U.number_of_nodes(), "n_edges": U.number_of_edges()}