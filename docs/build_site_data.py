"""Build the data files of the project page from the frozen results.

Every number on the page comes from experiments/c3_agents/frozen/; the example
graph is exported from the evidence graph of the worked example in the paper.

    python docs/build_site_data.py --graphs ~/data/XD-Violence/graphs
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C3 = ROOT / "experiments" / "c3_agents"
FROZEN = C3 / "frozen"
sys.path.insert(0, str(C3))
sys.path.insert(0, str(ROOT / "src"))

SYSTEMS = {
    "s1": "S1 audio-visual model on the raw clip",
    "s1_ts": "S1-ts same, with time-labelled chunks",
    "s2": "S2 agents exchange messages",
    "s3": "S3 graph agent with tools (v1)",
    "s3v2": "S3 graph agent with tools (v2)",
    "s4": "S4 graph as flat text, one pass",
}


def load(name):
    return json.loads((FROZEN / name).read_text())


def results():
    ev, rob, cost, stats = load("eval_final.json"), load("eval_robustness.json"), load("cost.json"), load("graph_stats.json")
    rows = []
    for key, label in SYSTEMS.items():
        a = ev["runs"][key]["all"]
        rows.append({"key": key, "label": label, "f1": a["f1"], "f1_ci": a["f1_ci"], "auc": a["auc"],
                     "grounding_precision": a["grounding_precision"]})
    robust = []
    for key, gemma, para, qwen in (("S2", "s2", None, "s2_qwen36"), ("S3 v1", "s3", "s3_p2", "s3_qwen36"),
                                   ("S4", "s4", "s4_p2", "s4_qwen36")):
        r = rob["runs"]
        robust.append({"system": key, "gemma": r[gemma]["all"]["f1"],
                       "paraphrased": r[para]["all"]["f1"] if para else None, "qwen": r[qwen]["all"]["f1"]})
    return {"systems": rows, "robustness": robust, "clips": ev["runs"]["s1"]["all"]["clips"],
            "faithfulness": {k: {"cited": v["flip_rate_cited"], "random": v["flip_rate_random"]}
                             for k, v in ev["faithfulness"].items()},
            "graph": {"nodes": stats["nodes_total"], "edges": sum(stats["edges"].values()),
                      "median_nodes": stats["nodes_median_per_clip"], "node_kinds": stats["nodes"],
                      "edge_types": stats["edges"]},
            "sound_gain": {k: ev["pairs"]["s4 - s4_no_audio"]["with_sound"][k] for k in ("f1_diff", "f1_diff_ci")},
            "cost_ratio": cost["s4_total"] / cost["s1"]["s"]}


def example(graphs):
    import reasoning
    from audio_events import clip_graph

    ex = load("example.json")
    view = reasoning.View(clip_graph(graphs, ex["clip_id"]))
    nodes = [{"id": view.short[n.id], "kind": n.kind, "modality": n.modality, "t": list(n.t_obs),
              "confidence": n.confidence, "producer": n.source.producer, "text": view.line(n).split("] ", 1)[1]}
             for n in view.nodes]
    edges = [{"src": view.short[e.src], "dst": view.short[e.dst], "type": e.type.value, "score": e.score}
             for es in view.graph.out_edges.values() for e in es if e.src in view.short and e.dst in view.short]
    systems = {k: {"score": v["score"], "answer": v["answer"], "trace": v.get("trace", [])}
               for k, v in ex["systems"].items()}
    return {"duration": ex["duration"], "nodes": nodes, "edges": edges, "systems": systems}


KIND_LANES = [("caption", "Video description", "#3b82f6"), ("detection", "Object detection", "#8b5cf6"),
              ("sound_event", "Sound event", "#f59e0b"), ("transcript_segment", "Speech transcript", "#14b8a6"),
              ("speech_tag", "Speech tag", "#e11d48")]
EDGE_STYLE = {"same_event": ("#94a3b8", ""), "supports": ("#16a34a", ""),
              "contradicts": ("#dc2626", ' stroke-dasharray="6 4"'), "derived_from": ("#64748b", ' stroke-dasharray="2 4"')}


def timeline_svg(g):
    """Static timeline drawing of the example graph, for the README."""
    W, H, left, right, top, gap = 1000, 430, 150, 30, 50, 66
    lane = {k: top + i * gap for i, (k, _, _) in enumerate(KIND_LANES)}
    color = {k: c for k, _, c in KIND_LANES}
    x = lambda t: left + t / g["duration"] * (W - left - right)
    pos, last = {}, {}
    for n in sorted(g["nodes"], key=lambda n: sum(n["t"])):
        k, cx = n["kind"], x(sum(n["t"]) / 2)
        row = 0
        while cx - last.get((k, row), -1e9) < 46:
            row += 1
        last[(k, row)] = cx
        pos[n["id"]] = (cx, lane[k] + (0, 20, -20, 40)[row % 4])
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
           'font-family="Inter, -apple-system, Segoe UI, Helvetica, Arial, sans-serif">',
           f'<rect width="{W}" height="{H}" rx="18" fill="#f8fafc"/>', "<defs>"]
    for t, (c, _) in EDGE_STYLE.items():
        out.append(f'<marker id="a-{t}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
                   f'orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
    out.append("</defs>")
    for k, name, c in KIND_LANES:
        out.append(f'<text x="18" y="{lane[k] + 4}" font-size="13" font-weight="600" fill="#475569">{name}</text>')
        out.append(f'<line x1="{left}" x2="{W - right}" y1="{lane[k]}" y2="{lane[k]}" stroke="#e2e8f0"/>')
    axis = top + 4 * gap + 46
    for t in range(0, int(g["duration"]) + 1, 2):
        out.append(f'<text x="{x(t):.1f}" y="{axis + 16}" font-size="11" fill="#64748b" text-anchor="middle">{t}s</text>')
    out.append(f'<line x1="{left}" x2="{W - right}" y1="{axis}" y2="{axis}" stroke="#cbd5e1"/>')
    for n in g["nodes"]:
        cy = pos[n["id"]][1]
        out.append(f'<rect x="{x(n["t"][0]):.1f}" y="{cy - 4}" width="{max(4, x(n["t"][1]) - x(n["t"][0])):.1f}" '
                   f'height="8" rx="4" fill="{color[n["kind"]]}" opacity="0.2"/>')
    seen = set()
    for e in g["edges"]:
        if e["type"] == "same_event":
            key = tuple(sorted((e["src"], e["dst"])))
            if key in seen:
                continue
            seen.add(key)
        (x1, y1), (x2, y2) = pos[e["src"]], pos[e["dst"]]
        bend = {"same_event": 0.0, "supports": 0.15}.get(e["type"], -0.2)
        mx, my = (x1 + x2) / 2 - (y2 - y1) * bend, (y1 + y2) / 2 + (x2 - x1) * bend
        c, dash = EDGE_STYLE[e["type"]]
        w = 1 + 2 * (e["score"] if e["score"] is not None else 0.5)
        arrow = "" if e["type"] == "same_event" else f' marker-end="url(#a-{e["type"]})"'
        r = 19
        d = ((x2 - mx) ** 2 + (y2 - my) ** 2) ** 0.5 or 1
        ex, ey = x2 - (x2 - mx) / d * r, y2 - (y2 - my) / d * r
        out.append(f'<path d="M{x1:.1f},{y1:.1f} Q{mx:.1f},{my:.1f} {ex:.1f},{ey:.1f}" fill="none" stroke="{c}" '
                   f'stroke-width="{w:.1f}" opacity="0.75"{dash}{arrow}/>')
    for n in g["nodes"]:
        cx, cy = pos[n["id"]]
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{12 + 4 * n["confidence"]:.1f}" fill="{color[n["kind"]]}" '
                   'stroke="#fff" stroke-width="2.5"/>')
        out.append(f'<text x="{cx:.1f}" y="{cy + 4:.1f}" font-size="11" font-weight="700" fill="#fff" '
                   f'text-anchor="middle">{n["id"][1:]}</text>')
    lx = left
    for t, label in (("same_event", "same event"), ("supports", "supports"), ("derived_from", "derived from")):
        c, dash = EDGE_STYLE[t]
        out.append(f'<line x1="{lx}" x2="{lx + 26}" y1="{H - 18}" y2="{H - 18}" stroke="{c}" stroke-width="2.5"{dash}/>')
        out.append(f'<text x="{lx + 32}" y="{H - 14}" font-size="12" fill="#334155">{label}</text>')
        lx += 130
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", required=True, help="folder with the per-agent graph files")
    a = p.parse_args(argv)
    out = ROOT / "docs" / "json"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(results(), indent=1) + "\n")
    g = example(a.graphs)
    (out / "example_graph.json").write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
    (ROOT / "docs" / "assets" / "example_graph.svg").write_text(timeline_svg(g))


if __name__ == "__main__":
    main()
