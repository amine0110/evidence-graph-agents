// Project page: every number is read from json/*.json, built by build_site_data.py.
(() => {
  "use strict";

  const KINDS = {
    caption: "Video description",
    detection: "Object detection",
    sound_event: "Sound event",
    transcript_segment: "Speech transcript",
    speech_tag: "Speech tag",
  };
  const EDGES = {
    same_event: "same event",
    supports: "supports",
    contradicts: "contradicts",
    derived_from: "derived from",
  };
  const SYSTEMS = [
    ["s1_omni", "S1", "one audio-visual model on the raw clip"],
    ["s1_ts", "S1-ts", "same, with time-labelled 20 s chunks"],
    ["s2", "S2", "agents exchange messages"],
    ["s3", "S3", "graph agent with tools"],
    ["s4", "S4", "graph as flat text, one pass"],
  ];

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const f3 = (x) => x.toFixed(3);
  const signed = (x) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(3);
  const el = (tag, attrs = {}, parent) => {
    const ns = ["svg", "g", "path", "circle", "line", "text", "rect", "marker", "defs", "title"].includes(tag);
    const e = ns ? document.createElementNS("http://www.w3.org/2000/svg", tag) : document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "text") e.textContent = v;
      else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v);
    }
    if (parent) parent.appendChild(e);
    return e;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  Promise.all([fetch("json/results.json").then((r) => r.json()), fetch("json/example_graph.json").then((r) => r.json())])
    .then(([res, ex]) => {
      fill(res, ex);
      stats(res);
      graph(ex);
      chart(res);
      robustness(res);
      findings(res);
    })
    .catch((err) => {
      console.error(err);
      document.getElementById("stats").textContent = "The data files could not be loaded.";
    });

  // --- small text fills -------------------------------------------------------
  function fill(res, ex) {
    document.querySelectorAll('[data-k="duration"]').forEach((e) => (e.textContent = Math.round(ex.duration)));
    document.querySelectorAll('[data-k="clips"]').forEach((e) => (e.textContent = res.clips));
  }

  function sys(res, key) {
    return res.systems.find((s) => s.key === key);
  }

  function stats(res) {
    const s1 = sys(res, "s1"), s4 = sys(res, "s4");
    const items = [
      [res.clips, "test clips, each reviewed by every system"],
      [res.graph.nodes.toLocaleString("en"), "observations written by five perception agents"],
      [f3(s4.f1) + " / " + f3(s1.f1), "F1 of flat text (S4) and of the single model (S1)"],
      [signed(res.sound_gain.f1_diff), "F1 from adding sound events to S4"],
      ["×" + Math.round(res.cost_ratio), "GPU time of the agent pipeline vs. the single model"],
    ];
    const box = document.getElementById("stats");
    for (const [v, t] of items) el("div", { class: "stat", html: `<b>${esc(v)}</b><span>${esc(t)}</span>` }, box);
  }

  // --- evidence graph ---------------------------------------------------------
  function graph(ex) {
    const svg = document.getElementById("graph-svg");
    const panel = document.getElementById("node-panel");
    const W = 1000, H = 520;
    const nodes = ex.nodes.map((n) => ({ ...n, mid: (n.t[0] + n.t[1]) / 2, r: 14 + 8 * n.confidence }));
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));

    // same_event is symmetric and stored both ways: keep one copy
    const seen = new Set();
    const edges = ex.edges.filter((e) => {
      if (e.type !== "same_event") return true;
      const k = [e.src, e.dst].sort().join("|");
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    }).map((e) => ({ ...e, s: byId[e.src], d: byId[e.dst] }));
    const nb = Object.fromEntries(nodes.map((n) => [n.id, new Set([n.id])]));
    edges.forEach((e) => { nb[e.src].add(e.dst); nb[e.dst].add(e.src); });

    // layouts
    const lanes = Object.keys(KINDS);
    const tx = (t, left = 140, right = 40) => left + (t / ex.duration) * (W - left - right);
    const laneY = (k) => 70 + lanes.indexOf(k) * ((H - 140) / (lanes.length - 1));
    const timeline = {};
    for (const k of lanes) {
      const rows = [];
      nodes.filter((n) => n.kind === k).sort((a, b) => a.mid - b.mid).forEach((n) => {
        let row = rows.findIndex((last) => tx(n.mid) - last > 2 * n.r + 6);
        if (row < 0) { row = rows.length; rows.push(0); }
        rows[row] = tx(n.mid);
        timeline[n.id] = { x: tx(n.mid), y: laneY(k) + (row % 2 ? 1 : -1) * Math.ceil(row / 2) * 22 * (row ? 1 : 0) };
      });
    }
    const network = force(nodes, edges, W, H, tx);

    // defs: one arrow per edge type
    const defs = el("defs", {}, svg);
    for (const t of Object.keys(EDGES)) {
      const m = el("marker", { id: `ar-${t}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
      el("path", { d: "M0,0 L10,5 L0,10 z", fill: css(`--e-${t}`) }, m);
    }

    const gAxis = el("g", {}, svg);
    const gSpan = el("g", {}, svg);
    const gEdges = el("g", {}, svg);
    const gNodes = el("g", {}, svg);

    const hidden = new Set();
    const ePaths = edges.map((e) => {
      const attrs = {
        class: "g-edge",
        stroke: css(`--e-${e.type}`),
        "stroke-width": 1.2 + 2.2 * (e.score ?? 0.5),
      };
      if (e.type === "contradicts") attrs["stroke-dasharray"] = "6 4";
      if (e.type === "derived_from") attrs["stroke-dasharray"] = "2 4";
      if (e.type !== "same_event") attrs["marker-end"] = `url(#ar-${e.type})`;
      const p = el("path", attrs, gEdges);
      el("title", { text: `${e.src} ${EDGES[e.type]} ${e.dst}${e.score != null ? ` (score ${e.score.toFixed(2)})` : ""}` }, p);
      return p;
    });
    const spans = nodes.map((n) => el("rect", { class: "g-span", rx: 4, height: 8, fill: css(`--m-${n.kind}`) }, gSpan));
    const gs = nodes.map((n) => {
      const g = el("g", { class: "g-node", tabindex: 0, role: "button", "aria-label": `${n.id}: ${KINDS[n.kind]}` }, gNodes);
      el("circle", { r: n.r, fill: css(`--m-${n.kind}`) }, g);
      el("text", { text: n.id.slice(1) }, g);
      g.addEventListener("mouseenter", () => focus(n));
      g.addEventListener("focus", () => focus(n));
      g.addEventListener("click", () => { pinned = pinned === n ? null : n; focus(n); });
      g.addEventListener("mouseleave", () => focus(pinned));
      return g;
    });
    let pinned = null, cited = new Set(), pos = network, layout = "network";

    function draw(P) {
      nodes.forEach((n, i) => gs[i].setAttribute("transform", `translate(${P[n.id].x.toFixed(1)},${P[n.id].y.toFixed(1)})`));
      edges.forEach((e, i) => {
        const a = P[e.src], b = P[e.dst];
        const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1;
        const bend = e.type === "same_event" ? 0 : e.type === "supports" ? 0.18 : -0.18;
        const mx = (a.x + b.x) / 2 - dy * bend, my = (a.y + b.y) / 2 + dx * bend;
        // end the arrow at the target's rim
        const ex2 = b.x - ((b.x - mx) / Math.hypot(b.x - mx, b.y - my || 1)) * (e.d.r + 3);
        const ey2 = b.y - ((b.y - my) / Math.hypot(b.x - mx, b.y - my || 1)) * (e.d.r + 3);
        ePaths[i].setAttribute("d", L < 1 ? "" : `M${a.x.toFixed(1)},${a.y.toFixed(1)} Q${mx.toFixed(1)},${my.toFixed(1)} ${ex2.toFixed(1)},${ey2.toFixed(1)}`);
      });
    }

    function axis(show) {
      gAxis.replaceChildren();
      gSpan.style.display = show ? "" : "none";
      if (!show) return;
      for (const k of lanes) el("text", { class: "g-lane", x: 12, y: laneY(k) + 4, text: KINDS[k] }, gAxis);
      el("line", { class: "g-axis", x1: tx(0), x2: tx(ex.duration), y1: H - 30, y2: H - 30 }, gAxis);
      const step = ex.duration > 30 ? 10 : 2;
      for (let t = 0; t <= ex.duration + 1e-6; t += step) {
        el("line", { class: "g-axis", x1: tx(t), x2: tx(t), y1: 40, y2: H - 30 }, gAxis).style.opacity = 0.5;
        el("text", { class: "g-tick", x: tx(t), y: H - 12, "text-anchor": "middle", text: `${t}s` }, gAxis);
      }
      nodes.forEach((n, i) => {
        spans[i].setAttribute("x", tx(n.t[0]));
        spans[i].setAttribute("width", Math.max(4, tx(n.t[1]) - tx(n.t[0])));
        spans[i].setAttribute("y", timeline[n.id].y - 4);
      });
    }

    function animate(to) {
      const from = pos, t0 = performance.now(), dur = matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : 650;
      const ease = (u) => 1 - Math.pow(1 - u, 3);
      function frame(now) {
        const u = dur ? Math.min(1, (now - t0) / dur) : 1, k = ease(u);
        const P = {};
        for (const n of nodes) P[n.id] = { x: from[n.id].x + (to[n.id].x - from[n.id].x) * k, y: from[n.id].y + (to[n.id].y - from[n.id].y) * k };
        draw(P);
        if (u < 1) requestAnimationFrame(frame);
        else pos = to;
      }
      requestAnimationFrame(frame);
    }

    function focus(n) {
      const keep = n ? nb[n.id] : null;
      nodes.forEach((m, i) => {
        gs[i].classList.toggle("g-dim", !!keep && !keep.has(m.id));
        gs[i].classList.toggle("cited", cited.has(m.id));
      });
      edges.forEach((e, i) => {
        const off = hidden.has(e.type) || (keep && !(e.src === n.id || e.dst === n.id));
        ePaths[i].classList.toggle("g-dim", !!off);
        ePaths[i].style.display = hidden.has(e.type) ? "none" : "";
      });
      if (!n) {
        panel.innerHTML = '<p class="hint">Hover or tap a node to see what the agent recorded. Tap again to release it.</p>';
        return;
      }
      const text = n.text.replace(/^\S+\s/, "");
      const out = edges.filter((e) => e.src === n.id || e.dst === n.id);
      const links = out.map((e) => e.type === "same_event"
        ? `same event as ${e.src === n.id ? e.dst : e.src}`
        : e.src === n.id ? `${EDGES[e.type]} ${e.dst}` : `${e.src} ${EDGES[e.type]} this`);
      panel.innerHTML = `
        <p class="tag"><i style="background:${css(`--m-${n.kind}`)}"></i>${esc(n.id)} · ${esc(KINDS[n.kind])}</p>
        <p class="time">${n.t[0].toFixed(1)}–${n.t[1].toFixed(1)} s</p>
        <blockquote>${esc(text)}</blockquote>
        <dl><dt>confidence</dt><dd>${n.confidence.toFixed(2)}</dd><dt>agent</dt><dd>${esc(n.producer)}</dd></dl>
        ${links.length ? `<p class="links"><b>Edges:</b> ${esc([...new Set(links)].join("; "))}</p>` : ""}`;
    }

    // controls
    document.querySelectorAll("[data-layout]").forEach((b) => b.addEventListener("click", () => {
      layout = b.dataset.layout;
      document.querySelectorAll("[data-layout]").forEach((x) => {
        x.classList.toggle("on", x === b);
        x.setAttribute("aria-selected", x === b);
      });
      axis(layout === "timeline");
      animate(layout === "timeline" ? timeline : network);
    }));
    const legend = document.getElementById("edge-legend");
    const counts = {};
    edges.forEach((e) => (counts[e.type] = (counts[e.type] || 0) + 1));
    for (const [t, name] of Object.entries(EDGES)) {
      if (!counts[t]) continue;
      const b = el("button", { type: "button", "aria-pressed": "true" }, legend);
      const dash = t === "contradicts" ? 'stroke-dasharray="5 3"' : t === "derived_from" ? 'stroke-dasharray="2 3"' : "";
      b.innerHTML = `<svg viewBox="0 0 24 8"><line x1="0" y1="4" x2="24" y2="4" stroke="${css(`--e-${t}`)}" stroke-width="2.5" ${dash}/></svg>${name} <span class="muted">${counts[t]}</span>`;
      b.addEventListener("click", () => {
        hidden.has(t) ? hidden.delete(t) : hidden.add(t);
        b.classList.toggle("off", hidden.has(t));
        b.setAttribute("aria-pressed", !hidden.has(t));
        focus(pinned);
      });
    }
    const nl = document.getElementById("node-legend");
    for (const [k, name] of Object.entries(KINDS)) {
      if (nodes.some((n) => n.kind === k)) el("span", { html: `<i style="background:${css(`--m-${k}`)}"></i>${name}` }, nl);
    }
    el("span", { class: "muted", text: "Size = confidence" }, nl);

    // answers
    const box = document.getElementById("answers");
    const cards = [];
    for (const [key, name, desc] of SYSTEMS) {
      const s = ex.systems[key];
      if (!s) continue;
      const lines = s.answer.split("\n");
      const yes = /^yes/i.test(lines[0]);
      const cites = (lines[2] || "").match(/n\d+/g) || [];
      const b = el("button", { class: "answer", type: "button" }, box);
      b.innerHTML = `
        <div class="name">${name}</div><div class="desc">${esc(desc)}</div>
        <div class="verdict"><b class="${yes ? "yes" : "no"}">${yes ? "Yes" : "No"}</b><span class="muted">score ${s.score.toFixed(2)}</span></div>
        <div class="bar"><i style="width:${(s.score * 100).toFixed(0)}%;background:${yes ? css("--good") : css("--bad")}"></i></div>
        <div class="cites">${cites.length ? `cites ${cites.join(", ")}` : key.startsWith("s1") ? "reads the raw clip, no nodes" : "cites no nodes"}</div>`;
      b.addEventListener("click", () => {
        const on = !b.classList.contains("on");
        cards.forEach((c) => c.classList.remove("on"));
        b.classList.toggle("on", on);
        cited = new Set(on ? cites : []);
        focus(pinned);
      });
      cards.push(b);
    }
    const tr = ex.systems.s3 && ex.systems.s3.trace;
    if (tr && tr.length) {
      const steps = tr.map((t) => t.tool === "window" ? `opened the ${t.kind === "caption" ? "video descriptions" : (t.kind || "node") + "s"} of ${t.start}–${t.end} s`
        : t.tool === "search" ? `searched for “${t.concept}”`
        : t.tool === "done" ? "stopped" : `called ${t.tool}`);
      document.getElementById("trace").innerHTML =
        `S3’s tool calls on this clip: ${esc(steps.join(", then "))}. It answered No; every other system answered Yes. The paper analyses where the graph agent goes wrong.`;
    }

    axis(false);
    draw(network);
    focus(null);
  }

  // deterministic force layout, pulled left-to-right by time
  function force(nodes, edges, W, H, tx) {
    let seed = 7;
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
    const P = Object.fromEntries(nodes.map((n) => [n.id, { x: tx(n.mid, 80, 80), y: H / 2 + (rnd() - 0.5) * 200, vx: 0, vy: 0 }]));
    for (let it = 0; it < 500; it++) {
      const a = 1 - it / 500;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const p = P[nodes[i].id], q = P[nodes[j].id];
          let dx = q.x - p.x, dy = q.y - p.y, d2 = dx * dx + dy * dy + 0.01;
          const f = (42000 / d2) * a, d = Math.sqrt(d2);
          dx /= d; dy /= d;
          p.vx -= dx * f; p.vy -= dy * f; q.vx += dx * f; q.vy += dy * f;
        }
      }
      for (const e of edges) {
        const p = P[e.src], q = P[e.dst];
        const dx = q.x - p.x, dy = q.y - p.y, d = Math.hypot(dx, dy) || 1;
        const f = (d - 170) * 0.008 * a;
        p.vx += (dx / d) * f; p.vy += (dy / d) * f; q.vx -= (dx / d) * f; q.vy -= (dy / d) * f;
      }
      for (const n of nodes) {
        const p = P[n.id];
        p.vx += (tx(n.mid, 80, 80) - p.x) * 0.02 * a;
        p.vy += (H / 2 - p.y) * 0.004 * a;
        p.x += Math.max(-20, Math.min(20, p.vx)); p.y += Math.max(-20, Math.min(20, p.vy));
        p.vx *= 0.6; p.vy *= 0.6;
        p.x = Math.max(n.r + 10, Math.min(W - n.r - 10, p.x));
        p.y = Math.max(n.r + 10, Math.min(H - n.r - 10, p.y));
      }
    }
    return Object.fromEntries(Object.entries(P).map(([k, p]) => [k, { x: p.x, y: p.y }]));
  }

  // --- F1 chart ---------------------------------------------------------------
  function chart(res) {
    const svg = document.getElementById("f1-chart");
    const rows = res.systems;
    const left = 300, right = 820, top = 30, rowH = 44;
    const lo = Math.floor(Math.min(...rows.map((r) => r.f1_ci[0])) * 100) / 100;
    const hi = Math.ceil(Math.max(...rows.map((r) => r.f1_ci[1])) * 100) / 100;
    svg.setAttribute("viewBox", `0 0 900 ${top + rows.length * rowH + 40}`);
    const x = (v) => left + ((v - lo) / (hi - lo)) * (right - left);
    const bottom = top + rows.length * rowH;
    for (let v = lo; v <= hi + 1e-9; v += 0.01) {
      el("line", { class: "c-grid", x1: x(v), x2: x(v), y1: top - 10, y2: bottom }, svg);
      el("text", { class: "c-tick", x: x(v), y: bottom + 18, "text-anchor": "middle", text: v.toFixed(2) }, svg);
    }
    el("text", { class: "c-tick", x: right, y: bottom + 34, "text-anchor": "end", text: "F1 (axis does not start at 0)" }, svg);
    const s1 = rows.find((r) => r.key === "s1");
    el("line", { x1: x(s1.f1), x2: x(s1.f1), y1: top - 10, y2: bottom, stroke: css("--m-speech_tag"), "stroke-dasharray": "4 4", "stroke-width": 1.5 }, svg);
    const color = { s1: css("--m-speech_tag"), s1_ts: css("--m-speech_tag"), s2: css("--m-sound_event"), s3: css("--m-caption"), s3v2: css("--m-caption"), s4: css("--good") };
    rows.forEach((r, i) => {
      const y = top + i * rowH + rowH / 2;
      const [name, ...rest] = r.label.split(" ");
      el("text", { class: "c-label", x: 16, y: y - 2, text: name }, svg);
      el("text", { class: "c-sub", x: 16, y: y + 14, text: rest.join(" ") }, svg);
      el("line", { x1: x(r.f1_ci[0]), x2: x(r.f1_ci[1]), y1: y, y2: y, stroke: color[r.key], "stroke-width": 3, "stroke-linecap": "round", opacity: 0.45 }, svg);
      el("circle", { cx: x(r.f1), cy: y, r: 7, fill: color[r.key] }, svg);
      el("text", { class: "c-val", x: x(r.f1_ci[1]) + 10, y: y + 4, text: f3(r.f1) }, svg);
    });
  }

  // --- robustness table -------------------------------------------------------
  function robustness(res) {
    const t = document.getElementById("robust");
    const cell = (v, base) => {
      if (v == null) return "<td>–</td>";
      if (base == null) return `<td>${f3(v)}</td>`;
      const d = v - base;
      return `<td>${f3(v)}<span class="d ${d >= 0 ? "up" : "down"}">${signed(d)}</span></td>`;
    };
    t.innerHTML = `<thead><tr><th>System (F1)</th><th>Gemma 4 12B</th><th>Gemma, reworded prompts</th><th>Qwen3.6-35B-A3B</th></tr></thead><tbody>${
      res.robustness.map((r) => `<tr><td>${esc(r.system)}</td>${cell(r.gemma)}${cell(r.paraphrased, r.gemma)}${cell(r.qwen, r.gemma)}</tr>`).join("")
    }</tbody>`;
  }

  // --- findings ---------------------------------------------------------------
  function findings(res) {
    const s = (k) => f3(sys(res, k).f1);
    const rob = Object.fromEntries(res.robustness.map((r) => [r.system, r]));
    const fa = res.faithfulness;
    const pct = (x) => `${Math.round(x * 100)}%`;
    const sg = res.sound_gain;
    const items = [
      `<b>Flat text is enough.</b> Reading every observation as text in one pass (S4, F1 ${s("s4")}) matches the single audio-visual model (S1, ${s("s1")}).`,
      `<b>Structure did not add accuracy.</b> The graph agent (S3, ${s("s3")}) and message passing (S2, ${s("s2")}) are less accurate. A graph-agent variant developed on a separate set (${s("s3v2")}) did not close the gap; graph summaries and multi-turn tool use both lowered accuracy.`,
      `<b>The comparison is robust.</b> With reworded prompts and with a larger reasoning model, S4 stays ahead of S3. The larger model weakens message passing (S2 ${f3(rob["S2"].gemma)} → ${f3(rob["S2"].qwen)}), enough for the graph agent to beat it.`,
      `<b>Sound matters.</b> Sound events improve F1 by ${signed(sg.f1_diff)} [${signed(sg.f1_diff_ci[0])}, ${signed(sg.f1_diff_ci[1])}] on clips with sound, in line with the annotators, who used sound or speech on most violent clips.`,
      `<b>Citations are faithful, but not more so with a graph.</b> Removing the nodes a system cites changes ${pct(fa.s3.cited)} of the graph agent’s decisions (${pct(fa.s3.random)} for random nodes), and ${pct(fa.s4.cited)} for flat text.`,
      `<b>Agents cost more.</b> The agent pipeline uses about ${Math.round(res.cost_ratio)} times the GPU time of the single model.`,
      `<b>Describers have artefacts.</b> We report two artefacts of vision-language describers that mislead pipelines built on them.`,
    ];
    const ol = document.getElementById("findings");
    for (const h of items) el("li", { html: h }, ol);
  }

  // --- copy BibTeX -----------------------------------------------------------
  const copy = document.getElementById("copy-bib");
  copy.addEventListener("click", () => {
    navigator.clipboard.writeText(document.getElementById("bibtex").textContent).then(() => {
      copy.textContent = "Copied";
      setTimeout(() => (copy.textContent = "Copy"), 1500);
    });
  });
})();
