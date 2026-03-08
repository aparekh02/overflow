/**
 * AnalyticsPage — Scientific research dashboard.
 * Serif headings, light-background figure insets with proper axes,
 * figure captions, statistical notation, clean table design.
 */

import { useState, useMemo, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronDown, ChevronUp, ExternalLink } from "lucide-react";
import Badge from "../components/ui/Badge";
import { useStore } from "../store";
import { useSimManager, startSimLoop, stopSimLoop } from "../lib/simManager";
import type { CounterfactualRun } from "../lib/simTypes";
import type { OpenEnvOutput } from "../lib/openenvClient";
import { colors, fonts, radius } from "../theme";

/* ── fonts & palette ────────────────────────────────────────────── */

const FONT_HREF =
  "https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,300;8..60,400;8..60,600;8..60,700&display=swap";

const serif = "'Source Serif 4', Georgia, 'Times New Roman', serif";
const mono = fonts.mono;

// Scientific palette — no neon, no glows
const P = {
  blue: "#3b82f6",
  red: "#ef4444",
  green: "#22c55e",
  amber: "#f59e0b",
  cyan: "#06b6d4",
  purple: "#8b5cf6",
  slate: "#94a3b8",
  // figure inset
  paper: "#0e1118",
  paperBorder: "#1e2433",
  // text on dark bg
  t1: "#e2e8f0",
  t2: "#94a3b8",
  t3: "#64748b",
  t4: "#475569",
  rule: "#1e293b",
};

const ACTION_CLR: Record<string, string> = {
  keep_lane: P.slate, brake_mild: P.amber, brake_hard: P.red,
  accelerate: P.green, merge_left: P.blue, merge_right: P.cyan,
  yield: "#f97316", nudge_left: P.purple, nudge_right: "#7c3aed",
};

const ACTION_LBL: Record<string, string> = {
  keep_lane: "keep_lane", brake_mild: "brake_mild", brake_hard: "brake_hard",
  accelerate: "accelerate", merge_left: "merge_left", merge_right: "merge_right",
  yield: "yield", nudge_left: "nudge_left", nudge_right: "nudge_right",
};

type SortField = "label" | "reward" | "delta" | "ttc" | "status" | "steps";
type SortDir = "asc" | "desc";

/* ── helpers ─────────────────────────────────────────────────────── */

function mean(arr: number[]) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function stddev(arr: number[]) {
  if (arr.length < 2) return 0;
  const m = mean(arr);
  return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / (arr.length - 1));
}

function actionStats(runs: CounterfactualRun[]) {
  const m: Record<string, { n: number; r: number }> = {};
  let total = 0;
  for (const run of runs) for (const a of run.actionStream) {
    if (!m[a.action]) m[a.action] = { n: 0, r: 0 };
    m[a.action].n++;
    m[a.action].r += a.reward;
    total++;
  }
  return Object.entries(m)
    .map(([a, { n, r }]) => ({ action: a, count: n, pct: total ? n / total : 0, avg: n ? r / n : 0 }))
    .sort((a, b) => b.count - a.count);
}

function f(n: number, d = 3) { return n.toFixed(d); }

/* ── main component ──────────────────────────────────────────────── */

export default function AnalyticsPage() {
  const navigate = useNavigate();
  const scenario = useStore((s) => s.scenarioId);
  const runs = useSimManager((s) => s.runs);
  const mainState = useSimManager((s) => s.mainState);
  const totalSpawned = useSimManager((s) => s.totalSpawned);

  const [sortField, setSortField] = useState<SortField>("steps");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [statusFilter, setStatusFilter] = useState("all");
  const [ticketFilter, setTicketFilter] = useState("all");

  useEffect(() => {
    if (!document.querySelector(`link[href="${FONT_HREF}"]`)) {
      const l = document.createElement("link");
      l.rel = "stylesheet";
      l.href = FONT_HREF;
      document.head.appendChild(l);
    }
  }, []);

  useEffect(() => { startSimLoop(); return () => stopSimLoop(); }, []);

  /* statistics */
  const rewards = runs.map((r) => r.metrics.cumulativeReward);
  const deltas = runs.map((r) => r.metrics.deltaVsMain);
  const ttcs = runs.map((r) => r.metrics.minTTC).filter((v) => v < 100);
  const R_mean = mean(rewards);
  const R_std = stddev(rewards);
  const D_best = deltas.length ? Math.max(...deltas) : 0;
  const TTC_min = ttcs.length ? Math.min(...ttcs) : Infinity;
  const TTC_mean = ttcs.length ? mean(ttcs) : Infinity;
  const n_intv = runs.reduce((s, r) => s + r.metrics.interventionCount, 0);
  const n_active = runs.filter((r) => r.status === "running").length;
  const n_fin = runs.filter((r) => r.status === "finished").length;
  const actions = useMemo(() => actionStats(runs), [runs]);
  const ranked = useMemo(() => [...runs].sort((a, b) => b.metrics.deltaVsMain - a.metrics.deltaVsMain).slice(0, 6), [runs]);

  const filtered = useMemo(() => {
    let list = [...runs];
    if (statusFilter !== "all") list = list.filter((r) => r.status === statusFilter);
    list.sort((a, b) => {
      let c = 0;
      switch (sortField) {
        case "label": c = a.label.localeCompare(b.label); break;
        case "reward": c = a.metrics.cumulativeReward - b.metrics.cumulativeReward; break;
        case "delta": c = a.metrics.deltaVsMain - b.metrics.deltaVsMain; break;
        case "ttc": c = a.metrics.minTTC - b.metrics.minTTC; break;
        case "status": c = a.status.localeCompare(b.status); break;
        case "steps": c = a.actionStream.length - b.actionStream.length; break;
      }
      return sortDir === "desc" ? -c : c;
    });
    return list;
  }, [runs, sortField, sortDir, statusFilter]);

  const toggleSort = (fld: SortField) => {
    if (sortField === fld) setSortDir((d) => d === "asc" ? "desc" : "asc");
    else { setSortField(fld); setSortDir("desc"); }
  };

  const tickets = useMemo(() => genTickets(runs), [runs]);
  const filteredTickets = ticketFilter === "all" ? tickets : tickets.filter((t) => t.severity === ticketFilter);

  return (
    <div style={{ padding: "32px 40px 64px", minHeight: "100%", maxWidth: 1200, margin: "0 auto" }}>

      {/* ── Title ─────────────────────────────────────── */}
      <h1 style={{ fontFamily: serif, fontSize: 26, fontWeight: 700, color: P.t1, margin: 0, letterSpacing: "-0.02em" }}>
        Counterfactual Simulation Analysis
      </h1>
      <p style={{ fontFamily: serif, fontSize: 13, color: P.t3, margin: "6px 0 0", lineHeight: 1.5 }}>
        Scenario: <em>{scenario.replace(/_/g, " ")}</em>
        {" \u00a0|\u00a0 "}
        <em>N</em> = {runs.length} rollouts
        {totalSpawned > 0 && <>, {totalSpawned} total spawned</>}
        {" \u00a0|\u00a0 "}
        {n_active > 0 ? <span style={{ color: P.green }}>{n_active} active</span> : <span style={{ color: P.t4 }}>idle</span>}
        {", "}{n_fin} finished
      </p>
      <div style={{ height: 1, background: P.rule, margin: "20px 0 28px" }} />

      {/* ── Summary Statistics ─────────────────────────── */}
      <FigureLabel text="Summary Statistics" />
      <div style={{
        display: "grid", gridTemplateColumns: "repeat(5, 1fr)",
        border: `1px solid ${P.paperBorder}`, borderRadius: 4, overflow: "hidden",
        background: P.paper, marginBottom: 32,
      }}>
        {[
          { label: "Mean Reward", sym: "R\u0304", val: f(R_mean), sub: `\u03c3 = ${f(R_std)}`, color: R_mean > 0 ? P.green : P.red },
          { label: "Baseline", sym: "R\u2080", val: f(mainState.cumulativeReward), sub: "ground truth", color: P.t2 },
          { label: "Best \u0394", sym: "\u0394*", val: `${D_best >= 0 ? "+" : ""}${f(D_best)}`, sub: ranked[0]?.label || "\u2014", color: D_best > 0 ? P.green : P.t3 },
          { label: "Min TTC", sym: "TTC\u2098\u1d62\u2099", val: TTC_min < 100 ? `${TTC_min.toFixed(1)}s` : "\u2014", sub: TTC_mean < 100 ? `\u03bc = ${TTC_mean.toFixed(1)}s` : "all safe", color: TTC_min < 3 ? P.red : TTC_min < 5 ? P.amber : P.t2 },
          { label: "Interventions", sym: "n\u1d62\u2099\u209c", val: String(n_intv), sub: `${runs.filter((r) => r.metrics.minTTC < 2).length} critical`, color: n_intv > 0 ? P.amber : P.t3 },
        ].map((s, i) => (
          <div key={i} style={{
            padding: "16px 18px",
            borderLeft: i > 0 ? `1px solid ${P.paperBorder}` : undefined,
          }}>
            <div style={{ fontSize: 9, fontFamily: mono, color: P.t4, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 8 }}>
              {s.label}
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
              <span style={{ fontSize: 11, fontFamily: serif, fontStyle: "italic", color: P.t3 }}>{s.sym}</span>
              <span style={{ fontSize: 24, fontWeight: 700, fontFamily: mono, color: s.color, letterSpacing: "-0.02em" }}>
                {s.val}
              </span>
            </div>
            <div style={{ fontSize: 10, fontFamily: mono, color: P.t4, marginTop: 6 }}>
              {s.sub}
            </div>
          </div>
        ))}
      </div>

      {/* ── Fig. 1 — Reward Timeline ──────────────────── */}
      <FigureLabel text="Fig. 1" />
      <div style={{
        background: P.paper, border: `1px solid ${P.paperBorder}`,
        borderRadius: 4, padding: "20px 20px 12px", marginBottom: 6,
      }}>
        <RewardChart runs={runs} mainReward={mainState.cumulativeReward} />
      </div>
      <Caption>
        Cumulative reward <Em>R</Em>(<Em>t</Em>) for each counterfactual rollout over time.
        Dashed line indicates ground-truth baseline <Em>R</Em><sub>0</sub> = {f(mainState.cumulativeReward)}.
        Points colored by performance relative to baseline: <span style={{ color: P.green }}>above</span>, <span style={{ color: P.red }}>below</span>.
      </Caption>

      <div style={{ height: 32 }} />

      {/* ── Fig. 2 + Fig. 3 side by side ──────────────── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, marginBottom: 6 }}>
        <div>
          <FigureLabel text="Fig. 2" />
          <div style={{ background: P.paper, border: `1px solid ${P.paperBorder}`, borderRadius: 4, padding: 16 }}>
            <ActionTable stats={actions} />
          </div>
          <Caption>
            Action policy distribution across all rollouts. <Em>n</Em> = frequency, <Em>R\u0304</Em> = mean reward per action.
          </Caption>
        </div>
        <div>
          <FigureLabel text="Fig. 3" />
          <div style={{ background: P.paper, border: `1px solid ${P.paperBorder}`, borderRadius: 4, padding: 16 }}>
            <RankingTable ranked={ranked} />
          </div>
          <Caption>
            Top counterfactual runs ranked by improvement <Em>\u0394</Em> over ground-truth baseline.
          </Caption>
        </div>
      </div>

      <div style={{ height: 32 }} />

      {/* ── Table 1 — Run Details ─────────────────────── */}
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10 }}>
        <FigureLabel text="Table 1 — Counterfactual Run Details" />
        <div style={{ display: "flex", gap: 6 }}>
          <Pill label="All" on={statusFilter === "all"} onClick={() => setStatusFilter("all")} />
          <Pill label="Running" on={statusFilter === "running"} onClick={() => setStatusFilter("running")} />
          <Pill label="Finished" on={statusFilter === "finished"} onClick={() => setStatusFilter("finished")} />
        </div>
      </div>
      <div style={{ background: P.paper, border: `1px solid ${P.paperBorder}`, borderRadius: 4, overflow: "hidden", marginBottom: 6 }}>
        <div style={{ overflow: "auto", maxHeight: 440 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${P.rule}` }}>
                <SortTH label="Run" field="label" cur={sortField} dir={sortDir} onSort={toggleSort} />
                <SortTH label="Status" field="status" cur={sortField} dir={sortDir} onSort={toggleSort} />
                <SortTH label="R" field="reward" cur={sortField} dir={sortDir} onSort={toggleSort} />
                <SortTH label={"\u0394"} field="delta" cur={sortField} dir={sortDir} onSort={toggleSort} />
                <SortTH label="TTC\u2098\u1d62\u2099" field="ttc" cur={sortField} dir={sortDir} onSort={toggleSort} />
                <SortTH label="Steps" field="steps" cur={sortField} dir={sortDir} onSort={toggleSort} />
                <th style={thS}>Action Sequence</th>
                <th style={thS} />
              </tr>
            </thead>
            <tbody>
              {filtered.map((run, idx) => (
                <tr key={run.id}
                  style={{ borderBottom: `1px solid ${P.paperBorder}`, background: idx % 2 === 0 ? "transparent" : "rgba(255,255,255,0.008)" }}
                >
                  <td style={tdS}>
                    <span style={{ fontWeight: 600, fontFamily: mono, color: P.t1, fontSize: 12 }}>{run.label}</span>
                    <div style={{ fontSize: 9, fontFamily: mono, color: P.t4, marginTop: 1 }}>{run.branchId.slice(0, 10)}</div>
                  </td>
                  <td style={tdS}>
                    <Badge variant={run.status === "running" ? "success" : run.status === "finished" ? "info" : "default"} dot>
                      {run.status}
                    </Badge>
                  </td>
                  <td style={tdS}>
                    <span style={{ fontFamily: mono, fontSize: 12, color: run.metrics.cumulativeReward > 0 ? P.green : P.red }}>
                      {f(run.metrics.cumulativeReward)}
                    </span>
                  </td>
                  <td style={tdS}>
                    <span style={{
                      fontFamily: mono, fontSize: 11, padding: "1px 5px", borderRadius: 3,
                      color: run.metrics.deltaVsMain > 0.01 ? P.green : run.metrics.deltaVsMain < -0.01 ? P.red : P.t3,
                      background: run.metrics.deltaVsMain > 0.01 ? "rgba(34,197,94,0.06)" : run.metrics.deltaVsMain < -0.01 ? "rgba(239,68,68,0.06)" : "transparent",
                    }}>
                      {run.metrics.deltaVsMain >= 0 ? "+" : ""}{f(run.metrics.deltaVsMain)}
                    </span>
                  </td>
                  <td style={tdS}>
                    <span style={{ fontFamily: mono, fontSize: 11, color: run.metrics.minTTC < 2 ? P.red : run.metrics.minTTC < 5 ? P.amber : P.t3 }}>
                      {run.metrics.minTTC < 100 ? `${run.metrics.minTTC.toFixed(1)}s` : "\u2014"}
                    </span>
                  </td>
                  <td style={tdS}>
                    <span style={{ fontFamily: mono, fontSize: 11, color: P.t3 }}>{run.actionStream.length}</span>
                  </td>
                  <td style={tdS}><ActionSeq actions={run.actionStream} /></td>
                  <td style={{ ...tdS, textAlign: "right" }}>
                    <button onClick={() => navigate("/dashboard")} style={{
                      padding: "3px 8px", borderRadius: 3, fontSize: 10, fontFamily: mono,
                      background: "transparent", border: `1px solid ${P.paperBorder}`,
                      color: P.t3, cursor: "pointer", display: "inline-flex", alignItems: "center", gap: 4,
                    }}>
                      view <ExternalLink size={9} />
                    </button>
                  </td>
                </tr>
              ))}
              {filtered.length === 0 && (
                <tr><td colSpan={8} style={{ ...tdS, textAlign: "center", color: P.t4, padding: 32, fontFamily: serif, fontStyle: "italic" }}>
                  {runs.length === 0 ? "Awaiting counterfactual rollout data\u2026" : "No runs match the selected filter."}
                </td></tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
      <Caption>
        Complete listing of counterfactual runs. <Em>R</Em> = cumulative reward, <Em>\u0394</Em> = deviation from baseline,
        TTC<sub>min</sub> = minimum time-to-collision. Action sequences show the last 20 decisions color-coded by type.
      </Caption>

      <div style={{ height: 32 }} />

      {/* ── Observations ──────────────────────────────── */}
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 10 }}>
        <FigureLabel text="Observations" />
        <div style={{ display: "flex", gap: 6 }}>
          <Pill label="All" on={ticketFilter === "all"} onClick={() => setTicketFilter("all")} />
          <Pill label="Critical" on={ticketFilter === "critical"} onClick={() => setTicketFilter("critical")} />
          <Pill label="Warning" on={ticketFilter === "warning"} onClick={() => setTicketFilter("warning")} />
        </div>
      </div>
      <div style={{ background: P.paper, border: `1px solid ${P.paperBorder}`, borderRadius: 4, overflow: "hidden" }}>
        {filteredTickets.length === 0 ? (
          <div style={{ padding: 28, textAlign: "center", fontFamily: serif, fontStyle: "italic", color: P.t4, fontSize: 12 }}>
            {tickets.length === 0 ? "No anomalies detected." : "No observations match the selected filter."}
          </div>
        ) : (
          <div style={{ maxHeight: 280, overflow: "auto" }}>
            {filteredTickets.map((t, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "baseline", gap: 12,
                padding: "9px 16px",
                borderBottom: i < filteredTickets.length - 1 ? `1px solid ${P.paperBorder}` : undefined,
                background: i % 2 === 0 ? "transparent" : "rgba(255,255,255,0.006)",
              }}>
                <span style={{
                  fontSize: 9, fontFamily: mono, color: P.t4, width: 40, flexShrink: 0,
                }}>{t.time}</span>
                <span style={{
                  width: 6, height: 6, borderRadius: "50%", flexShrink: 0,
                  background: t.severity === "critical" ? P.red : t.severity === "warning" ? P.amber : P.blue,
                  marginTop: 4,
                }} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <span style={{ fontSize: 12, color: P.t1, fontFamily: mono }}>{t.title}</span>
                  <span style={{ fontSize: 11, color: P.t4, marginLeft: 8 }}>{t.description}</span>
                </div>
                <span style={{
                  fontSize: 9, fontFamily: mono, textTransform: "uppercase", letterSpacing: "0.06em",
                  color: t.severity === "critical" ? P.red : t.severity === "warning" ? P.amber : P.t4,
                }}>
                  {t.severity}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ================================================================ */
/*  Sub-components                                                   */
/* ================================================================ */

function FigureLabel({ text }: { text: string }) {
  return (
    <div style={{
      fontFamily: serif, fontSize: 13, fontWeight: 600, color: P.t2,
      marginBottom: 8, letterSpacing: "-0.01em",
    }}>
      {text}
    </div>
  );
}

function Caption({ children }: { children: React.ReactNode }) {
  return (
    <p style={{
      fontFamily: serif, fontSize: 11, color: P.t4, lineHeight: 1.6,
      margin: "8px 0 0", maxWidth: 720, fontStyle: "italic",
    }}>
      {children}
    </p>
  );
}

function Em({ children }: { children: React.ReactNode }) {
  return <span style={{ fontFamily: serif, fontStyle: "italic", color: P.t3 }}>{children}</span>;
}

/* ── Reward chart ── */

function RewardChart({ runs, mainReward }: { runs: CounterfactualRun[]; mainReward: number }) {
  const W = 680, H = 200;
  const pad = { t: 20, r: 56, b: 36, l: 56 };
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

  const sorted = [...runs].sort((a, b) => a.createdAt - b.createdAt);
  if (!sorted.length) {
    return <div style={{ height: 140, display: "flex", alignItems: "center", justifyContent: "center", fontFamily: serif, fontStyle: "italic", color: P.t4, fontSize: 12 }}>Awaiting data\u2026</div>;
  }

  const t0 = sorted[0].createdAt, t1 = sorted[sorted.length - 1].createdAt;
  const tr = Math.max(t1 - t0, 1000);
  const rews = sorted.map((r) => r.metrics.cumulativeReward);
  const rMin = Math.min(...rews, mainReward) - 0.5;
  const rMax = Math.max(...rews, mainReward) + 0.5;
  const rr = Math.max(rMax - rMin, 0.1);
  const x = (t: number) => pad.l + ((t - t0) / tr) * pw;
  const y = (v: number) => pad.t + ph - ((v - rMin) / rr) * ph;

  const pts = sorted.map((r) => ({ px: x(r.createdAt), py: y(r.metrics.cumulativeReward), run: r }));
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p.px},${p.py}`).join(" ");
  const area = `${line} L${pts[pts.length - 1].px},${pad.t + ph} L${pts[0].px},${pad.t + ph} Z`;

  const nTicks = 5;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
      <defs>
        <linearGradient id="areaFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={P.blue} stopOpacity="0.08" />
          <stop offset="100%" stopColor={P.blue} stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* Y-axis grid + labels + tick marks */}
      {Array.from({ length: nTicks + 1 }, (_, i) => i / nTicks).map((frac) => {
        const yy = pad.t + frac * ph;
        const val = rMax - frac * rr;
        return (
          <g key={frac}>
            <line x1={pad.l} y1={yy} x2={W - pad.r} y2={yy} stroke={P.paperBorder} strokeWidth={.5} />
            <line x1={pad.l - 4} y1={yy} x2={pad.l} y2={yy} stroke={P.t4} strokeWidth={1} />
            <text x={pad.l - 8} y={yy + 3.5} fill={P.t4} fontSize={9} textAnchor="end" fontFamily={mono}>{val.toFixed(1)}</text>
          </g>
        );
      })}

      {/* Y-axis label */}
      <text x={14} y={pad.t + ph / 2} fill={P.t4} fontSize={10} fontFamily={serif} fontStyle="italic"
        textAnchor="middle" transform={`rotate(-90, 14, ${pad.t + ph / 2})`}>
        R (cumulative)
      </text>

      {/* X-axis line */}
      <line x1={pad.l} y1={pad.t + ph} x2={W - pad.r} y2={pad.t + ph} stroke={P.t4} strokeWidth={1} />
      {/* X-axis label */}
      <text x={pad.l + pw / 2} y={H - 4} fill={P.t4} fontSize={10} fontFamily={serif} fontStyle="italic" textAnchor="middle">
        t (elapsed)
      </text>
      {/* X-axis tick labels */}
      {sorted.length >= 2 && [0, 0.5, 1].map((frac) => {
        const idx = Math.min(Math.floor(frac * (sorted.length - 1)), sorted.length - 1);
        const r = sorted[idx];
        const xx = x(r.createdAt);
        return (
          <g key={frac}>
            <line x1={xx} y1={pad.t + ph} x2={xx} y2={pad.t + ph + 4} stroke={P.t4} strokeWidth={1} />
            <text x={xx} y={pad.t + ph + 14} fill={P.t4} fontSize={8} textAnchor="middle" fontFamily={mono}>
              {new Date(r.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
            </text>
          </g>
        );
      })}

      {/* baseline */}
      <line x1={pad.l} y1={y(mainReward)} x2={W - pad.r} y2={y(mainReward)}
        stroke={P.red} strokeWidth={1} strokeDasharray="6 4" opacity={.45} />

      {/* area + line */}
      <path d={area} fill="url(#areaFill)" />
      <path d={line} fill="none" stroke={P.blue} strokeWidth={1.5} />

      {/* data points */}
      {pts.map(({ px, py, run }) => {
        const c = run.metrics.deltaVsMain > 0 ? P.green : run.metrics.deltaVsMain < -.05 ? P.red : P.blue;
        return (
          <g key={run.id}>
            <circle cx={px} cy={py} r={3.5} fill={c} stroke={P.paper} strokeWidth={1.5} />
          </g>
        );
      })}

      {/* legend box */}
      <g transform={`translate(${W - pad.r - 120}, ${pad.t})`}>
        <rect x={0} y={0} width={115} height={42} rx={3} fill={P.paper} stroke={P.paperBorder} strokeWidth={1} />
        <line x1={8} y1={14} x2={22} y2={14} stroke={P.blue} strokeWidth={1.5} />
        <circle cx={15} cy={14} r={2.5} fill={P.blue} />
        <text x={28} y={17} fill={P.t3} fontSize={9} fontFamily={mono}>CF rollouts</text>
        <line x1={8} y1={30} x2={22} y2={30} stroke={P.red} strokeWidth={1} strokeDasharray="4 3" />
        <text x={28} y={33} fill={P.t3} fontSize={9} fontFamily={mono}>baseline R₀</text>
      </g>
    </svg>
  );
}

/* ── Action table (Fig. 2) ── */

function ActionTable({ stats }: { stats: ReturnType<typeof actionStats> }) {
  if (!stats.length) return <div style={{ padding: 24, textAlign: "center", fontFamily: serif, fontStyle: "italic", color: P.t4, fontSize: 12 }}>Collecting\u2026</div>;
  const mx = Math.max(...stats.map((s) => s.pct), .01);
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
      <thead>
        <tr style={{ borderBottom: `2px solid ${P.rule}` }}>
          <th style={{ ...thS2, textAlign: "left" }}>Action</th>
          <th style={thS2}>n</th>
          <th style={thS2}>%</th>
          <th style={{ ...thS2, textAlign: "left", paddingLeft: 8 }}>Distribution</th>
          <th style={thS2}>R\u0304</th>
        </tr>
      </thead>
      <tbody>
        {stats.map(({ action, count, pct, avg }, i) => (
          <tr key={action} style={{ borderBottom: `1px solid ${P.paperBorder}`, background: i % 2 ? "rgba(255,255,255,0.006)" : "transparent" }}>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 10, color: P.t2 }}>{ACTION_LBL[action]}</td>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 10, color: P.t3, textAlign: "center" }}>{count}</td>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 10, color: P.t3, textAlign: "center" }}>{(pct * 100).toFixed(0)}</td>
            <td style={{ padding: "6px 8px" }}>
              <div style={{ height: 10, background: "rgba(255,255,255,0.03)", borderRadius: 2, overflow: "hidden" }}>
                <div style={{
                  width: `${(pct / mx) * 100}%`, height: "100%", borderRadius: 2,
                  background: ACTION_CLR[action] || P.slate, opacity: 0.5,
                }} />
              </div>
            </td>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 10, textAlign: "center", color: avg > 0.5 ? P.green : avg > 0 ? P.t2 : P.red }}>
              {avg >= 0 ? "+" : ""}{avg.toFixed(2)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ── Ranking table (Fig. 3) ── */

function RankingTable({ ranked }: { ranked: CounterfactualRun[] }) {
  if (!ranked.length) return <div style={{ padding: 24, textAlign: "center", fontFamily: serif, fontStyle: "italic", color: P.t4, fontSize: 12 }}>Ranking\u2026</div>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
      <thead>
        <tr style={{ borderBottom: `2px solid ${P.rule}` }}>
          <th style={thS2}>#</th>
          <th style={{ ...thS2, textAlign: "left" }}>Run</th>
          <th style={thS2}>{"\u0394"}</th>
          <th style={thS2}>R</th>
          <th style={thS2}>TTC<sub style={{ fontSize: 7 }}>min</sub></th>
          <th style={thS2}>Steps</th>
        </tr>
      </thead>
      <tbody>
        {ranked.map((r, i) => (
          <tr key={r.id} style={{
            borderBottom: `1px solid ${P.paperBorder}`,
            background: i === 0 ? "rgba(34,197,94,0.03)" : i % 2 ? "rgba(255,255,255,0.006)" : "transparent",
          }}>
            <td style={{ padding: "6px 8px", fontFamily: serif, fontSize: 11, color: i === 0 ? P.green : P.t4, textAlign: "center", fontWeight: i === 0 ? 700 : 400 }}>
              {i + 1}
            </td>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 11, color: P.t1, fontWeight: 600 }}>{r.label}</td>
            <td style={{
              padding: "6px 8px", fontFamily: mono, fontSize: 11, textAlign: "center",
              color: r.metrics.deltaVsMain > 0 ? P.green : r.metrics.deltaVsMain < -0.01 ? P.red : P.t3,
            }}>
              {r.metrics.deltaVsMain >= 0 ? "+" : ""}{f(r.metrics.deltaVsMain)}
            </td>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 11, color: P.t2, textAlign: "center" }}>
              {f(r.metrics.cumulativeReward)}
            </td>
            <td style={{
              padding: "6px 8px", fontFamily: mono, fontSize: 11, textAlign: "center",
              color: r.metrics.minTTC < 2 ? P.red : r.metrics.minTTC < 5 ? P.amber : P.t3,
            }}>
              {r.metrics.minTTC < 100 ? `${r.metrics.minTTC.toFixed(1)}` : "\u2014"}
            </td>
            <td style={{ padding: "6px 8px", fontFamily: mono, fontSize: 11, color: P.t3, textAlign: "center" }}>
              {r.actionStream.length}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ── Tiny components ── */

function ActionSeq({ actions }: { actions: OpenEnvOutput[] }) {
  const d = actions.slice(-20);
  if (!d.length) return <span style={{ fontFamily: mono, fontSize: 9, color: P.t4 }}>\u2014</span>;
  return (
    <div style={{ display: "flex", gap: 1, alignItems: "center" }}>
      {d.map((a, i) => (
        <div key={i} style={{ width: 4, height: 10, borderRadius: 1, background: ACTION_CLR[a.action] || P.slate, opacity: .35 + (i / d.length) * .65 }} />
      ))}
    </div>
  );
}

function Pill({ label, on, onClick }: { label: string; on: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} style={{
      padding: "2px 8px", borderRadius: 3, fontSize: 10, fontFamily: mono,
      background: on ? "rgba(255,255,255,0.06)" : "transparent",
      color: on ? P.t1 : P.t4,
      border: `1px solid ${on ? P.paperBorder : "transparent"}`,
      cursor: "pointer",
    }}>
      {label}
    </button>
  );
}

function SortTH({ label, field, cur, dir, onSort }: { label: string; field: SortField; cur: SortField; dir: SortDir; onSort: (f: SortField) => void }) {
  const on = cur === field;
  return (
    <th onClick={() => onSort(field)} style={{ ...thS, cursor: "pointer", color: on ? P.t1 : P.t4 }}>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 3 }}>
        {label}
        {on && (dir === "desc" ? <ChevronDown size={10} /> : <ChevronUp size={10} />)}
      </div>
    </th>
  );
}

const thS: React.CSSProperties = {
  padding: "8px 12px", textAlign: "left", fontSize: 9, fontWeight: 500,
  color: P.t4, textTransform: "uppercase", letterSpacing: "0.06em",
  fontFamily: mono, whiteSpace: "nowrap",
};
const thS2: React.CSSProperties = {
  padding: "6px 8px", textAlign: "center", fontSize: 9, fontWeight: 500,
  color: P.t4, letterSpacing: "0.06em", fontFamily: mono, whiteSpace: "nowrap",
};
const tdS: React.CSSProperties = { padding: "8px 12px", verticalAlign: "middle" };

/* ── ticket gen ── */
interface Ticket { title: string; description: string; severity: "critical" | "warning" | "info"; time: string }
function genTickets(runs: CounterfactualRun[]): Ticket[] {
  const t: Ticket[] = [];
  const tf = (ts: number) => new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  for (const r of runs) {
    if (r.metrics.minTTC < 2) t.push({ title: `TTC\u2098\u1d62\u2099 < 2s`, description: `Run ${r.label} — ${r.metrics.minTTC.toFixed(1)}s`, severity: "critical", time: tf(r.createdAt) });
    if (r.metrics.interventionCount > 0) t.push({ title: `Hard brake`, description: `Run ${r.label} — ${r.metrics.interventionCount}\u00d7`, severity: "warning", time: tf(r.createdAt) });
    if (r.metrics.deltaVsMain > .5) t.push({ title: `\u0394 > 0.5`, description: `Run ${r.label} — +${r.metrics.deltaVsMain.toFixed(2)}`, severity: "info", time: tf(r.createdAt) });
    if (r.metrics.cumulativeReward < -.5) t.push({ title: `R < \u22120.5`, description: `Run ${r.label} — ${r.metrics.cumulativeReward.toFixed(2)}`, severity: "warning", time: tf(r.createdAt) });
  }
  return t.sort((a, b) => ({ critical: 0, warning: 1, info: 2 }[a.severity] - { critical: 0, warning: 1, info: 2 }[b.severity]));
}
