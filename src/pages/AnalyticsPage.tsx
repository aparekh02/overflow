/**
 * AnalyticsPage — Clean dashboard with overview cards, run table, timeline chart,
 * and incident ticket feed.
 */

import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Shield,
  TrendingUp,
  Award,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
} from "lucide-react";
import Badge from "../components/ui/Badge";
import { useSimManager } from "../lib/simManager";
import type { CounterfactualRun } from "../lib/simTypes";
import { colors, fonts, typeScale, spacing, radius } from "../theme";

type SortField = "label" | "reward" | "delta" | "ttc" | "status" | "created";
type SortDir = "asc" | "desc";

export default function AnalyticsPage() {
  const navigate = useNavigate();
  const runs = useSimManager((s) => s.runs);
  const mainState = useSimManager((s) => s.mainState);
  const [sortField, setSortField] = useState<SortField>("created");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [statusFilter, setStatusFilter] = useState<string>("all");
  const [ticketFilter, setTicketFilter] = useState<string>("all");

  // ── Summary stats ──
  const allRuns = runs;
  const activeRuns = allRuns.filter((r) => r.status === "running");
  const finishedRuns = allRuns.filter((r) => r.status === "finished");

  const nearMissesAvoided = allRuns.filter((r) =>
    r.metrics.minTTC > 3 && r.actionStream.some((a) => a.action.includes("brake") || a.action === "yield"),
  ).length;

  const avgReward = allRuns.length > 0
    ? allRuns.reduce((s, r) => s + r.metrics.avgReward, 0) / allRuns.length
    : 0;

  const bestDelta = allRuns.length > 0
    ? Math.max(...allRuns.map((r) => r.metrics.deltaVsMain))
    : 0;

  const totalInterventions = allRuns.reduce((s, r) => s + r.metrics.interventionCount, 0);

  // ── Sorted + filtered runs ──
  const filteredRuns = useMemo(() => {
    let result = [...allRuns];
    if (statusFilter !== "all") {
      result = result.filter((r) => r.status === statusFilter);
    }
    result.sort((a, b) => {
      let cmp = 0;
      switch (sortField) {
        case "label": cmp = a.label.localeCompare(b.label); break;
        case "reward": cmp = a.metrics.cumulativeReward - b.metrics.cumulativeReward; break;
        case "delta": cmp = a.metrics.deltaVsMain - b.metrics.deltaVsMain; break;
        case "ttc": cmp = a.metrics.minTTC - b.metrics.minTTC; break;
        case "status": cmp = a.status.localeCompare(b.status); break;
        case "created": cmp = a.createdAt - b.createdAt; break;
      }
      return sortDir === "desc" ? -cmp : cmp;
    });
    return result;
  }, [allRuns, sortField, sortDir, statusFilter]);

  const toggleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  // ── Incident tickets ──
  const tickets = useMemo(() => generateTickets(allRuns), [allRuns]);
  const filteredTickets = ticketFilter === "all" ? tickets : tickets.filter((t) => t.severity === ticketFilter);

  return (
    <div style={{
      padding: spacing.xl,
      height: "100%",
      overflow: "auto",
      display: "flex",
      flexDirection: "column",
      gap: spacing.xl,
    }}>
      {/* ── Header ── */}
      <div>
        <h1 style={{ ...typeScale.h1, color: colors.textPrimary, margin: 0 }}>Analytics</h1>
        <p style={{ ...typeScale.body, color: colors.textDim, marginTop: 4 }}>
          Aggregate metrics across all counterfactual rollouts
        </p>
      </div>

      {/* ── Overview cards ── */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: spacing.md,
      }}>
        <OverviewCard
          icon={Shield}
          label="Near Misses Avoided"
          value={nearMissesAvoided}
          subtext="Runs where ego action prevented TTC < 3s"
          color={colors.success}
        />
        <OverviewCard
          icon={TrendingUp}
          label="Average Reward"
          value={avgReward.toFixed(3)}
          subtext="Across all counterfactual runs"
          color={colors.info}
        />
        <OverviewCard
          icon={Award}
          label="Best CF Delta"
          value={`+${bestDelta.toFixed(3)}`}
          subtext="Best improvement over ground truth"
          color={colors.accent}
        />
        <OverviewCard
          icon={AlertTriangle}
          label="Total Interventions"
          value={totalInterventions}
          subtext="Hard brakes across all runs"
          color={colors.warning}
        />
      </div>

      {/* ── Timeline chart (lightweight SVG) ── */}
      <div style={{
        background: colors.bgCard,
        border: `1px solid ${colors.border}`,
        borderRadius: radius.lg,
        padding: spacing.lg,
      }}>
        <div style={{ ...typeScale.h3, color: colors.textPrimary, marginBottom: spacing.md }}>
          Reward Timeline
        </div>
        <RewardTimeline runs={allRuns} mainReward={mainState.cumulativeReward} />
      </div>

      {/* ── Runs table ── */}
      <div style={{
        background: colors.bgCard,
        border: `1px solid ${colors.border}`,
        borderRadius: radius.lg,
        overflow: "hidden",
      }}>
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: `${spacing.md}px ${spacing.lg}px`,
          borderBottom: `1px solid ${colors.border}`,
        }}>
          <div style={{ ...typeScale.h3, color: colors.textPrimary }}>
            Counterfactual Runs ({filteredRuns.length})
          </div>
          <div style={{ display: "flex", gap: spacing.sm }}>
            <FilterPill label="All" active={statusFilter === "all"} onClick={() => setStatusFilter("all")} />
            <FilterPill label="Running" active={statusFilter === "running"} onClick={() => setStatusFilter("running")} />
            <FilterPill label="Finished" active={statusFilter === "finished"} onClick={() => setStatusFilter("finished")} />
          </div>
        </div>

        {/* Table */}
        <div style={{ overflow: "auto", maxHeight: 400 }}>
          <table style={{
            width: "100%",
            borderCollapse: "collapse",
            fontFamily: fonts.sans,
            fontSize: 12,
          }}>
            <thead>
              <tr style={{ borderBottom: `1px solid ${colors.border}` }}>
                <SortHeader label="Run" field="label" current={sortField} dir={sortDir} onSort={toggleSort} />
                <SortHeader label="Status" field="status" current={sortField} dir={sortDir} onSort={toggleSort} />
                <SortHeader label="Reward" field="reward" current={sortField} dir={sortDir} onSort={toggleSort} />
                <SortHeader label="Delta" field="delta" current={sortField} dir={sortDir} onSort={toggleSort} />
                <SortHeader label="Min TTC" field="ttc" current={sortField} dir={sortDir} onSort={toggleSort} />
                <SortHeader label="Created" field="created" current={sortField} dir={sortDir} onSort={toggleSort} />
                <th style={thStyle}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredRuns.map((run) => (
                <tr
                  key={run.id}
                  style={{
                    borderBottom: `1px solid ${colors.borderSubtle}`,
                    transition: "background 0.1s",
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                >
                  <td style={tdStyle}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <span style={{ fontWeight: 500, color: colors.textPrimary }}>{run.label}</span>
                      <span style={{ ...typeScale.mono, color: colors.textDim, fontSize: 9 }}>
                        {run.branchId.slice(0, 8)}
                      </span>
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <Badge
                      variant={run.status === "running" ? "success" : run.status === "finished" ? "info" : "default"}
                      dot
                    >
                      {run.status}
                    </Badge>
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      ...typeScale.mono,
                      color: run.metrics.cumulativeReward > 0 ? colors.accent : colors.error,
                    }}>
                      {run.metrics.cumulativeReward.toFixed(3)}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      ...typeScale.mono,
                      color: run.metrics.deltaVsMain > 0 ? colors.success : run.metrics.deltaVsMain < -0.1 ? colors.error : colors.textSecondary,
                    }}>
                      {run.metrics.deltaVsMain >= 0 ? "+" : ""}{run.metrics.deltaVsMain.toFixed(3)}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    <span style={{
                      ...typeScale.mono,
                      color: run.metrics.minTTC < 3 ? colors.warning : colors.textSecondary,
                    }}>
                      {run.metrics.minTTC < 100 ? `${run.metrics.minTTC.toFixed(1)}s` : "safe"}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    <span style={{ ...typeScale.mono, color: colors.textDim }}>
                      {new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
                    </span>
                  </td>
                  <td style={tdStyle}>
                    <button
                      onClick={() => navigate("/dashboard")}
                      style={{
                        padding: "3px 8px",
                        borderRadius: 4,
                        background: "rgba(0,232,157,0.06)",
                        border: `1px solid ${colors.borderAccent}`,
                        color: colors.accent,
                        fontSize: 10,
                        cursor: "pointer",
                        fontFamily: fonts.sans,
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 4,
                      }}
                    >
                      View
                      <ExternalLink size={10} />
                    </button>
                  </td>
                </tr>
              ))}
              {filteredRuns.length === 0 && (
                <tr>
                  <td colSpan={7} style={{ ...tdStyle, textAlign: "center", color: colors.textDim, padding: 24 }}>
                    No runs match the current filter
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Incident Ticket Feed ── */}
      <div style={{
        background: colors.bgCard,
        border: `1px solid ${colors.border}`,
        borderRadius: radius.lg,
        overflow: "hidden",
      }}>
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: `${spacing.md}px ${spacing.lg}px`,
          borderBottom: `1px solid ${colors.border}`,
        }}>
          <div style={{ ...typeScale.h3, color: colors.textPrimary }}>
            Incident Ticket Feed
          </div>
          <div style={{ display: "flex", gap: spacing.sm }}>
            <FilterPill label="All" active={ticketFilter === "all"} onClick={() => setTicketFilter("all")} />
            <FilterPill label="Critical" active={ticketFilter === "critical"} onClick={() => setTicketFilter("critical")} />
            <FilterPill label="Warning" active={ticketFilter === "warning"} onClick={() => setTicketFilter("warning")} />
            <FilterPill label="Info" active={ticketFilter === "info"} onClick={() => setTicketFilter("info")} />
          </div>
        </div>

        <div style={{ maxHeight: 300, overflow: "auto", padding: spacing.sm }}>
          {filteredTickets.map((ticket, i) => (
            <div key={i} style={{
              display: "flex",
              alignItems: "flex-start",
              gap: spacing.md,
              padding: `${spacing.sm}px ${spacing.md}px`,
              borderRadius: 6,
              marginBottom: 2,
              background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent",
            }}>
              <div style={{
                width: 4,
                height: 4,
                borderRadius: "50%",
                marginTop: 6,
                background: ticket.severity === "critical" ? colors.error
                  : ticket.severity === "warning" ? colors.warning
                  : colors.info,
                flexShrink: 0,
              }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                  <span style={{ ...typeScale.small, fontWeight: 500, color: colors.textPrimary }}>
                    {ticket.title}
                  </span>
                  <Badge
                    variant={ticket.severity === "critical" ? "error" : ticket.severity === "warning" ? "warning" : "info"}
                  >
                    {ticket.severity}
                  </Badge>
                </div>
                <p style={{ ...typeScale.small, color: colors.textDim, margin: 0 }}>
                  {ticket.description}
                </p>
              </div>
              <span style={{ ...typeScale.mono, color: colors.textDim, fontSize: 9, flexShrink: 0 }}>
                {ticket.time}
              </span>
            </div>
          ))}
          {filteredTickets.length === 0 && (
            <div style={{ textAlign: "center", padding: spacing.xl, color: colors.textDim, ...typeScale.small }}>
              No tickets match the current filter
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Reward timeline SVG
// ---------------------------------------------------------------------------

function RewardTimeline({ runs, mainReward }: { runs: CounterfactualRun[]; mainReward: number }) {
  const width = 800;
  const height = 120;
  const padding = { top: 10, right: 20, bottom: 20, left: 40 };
  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  // Use runs sorted by creation time
  const sorted = [...runs].sort((a, b) => a.createdAt - b.createdAt);
  if (sorted.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: spacing.lg, color: colors.textDim, ...typeScale.small }}>
        Waiting for runs to populate the timeline
      </div>
    );
  }

  const minTime = sorted[0].createdAt;
  const maxTime = sorted[sorted.length - 1].createdAt;
  const timeRange = Math.max(maxTime - minTime, 1000);

  const allRewards = sorted.map((r) => r.metrics.cumulativeReward);
  const minR = Math.min(...allRewards, mainReward, 0);
  const maxR = Math.max(...allRewards, mainReward, 1);
  const rewardRange = Math.max(maxR - minR, 0.1);

  const xScale = (t: number) => padding.left + ((t - minTime) / timeRange) * plotW;
  const yScale = (r: number) => padding.top + plotH - ((r - minR) / rewardRange) * plotH;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: "block" }}>
      {/* Grid lines */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => {
        const y = padding.top + f * plotH;
        const val = maxR - f * rewardRange;
        return (
          <g key={f}>
            <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke={colors.border} strokeWidth={0.5} />
            <text x={padding.left - 4} y={y + 3} fill={colors.textDim} fontSize={8} textAnchor="end" fontFamily={fonts.mono}>
              {val.toFixed(1)}
            </text>
          </g>
        );
      })}

      {/* Main reward line */}
      <line
        x1={padding.left}
        y1={yScale(mainReward)}
        x2={width - padding.right}
        y2={yScale(mainReward)}
        stroke={colors.warning}
        strokeWidth={1}
        strokeDasharray="4 3"
        opacity={0.6}
      />
      <text
        x={width - padding.right + 2}
        y={yScale(mainReward) + 3}
        fill={colors.warning}
        fontSize={8}
        fontFamily={fonts.mono}
      >
        main
      </text>

      {/* Data points */}
      {sorted.map((run, i) => {
        const x = xScale(run.createdAt);
        const y = yScale(run.metrics.cumulativeReward);
        const col = run.metrics.deltaVsMain > 0 ? colors.success : run.metrics.deltaVsMain < -0.1 ? colors.error : colors.accent;
        return (
          <g key={run.id}>
            <circle cx={x} cy={y} r={4} fill={col} opacity={run.status === "replaced" ? 0.3 : 0.8} />
            {i > 0 && (
              <line
                x1={xScale(sorted[i - 1].createdAt)}
                y1={yScale(sorted[i - 1].metrics.cumulativeReward)}
                x2={x}
                y2={y}
                stroke={colors.accent}
                strokeWidth={0.5}
                opacity={0.3}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Helper components
// ---------------------------------------------------------------------------

function OverviewCard({
  icon: Icon,
  label,
  value,
  subtext,
  color,
}: {
  icon: typeof Shield;
  label: string;
  value: string | number;
  subtext: string;
  color: string;
}) {
  return (
    <div style={{
      background: colors.bgCard,
      border: `1px solid ${colors.border}`,
      borderRadius: radius.lg,
      padding: spacing.lg,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: spacing.sm }}>
        <div style={{
          width: 28,
          height: 28,
          borderRadius: 6,
          background: `${color}10`,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}>
          <Icon size={14} color={color} />
        </div>
        <span style={{ ...typeScale.caption, color: colors.textDim }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 700, color, fontFamily: fonts.mono, lineHeight: 1 }}>
        {value}
      </div>
      <div style={{ ...typeScale.small, color: colors.textDim, marginTop: 4 }}>
        {subtext}
      </div>
    </div>
  );
}

function FilterPill({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "3px 10px",
        borderRadius: 4,
        fontSize: 10,
        fontFamily: fonts.sans,
        fontWeight: active ? 500 : 400,
        background: active ? "rgba(0,232,157,0.08)" : "transparent",
        color: active ? colors.accent : colors.textDim,
        border: `1px solid ${active ? colors.borderAccent : "transparent"}`,
        cursor: "pointer",
      }}
    >
      {label}
    </button>
  );
}

function SortHeader({
  label,
  field,
  current,
  dir,
  onSort,
}: {
  label: string;
  field: SortField;
  current: SortField;
  dir: SortDir;
  onSort: (f: SortField) => void;
}) {
  const active = current === field;
  return (
    <th
      onClick={() => onSort(field)}
      style={{
        ...thStyle,
        cursor: "pointer",
        color: active ? colors.accent : colors.textDim,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
        {label}
        {active && (dir === "desc" ? <ChevronDown size={10} /> : <ChevronUp size={10} />)}
      </div>
    </th>
  );
}

const thStyle: React.CSSProperties = {
  padding: "8px 12px",
  textAlign: "left",
  fontSize: 10,
  fontWeight: 500,
  color: colors.textDim,
  textTransform: "uppercase",
  letterSpacing: "0.04em",
  borderBottom: `1px solid ${colors.border}`,
  fontFamily: fonts.sans,
  whiteSpace: "nowrap",
};

const tdStyle: React.CSSProperties = {
  padding: "8px 12px",
  verticalAlign: "middle",
};

// ---------------------------------------------------------------------------
// Generate tickets from runs
// ---------------------------------------------------------------------------

interface Ticket {
  title: string;
  description: string;
  severity: "critical" | "warning" | "info";
  time: string;
}

function generateTickets(runs: CounterfactualRun[]): Ticket[] {
  const tickets: Ticket[] = [];

  runs.forEach((run) => {
    // Check for critical TTC
    if (run.metrics.minTTC < 2) {
      tickets.push({
        title: `TTC Drop: ${run.label}`,
        description: `Minimum time-to-collision dropped to ${run.metrics.minTTC.toFixed(1)}s during ${run.label}`,
        severity: "critical",
        time: new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
    }

    // Check for hard brakes
    if (run.metrics.interventionCount > 0) {
      tickets.push({
        title: `Hard Brake: ${run.label}`,
        description: `${run.metrics.interventionCount} emergency brake(s) triggered`,
        severity: "warning",
        time: new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
    }

    // Check for positive delta (good news)
    if (run.metrics.deltaVsMain > 0.5) {
      tickets.push({
        title: `Improved: ${run.label}`,
        description: `Reward +${run.metrics.deltaVsMain.toFixed(2)} vs ground truth`,
        severity: "info",
        time: new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
    }

    // Negative reward
    if (run.metrics.cumulativeReward < -0.5) {
      tickets.push({
        title: `Poor Outcome: ${run.label}`,
        description: `Negative cumulative reward: ${run.metrics.cumulativeReward.toFixed(2)}`,
        severity: "warning",
        time: new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      });
    }
  });

  return tickets.sort((a, b) => {
    const ord = { critical: 0, warning: 1, info: 2 };
    return ord[a.severity] - ord[b.severity];
  });
}
