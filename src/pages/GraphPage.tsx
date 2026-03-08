/**
 * GraphPage — Knowledge graph using ForceGraph3D (react-force-graph-3d).
 * Connects: scenario → incidents → tickets → actions → metrics → rewards.
 */

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import ForceGraph3D from "react-force-graph-3d";
import { X, ExternalLink } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Badge from "../components/ui/Badge";
import { useStore } from "../store";
import { useSimManager } from "../lib/simManager";
import { colors, fonts, typeScale, spacing, glass, radius } from "../theme";
import type { CounterfactualRun } from "../lib/simTypes";

// ---------------------------------------------------------------------------
// Graph data types
// ---------------------------------------------------------------------------

interface GraphNode {
  id: string;
  label: string;
  type: "scenario" | "incident" | "ticket" | "run" | "action" | "metric" | "reward";
  val: number; // size
  color: string;
  details?: Record<string, string | number>;
}

interface GraphLink {
  source: string;
  target: string;
  label: string;
  color: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

// Consistent low-saturation palette
const NODE_COLORS: Record<GraphNode["type"], string> = {
  scenario: "#60A5FA",
  incident: "#F59E0B",
  ticket: "#EF4444",
  run: "#00E89D",
  action: "#A78BFA",
  metric: "#6EE7B7",
  reward: "#FBBF24",
};

// ---------------------------------------------------------------------------
// Build graph from sim data
// ---------------------------------------------------------------------------

function buildGraphData(
  mockScenario: string,
  runs: CounterfactualRun[],
  mainReward: number,
): GraphData {
  const nodes: GraphNode[] = [];
  const links: GraphLink[] = [];
  const nodeIds = new Set<string>();

  const addNode = (n: GraphNode) => {
    if (nodeIds.has(n.id)) return;
    nodeIds.add(n.id);
    nodes.push(n);
  };

  const addLink = (l: GraphLink) => {
    links.push(l);
  };

  // Scenario node
  const scenarioId = `scenario-${mockScenario}`;
  addNode({
    id: scenarioId,
    label: mockScenario.replace(/_/g, " "),
    type: "scenario",
    val: 12,
    color: NODE_COLORS.scenario,
    details: { type: "Waymo Segment", scenario: mockScenario },
  });

  // Incidents based on scenario type
  const incidents = getIncidentsForScenario(mockScenario);
  incidents.forEach((inc, i) => {
    const incId = `incident-${i}`;
    addNode({
      id: incId,
      label: inc.label,
      type: "incident",
      val: 8,
      color: NODE_COLORS.incident,
      details: { severity: inc.severity, time: inc.time, description: inc.desc },
    });
    addLink({
      source: scenarioId,
      target: incId,
      label: "has_incident",
      color: "rgba(245,158,11,0.3)",
    });

    // Ticket for each incident
    const ticketId = `ticket-${i}`;
    addNode({
      id: ticketId,
      label: `TKT-${1000 + i}`,
      type: "ticket",
      val: 6,
      color: NODE_COLORS.ticket,
      details: { status: "open", priority: inc.severity, incident: inc.label },
    });
    addLink({
      source: incId,
      target: ticketId,
      label: "triggered",
      color: "rgba(239,68,68,0.3)",
    });
  });

  // Counterfactual runs
  runs.slice(0, 12).forEach((run) => {
    const runId = `run-${run.id}`;
    addNode({
      id: runId,
      label: run.label,
      type: "run",
      val: 7,
      color: NODE_COLORS.run,
      details: {
        status: run.status,
        steps: run.actionStream.length,
        reward: run.metrics.cumulativeReward,
        delta: run.metrics.deltaVsMain,
      },
    });

    // Link run to scenario
    addLink({
      source: scenarioId,
      target: runId,
      label: "generated",
      color: "rgba(0,232,157,0.2)",
    });

    // Actions used in this run (unique)
    const uniqueActions = new Set(run.actionStream.map((a) => a.action));
    uniqueActions.forEach((action) => {
      const actionId = `action-${action}`;
      addNode({
        id: actionId,
        label: action.replace(/_/g, " "),
        type: "action",
        val: 5,
        color: NODE_COLORS.action,
        details: { type: "ego_action" },
      });
      addLink({
        source: runId,
        target: actionId,
        label: "used_action",
        color: "rgba(167,139,250,0.2)",
      });
    });

    // Metrics
    const metricId = `metric-ttc-${run.id}`;
    addNode({
      id: metricId,
      label: `TTC: ${run.metrics.minTTC < 100 ? run.metrics.minTTC.toFixed(1) + "s" : "safe"}`,
      type: "metric",
      val: 4,
      color: NODE_COLORS.metric,
      details: { minTTC: run.metrics.minTTC, interventions: run.metrics.interventionCount },
    });

    // Pick the most-used action for metric link
    const topAction = run.actionStream.length > 0
      ? run.actionStream[run.actionStream.length - 1].action
      : "keep_lane";
    addLink({
      source: `action-${topAction}`,
      target: metricId,
      label: "changed_metric",
      color: "rgba(110,231,183,0.2)",
    });

    // Reward node
    const rewardId = `reward-${run.id}`;
    addNode({
      id: rewardId,
      label: `R: ${run.metrics.cumulativeReward.toFixed(2)}`,
      type: "reward",
      val: 4,
      color: NODE_COLORS.reward,
      details: { cumulative: run.metrics.cumulativeReward, average: run.metrics.avgReward },
    });
    addLink({
      source: runId,
      target: rewardId,
      label: "achieved_reward",
      color: "rgba(251,191,36,0.2)",
    });
  });

  // Main sim reward node
  addNode({
    id: "reward-main",
    label: `Main R: ${mainReward.toFixed(2)}`,
    type: "reward",
    val: 8,
    color: "#FF9E00",
    details: { type: "ground_truth", cumulative: mainReward },
  });
  addLink({
    source: scenarioId,
    target: "reward-main",
    label: "ground_truth_reward",
    color: "rgba(255,158,0,0.3)",
  });

  return { nodes, links };
}

function getIncidentsForScenario(scenario: string) {
  const incidents: { label: string; severity: string; time: string; desc: string }[] = [];
  switch (scenario) {
    case "near_miss":
      incidents.push({ label: "Near Miss", severity: "critical", time: "8.0s", desc: "Vehicle swerved into ego lane" });
      break;
    case "rear_end":
      incidents.push({ label: "Hard Brake", severity: "warning", time: "6.0s", desc: "Lead vehicle emergency brake" });
      break;
    case "jaywalker":
      incidents.push({ label: "Pedestrian Incursion", severity: "critical", time: "7.0s", desc: "Jaywalker entered roadway" });
      break;
    case "red_light_runner":
      incidents.push({ label: "Red Light Violation", severity: "critical", time: "9.0s", desc: "Cross traffic ran red light" });
      break;
    case "swerving_vehicle":
      incidents.push({ label: "Erratic Driver", severity: "warning", time: "5.0s", desc: "Adjacent vehicle swerving" });
      break;
    default:
      incidents.push({ label: "Routine Check", severity: "info", time: "—", desc: "No incident detected" });
  }
  return incidents;
}

// ---------------------------------------------------------------------------
// Graph page component
// ---------------------------------------------------------------------------

export default function GraphPage() {
  const navigate = useNavigate();
  const mockScenario = useStore((s) => s.mockScenario);
  const runs = useSimManager((s) => s.runs);
  const mainReward = useSimManager((s) => s.mainState.cumulativeReward);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

  // Measure container
  useEffect(() => {
    const measure = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, []);

  const graphData = useMemo(
    () => buildGraphData(mockScenario, runs, mainReward),
    [mockScenario, runs, mainReward],
  );

  const handleNodeClick = useCallback((node: object) => {
    setSelectedNode(node as GraphNode);
  }, []);

  return (
    <div style={{
      display: "flex",
      height: "100%",
      overflow: "hidden",
    }}>
      {/* Graph area */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          position: "relative",
          background: colors.bgDeep,
        }}
      >
        <ForceGraph3D
          graphData={graphData}
          width={dimensions.width - (selectedNode ? 320 : 0)}
          height={dimensions.height}
          backgroundColor={colors.bgDeep}
          nodeLabel={(node: object) => (node as GraphNode).label}
          nodeVal={(node: object) => (node as GraphNode).val}
          nodeColor={(node: object) => (node as GraphNode).color}
          nodeOpacity={0.9}
          linkLabel={(link: object) => (link as GraphLink).label}
          linkColor={(link: object) => (link as GraphLink).color}
          linkWidth={1}
          linkOpacity={0.4}
          linkDirectionalParticles={2}
          linkDirectionalParticleWidth={1.5}
          linkDirectionalParticleSpeed={0.005}
          linkDirectionalParticleColor={(link: object) => (link as GraphLink).color}
          onNodeClick={handleNodeClick}
          nodeThreeObject={undefined}
          enableNodeDrag={true}
          enableNavigationControls={true}
          showNavInfo={false}
        />

        {/* Legend overlay */}
        <div style={{
          position: "absolute",
          bottom: spacing.lg,
          left: spacing.lg,
          ...glass,
          padding: spacing.md,
          display: "flex",
          gap: spacing.md,
          flexWrap: "wrap",
        }}>
          {Object.entries(NODE_COLORS).map(([type, color]) => (
            <div key={type} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: color,
              }} />
              <span style={{ ...typeScale.caption, color: colors.textSecondary, fontSize: 9 }}>
                {type}
              </span>
            </div>
          ))}
        </div>

        {/* Node count */}
        <div style={{
          position: "absolute",
          top: spacing.lg,
          left: spacing.lg,
        }}>
          <h2 style={{ ...typeScale.h2, color: colors.textPrimary, margin: 0, marginBottom: 4 }}>
            Knowledge Graph
          </h2>
          <span style={{ ...typeScale.small, color: colors.textDim }}>
            {graphData.nodes.length} nodes, {graphData.links.length} edges
          </span>
        </div>
      </div>

      {/* Inspector panel */}
      {selectedNode && (
        <div style={{
          width: 320,
          borderLeft: `1px solid ${colors.border}`,
          background: colors.bgBase,
          padding: spacing.lg,
          overflow: "auto",
          flexShrink: 0,
        }}>
          <div style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: spacing.lg,
          }}>
            <div>
              <Badge variant={nodeTypeToVariant(selectedNode.type)}>
                {selectedNode.type}
              </Badge>
              <h3 style={{ ...typeScale.h3, color: colors.textPrimary, marginTop: spacing.sm }}>
                {selectedNode.label}
              </h3>
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              style={{ cursor: "pointer", padding: 4, border: "none", background: "none" }}
            >
              <X size={16} color={colors.textDim} />
            </button>
          </div>

          <div style={{
            display: "flex",
            flexDirection: "column",
            gap: spacing.sm,
          }}>
            <div style={{ ...typeScale.caption, color: colors.textDim, marginBottom: 2 }}>
              Properties
            </div>
            {selectedNode.details && Object.entries(selectedNode.details).map(([key, val]) => (
              <div key={key} style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "6px 8px",
                borderRadius: 4,
                background: "rgba(255,255,255,0.02)",
              }}>
                <span style={{ ...typeScale.small, color: colors.textDim }}>{key}</span>
                <span style={{ ...typeScale.mono, color: colors.textSecondary }}>{String(val)}</span>
              </div>
            ))}
          </div>

          {/* Quick links */}
          {selectedNode.type === "run" && (
            <button
              onClick={() => navigate("/dashboard")}
              style={{
                marginTop: spacing.lg,
                width: "100%",
                padding: "8px 12px",
                borderRadius: 6,
                background: "rgba(0,232,157,0.06)",
                border: `1px solid ${colors.borderAccent}`,
                color: colors.accent,
                fontSize: 12,
                fontFamily: fonts.sans,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: 6,
              }}
            >
              Open in Dashboard
              <ExternalLink size={12} />
            </button>
          )}

          {/* Connected nodes */}
          <div style={{ marginTop: spacing.lg }}>
            <div style={{ ...typeScale.caption, color: colors.textDim, marginBottom: spacing.sm }}>
              Connections
            </div>
            {graphData.links
              .filter((l) => l.source === selectedNode.id || l.target === selectedNode.id)
              .slice(0, 10)
              .map((l, i) => {
                const otherId = l.source === selectedNode.id ? l.target : l.source;
                const otherNode = graphData.nodes.find((n) => n.id === otherId);
                return (
                  <div
                    key={i}
                    onClick={() => otherNode && setSelectedNode(otherNode)}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      padding: "6px 8px",
                      borderRadius: 4,
                      cursor: "pointer",
                      background: "rgba(255,255,255,0.02)",
                      marginBottom: 2,
                    }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{
                        width: 6,
                        height: 6,
                        borderRadius: "50%",
                        background: otherNode?.color || colors.textDim,
                      }} />
                      <span style={{ ...typeScale.small, color: colors.textSecondary }}>
                        {otherNode?.label || otherId}
                      </span>
                    </div>
                    <span style={{ ...typeScale.caption, color: colors.textDim, fontSize: 8 }}>
                      {l.label}
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </div>
  );
}

function nodeTypeToVariant(type: GraphNode["type"]) {
  switch (type) {
    case "scenario": return "info" as const;
    case "incident": return "warning" as const;
    case "ticket": return "error" as const;
    case "run": return "success" as const;
    case "action": return "accent" as const;
    case "metric": return "info" as const;
    case "reward": return "warning" as const;
    default: return "default" as const;
  }
}
