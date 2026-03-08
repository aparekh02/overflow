/**
 * GraphPage — Premium 3D knowledge graph with custom glowing nodes,
 * HUD overlays, and interactive inspector panel. Connects:
 * scenario -> incidents -> tickets -> runs -> actions -> metrics -> rewards.
 */

import { useState, useEffect, useMemo, useRef, useCallback } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import { X, ExternalLink, GitBranch, Zap, AlertTriangle, Activity } from "lucide-react";
import { useNavigate } from "react-router-dom";
import Badge from "../components/ui/Badge";
import { useStore } from "../store";
import { useSimManager } from "../lib/simManager";
import { colors, fonts, typeScale, spacing, radius } from "../theme";
import type { CounterfactualRun } from "../lib/simTypes";

/* ---------- Graph types ---------- */

interface GraphNode {
  id: string;
  label: string;
  type: "scenario" | "incident" | "ticket" | "run" | "action" | "metric" | "reward";
  val: number;
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

const NODE_COLORS: Record<GraphNode["type"], string> = {
  scenario: "#60A5FA",
  incident: "#F59E0B",
  ticket: "#EF4444",
  run: "#00E89D",
  action: "#A78BFA",
  metric: "#6EE7B7",
  reward: "#FBBF24",
};

const NODE_ICONS: Record<GraphNode["type"], string> = {
  scenario: "\u25C6",
  incident: "\u26A0",
  ticket: "\u2605",
  run: "\u25B6",
  action: "\u2BC1",
  metric: "\u25CF",
  reward: "\u2736",
};

/* ---------- Shared THREE.js geometries ---------- */

const SPHERE_GEO = new THREE.SphereGeometry(1, 32, 16);
const GLOW_GEO = new THREE.SphereGeometry(1, 16, 8);
const RING_GEO = new THREE.TorusGeometry(1, 0.06, 8, 48);

/* ---------- Build graph data ---------- */

function buildGraphData(
  scenarioId: string,
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

  const addLink = (l: GraphLink) => links.push(l);

  const scenNodeId = `scenario-${scenarioId}`;
  addNode({
    id: scenNodeId,
    label: scenarioId.replace(/_/g, " "),
    type: "scenario",
    val: 14,
    color: NODE_COLORS.scenario,
    details: { type: "Scenario", scenario: scenarioId },
  });

  const incidents = getIncidentsForScenario(scenarioId);
  incidents.forEach((inc, i) => {
    const incId = `incident-${i}`;
    addNode({
      id: incId, label: inc.label, type: "incident", val: 9,
      color: NODE_COLORS.incident,
      details: { severity: inc.severity, time: inc.time, description: inc.desc },
    });
    addLink({ source: scenNodeId, target: incId, label: "has_incident", color: "rgba(245,158,11,0.35)" });

    const ticketId = `ticket-${i}`;
    addNode({
      id: ticketId, label: `TKT-${1000 + i}`, type: "ticket", val: 7,
      color: NODE_COLORS.ticket,
      details: { status: "open", priority: inc.severity, incident: inc.label },
    });
    addLink({ source: incId, target: ticketId, label: "triggered", color: "rgba(239,68,68,0.35)" });
  });

  runs.slice(0, 12).forEach((run) => {
    const runId = `run-${run.id}`;
    addNode({
      id: runId, label: run.label, type: "run", val: 8,
      color: NODE_COLORS.run,
      details: {
        status: run.status, steps: run.actionStream.length,
        reward: run.metrics.cumulativeReward, delta: run.metrics.deltaVsMain,
      },
    });
    addLink({ source: scenNodeId, target: runId, label: "generated", color: "rgba(0,232,157,0.25)" });

    const uniqueActions = new Set(run.actionStream.map((a) => a.action));
    uniqueActions.forEach((action) => {
      const actionId = `action-${action}`;
      addNode({
        id: actionId, label: action.replace(/_/g, " "), type: "action", val: 5,
        color: NODE_COLORS.action, details: { type: "ego_action" },
      });
      addLink({ source: runId, target: actionId, label: "used_action", color: "rgba(167,139,250,0.25)" });
    });

    const metricId = `metric-ttc-${run.id}`;
    addNode({
      id: metricId,
      label: `TTC: ${run.metrics.minTTC < 100 ? run.metrics.minTTC.toFixed(1) + "s" : "safe"}`,
      type: "metric", val: 4, color: NODE_COLORS.metric,
      details: { minTTC: run.metrics.minTTC, interventions: run.metrics.interventionCount },
    });
    const topAction = run.actionStream.length > 0 ? run.actionStream[run.actionStream.length - 1].action : "keep_lane";
    addLink({ source: `action-${topAction}`, target: metricId, label: "changed_metric", color: "rgba(110,231,183,0.25)" });

    const rewardId = `reward-${run.id}`;
    addNode({
      id: rewardId, label: `R: ${run.metrics.cumulativeReward.toFixed(2)}`,
      type: "reward", val: 5, color: NODE_COLORS.reward,
      details: { cumulative: run.metrics.cumulativeReward, average: run.metrics.avgReward },
    });
    addLink({ source: runId, target: rewardId, label: "achieved_reward", color: "rgba(251,191,36,0.25)" });
  });

  addNode({
    id: "reward-main", label: `Main R: ${mainReward.toFixed(2)}`,
    type: "reward", val: 10, color: "#FF9E00",
    details: { type: "ground_truth", cumulative: mainReward },
  });
  addLink({ source: scenNodeId, target: "reward-main", label: "ground_truth_reward", color: "rgba(255,158,0,0.35)" });

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
      incidents.push({ label: "Routine Check", severity: "info", time: "\u2014", desc: "No incident detected" });
  }
  return incidents;
}

/* ---------- Custom Three.js node creator ---------- */

function createTextTexture(text: string, color: string, bgAlpha = 0.5): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d")!;
  canvas.width = 256;
  canvas.height = 64;
  ctx.clearRect(0, 0, 256, 64);

  ctx.font = "bold 22px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  const tw = ctx.measureText(text).width;
  const pw = tw + 24;
  const ph = 28;
  const px = 128 - pw / 2;
  const py = 18;

  // Background pill
  ctx.fillStyle = `rgba(10,13,22,${bgAlpha})`;
  ctx.beginPath();
  ctx.roundRect(px, py, pw, ph, 6);
  ctx.fill();
  ctx.strokeStyle = color + "40";
  ctx.lineWidth = 1;
  ctx.stroke();

  // Text
  ctx.fillStyle = color;
  ctx.fillText(text, 128, 38);

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}

function createNodeObject(node: GraphNode): THREE.Object3D {
  const group = new THREE.Group();
  const size = Math.cbrt(node.val) * 1.6;
  const color = new THREE.Color(node.color);

  // Core sphere
  const coreMat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.85 });
  const core = new THREE.Mesh(SPHERE_GEO, coreMat);
  core.scale.setScalar(size);
  group.add(core);

  // Inner bright point
  const pointMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 });
  const point = new THREE.Mesh(SPHERE_GEO, pointMat);
  point.scale.setScalar(size * 0.25);
  group.add(point);

  // Outer glow
  const glowMat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.06, depthWrite: false, side: THREE.BackSide,
  });
  const glow = new THREE.Mesh(GLOW_GEO, glowMat);
  glow.scale.setScalar(size * 3);
  group.add(glow);

  // Second glow layer
  const glow2Mat = new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.03, depthWrite: false, side: THREE.BackSide,
  });
  const glow2 = new THREE.Mesh(GLOW_GEO, glow2Mat);
  glow2.scale.setScalar(size * 5);
  group.add(glow2);

  // Ring for scenario & run nodes
  if (node.type === "scenario" || node.type === "run") {
    const ringMat = new THREE.MeshBasicMaterial({
      color, transparent: true, opacity: 0.3, side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(RING_GEO, ringMat);
    ring.scale.setScalar(size * 2);
    ring.rotation.x = Math.PI / 2;
    group.add(ring);

    // Second ring at angle
    const ring2 = new THREE.Mesh(RING_GEO, ringMat.clone());
    ring2.material.opacity = 0.15;
    ring2.scale.setScalar(size * 2.5);
    ring2.rotation.x = Math.PI / 3;
    ring2.rotation.z = Math.PI / 4;
    group.add(ring2);
  }

  // Text label sprite
  const labelText = node.label.toUpperCase();
  const texture = createTextTexture(labelText, node.color);
  const spriteMat = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(spriteMat);
  const labelScale = Math.max(size * 3.5, 10);
  sprite.scale.set(labelScale, labelScale * 0.25, 1);
  sprite.position.set(0, -(size + 2.5), 0);
  group.add(sprite);

  return group;
}

/* ---------- Component ---------- */

export default function GraphPage() {
  const navigate = useNavigate();
  const scenarioId = useStore((s) => s.scenarioId);
  const runs = useSimManager((s) => s.runs);
  const mainReward = useSimManager((s) => s.mainState.cumulativeReward);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

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
    () => buildGraphData(scenarioId, runs, mainReward),
    [scenarioId, runs, mainReward],
  );

  const handleNodeClick = useCallback((node: object) => {
    setSelectedNode(node as GraphNode);
  }, []);

  const nodeThreeObject = useCallback((node: object) => {
    return createNodeObject(node as GraphNode);
  }, []);

  const stats = useMemo(() => {
    const typeCounts: Record<string, number> = {};
    graphData.nodes.forEach((n) => { typeCounts[n.type] = (typeCounts[n.type] || 0) + 1; });
    return typeCounts;
  }, [graphData]);

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden", position: "relative" }}>

      {/* 3D Graph area */}
      <div
        ref={containerRef}
        style={{ flex: 1, position: "relative", background: colors.bgDeep }}
      >
        <ForceGraph3D
          graphData={graphData}
          width={dimensions.width - (selectedNode ? 360 : 0)}
          height={dimensions.height}
          backgroundColor={"#060810"}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          onNodeClick={handleNodeClick}
          linkColor={(link: object) => (link as GraphLink).color}
          linkWidth={1.5}
          linkOpacity={0.5}
          linkDirectionalParticles={3}
          linkDirectionalParticleWidth={2}
          linkDirectionalParticleSpeed={0.004}
          linkDirectionalParticleColor={(link: object) => (link as GraphLink).color}
          enableNodeDrag={true}
          enableNavigationControls={true}
          showNavInfo={false}
          warmupTicks={50}
          cooldownTime={3000}
        />

        {/* HUD corner brackets */}
        {[
          { top: 0, left: 0, borderTop: "2px solid rgba(0,232,157,0.2)", borderLeft: "2px solid rgba(0,232,157,0.2)" },
          { top: 0, right: 0, borderTop: "2px solid rgba(0,232,157,0.2)", borderRight: "2px solid rgba(0,232,157,0.2)" },
          { bottom: 0, left: 0, borderBottom: "2px solid rgba(0,232,157,0.2)", borderLeft: "2px solid rgba(0,232,157,0.2)" },
          { bottom: 0, right: 0, borderBottom: "2px solid rgba(0,232,157,0.2)", borderRight: "2px solid rgba(0,232,157,0.2)" },
        ].map((s, i) => (
          <div key={i} style={{ position: "absolute", width: 24, height: 24, pointerEvents: "none", ...s } as React.CSSProperties} />
        ))}

        {/* Header overlay */}
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0,
          padding: `${spacing.lg}px ${spacing.xl}px`,
          background: "linear-gradient(180deg, rgba(6,8,16,0.85) 0%, transparent 100%)",
          pointerEvents: "none",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: "rgba(0,232,157,0.08)",
              border: "1px solid rgba(0,232,157,0.2)",
              display: "flex", alignItems: "center", justifyContent: "center",
            }}>
              <GitBranch size={16} color={colors.accent} />
            </div>
            <div>
              <h2 style={{
                ...typeScale.h2, margin: 0,
                background: "linear-gradient(135deg, #E8ECF4, #00E89D)",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                letterSpacing: "-0.02em",
              }}>
                Knowledge Graph
              </h2>
              <div style={{ display: "flex", alignItems: "center", gap: 12, marginTop: 2 }}>
                <span style={{ ...typeScale.caption, color: colors.textDim, fontSize: 9 }}>
                  {graphData.nodes.length} NODES
                </span>
                <span style={{ ...typeScale.caption, color: colors.textDim, fontSize: 9 }}>
                  {graphData.links.length} EDGES
                </span>
                <span style={{
                  display: "inline-block", width: 5, height: 5, borderRadius: "50%",
                  background: colors.accent, boxShadow: `0 0 8px ${colors.accent}`,
                }} />
              </div>
            </div>
          </div>

          {/* Stats chips */}
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            {Object.entries(stats).map(([type, count]) => (
              <div key={type} style={{
                display: "flex", alignItems: "center", gap: 4,
                padding: "3px 8px", borderRadius: radius.pill,
                background: "rgba(255,255,255,0.03)",
                border: `1px solid ${NODE_COLORS[type as GraphNode["type"]] || colors.border}20`,
              }}>
                <div style={{
                  width: 6, height: 6, borderRadius: "50%",
                  background: NODE_COLORS[type as GraphNode["type"]] || colors.textDim,
                  boxShadow: `0 0 4px ${NODE_COLORS[type as GraphNode["type"]] || "transparent"}60`,
                }} />
                <span style={{ fontSize: 8, fontFamily: fonts.mono, color: colors.textDim }}>
                  {count} {type}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Legend overlay */}
        <div style={{
          position: "absolute",
          bottom: spacing.xl,
          left: spacing.xl,
          background: "rgba(10,13,22,0.85)",
          backdropFilter: "blur(20px)",
          border: "1px solid rgba(0,232,157,0.08)",
          borderRadius: radius.lg,
          padding: `${spacing.sm}px ${spacing.md}px`,
          pointerEvents: "none",
        }}>
          <div style={{
            ...typeScale.caption, color: colors.textDim, fontSize: 8,
            marginBottom: 6, letterSpacing: "1.5px",
          }}>
            NODE TYPES
          </div>
          <div style={{ display: "flex", gap: spacing.md, flexWrap: "wrap" }}>
            {Object.entries(NODE_COLORS).map(([type, color]) => (
              <div key={type} style={{ display: "flex", alignItems: "center", gap: 5 }}>
                <div style={{
                  width: 8, height: 8, borderRadius: "50%",
                  background: color,
                  boxShadow: `0 0 6px ${color}50`,
                }} />
                <span style={{ fontSize: 9, fontFamily: fonts.mono, color: colors.textSecondary }}>
                  {type}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Bottom-right interaction hint */}
        <div style={{
          position: "absolute", bottom: spacing.xl, right: spacing.xl,
          display: "flex", gap: 8, pointerEvents: "none",
        }}>
          {["Scroll: Zoom", "Drag: Rotate", "Click: Inspect"].map((hint, i) => (
            <span key={i} style={{
              fontSize: 8, fontFamily: fonts.mono, color: colors.textMuted,
              padding: "3px 8px", borderRadius: radius.sm,
              background: "rgba(255,255,255,0.02)",
              border: "1px solid rgba(255,255,255,0.04)",
            }}>
              {hint}
            </span>
          ))}
        </div>
      </div>

      {/* Inspector panel */}
      {selectedNode && (
        <div style={{
          width: 360, flexShrink: 0,
          borderLeft: `1px solid ${colors.border}`,
          background: "linear-gradient(180deg, #0F1220 0%, #0A0D16 100%)",
          overflow: "auto",
          position: "relative",
        }}>
          {/* Accent top line */}
          <div style={{
            height: 2,
            background: `linear-gradient(90deg, ${selectedNode.color}, transparent)`,
            opacity: 0.5,
          }} />

          <div style={{ padding: spacing.lg }}>
            {/* Close button */}
            <div style={{
              display: "flex", justifyContent: "space-between", alignItems: "flex-start",
              marginBottom: spacing.lg,
            }}>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <Badge variant={nodeTypeToVariant(selectedNode.type)}>
                    {selectedNode.type}
                  </Badge>
                  <span style={{ fontSize: 16 }}>{NODE_ICONS[selectedNode.type]}</span>
                </div>
                <h3 style={{
                  ...typeScale.h2, color: colors.textPrimary, margin: 0,
                  background: `linear-gradient(135deg, ${colors.textPrimary}, ${selectedNode.color})`,
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}>
                  {selectedNode.label}
                </h3>
                <div style={{
                  ...typeScale.caption, color: colors.textDim, marginTop: 4, fontSize: 8,
                }}>
                  ID: {selectedNode.id}
                </div>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                style={{
                  cursor: "pointer", padding: 6, border: "1px solid rgba(255,255,255,0.06)",
                  background: "rgba(255,255,255,0.03)", borderRadius: 6,
                }}
              >
                <X size={14} color={colors.textDim} />
              </button>
            </div>

            {/* Properties section */}
            <div style={{
              marginBottom: spacing.lg,
              padding: spacing.md,
              background: "rgba(255,255,255,0.02)",
              borderRadius: radius.md,
              border: "1px solid rgba(255,255,255,0.04)",
            }}>
              <div style={{
                ...typeScale.caption, color: colors.textDim, marginBottom: 8,
                display: "flex", alignItems: "center", gap: 6, fontSize: 9,
              }}>
                <Activity size={10} color={colors.textDim} />
                PROPERTIES
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {selectedNode.details && Object.entries(selectedNode.details).map(([key, val]) => (
                  <div key={key} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "6px 8px", borderRadius: radius.sm,
                    background: "rgba(255,255,255,0.02)",
                    borderLeft: `2px solid ${selectedNode.color}15`,
                  }}>
                    <span style={{ ...typeScale.small, color: colors.textDim }}>{key}</span>
                    <span style={{
                      ...typeScale.mono, color: colors.textSecondary, fontSize: 11,
                      background: "rgba(255,255,255,0.03)",
                      padding: "1px 6px", borderRadius: 3,
                    }}>
                      {typeof val === "number" ? (Number.isInteger(val) ? val : val.toFixed(3)) : String(val)}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Quick links */}
            {selectedNode.type === "run" && (
              <button
                onClick={() => navigate("/dashboard")}
                style={{
                  marginBottom: spacing.lg, width: "100%",
                  padding: "10px 14px", borderRadius: radius.md,
                  background: "rgba(0,232,157,0.06)",
                  border: `1px solid rgba(0,232,157,0.15)`,
                  color: colors.accent, fontSize: 12, fontFamily: fonts.sans,
                  cursor: "pointer", display: "flex", alignItems: "center",
                  justifyContent: "center", gap: 8,
                  transition: "all 0.2s ease",
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = "rgba(0,232,157,0.12)";
                  e.currentTarget.style.borderColor = "rgba(0,232,157,0.3)";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = "rgba(0,232,157,0.06)";
                  e.currentTarget.style.borderColor = "rgba(0,232,157,0.15)";
                }}
              >
                <Zap size={14} />
                Open in Dashboard
                <ExternalLink size={12} />
              </button>
            )}

            {/* Connections section */}
            <div style={{
              padding: spacing.md,
              background: "rgba(255,255,255,0.02)",
              borderRadius: radius.md,
              border: "1px solid rgba(255,255,255,0.04)",
            }}>
              <div style={{
                ...typeScale.caption, color: colors.textDim, marginBottom: 8,
                display: "flex", alignItems: "center", gap: 6, fontSize: 9,
              }}>
                <GitBranch size={10} color={colors.textDim} />
                CONNECTIONS
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                {graphData.links
                  .filter((l) => l.source === selectedNode.id || l.target === selectedNode.id)
                  .slice(0, 12)
                  .map((l, i) => {
                    const otherId = l.source === selectedNode.id ? l.target : l.source;
                    const otherNode = graphData.nodes.find((n) => n.id === otherId);
                    return (
                      <div
                        key={i}
                        onClick={() => otherNode && setSelectedNode(otherNode)}
                        style={{
                          display: "flex", alignItems: "center", justifyContent: "space-between",
                          padding: "7px 8px", borderRadius: radius.sm, cursor: "pointer",
                          background: "rgba(255,255,255,0.02)",
                          border: "1px solid transparent",
                          transition: "all 0.15s ease",
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.background = "rgba(255,255,255,0.04)";
                          e.currentTarget.style.borderColor = `${otherNode?.color || colors.border}30`;
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.background = "rgba(255,255,255,0.02)";
                          e.currentTarget.style.borderColor = "transparent";
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                          <div style={{
                            width: 7, height: 7, borderRadius: "50%",
                            background: otherNode?.color || colors.textDim,
                            boxShadow: `0 0 4px ${otherNode?.color || "transparent"}40`,
                          }} />
                          <div>
                            <span style={{ ...typeScale.small, color: colors.textSecondary, display: "block" }}>
                              {otherNode?.label || otherId}
                            </span>
                            {otherNode && (
                              <span style={{ fontSize: 8, color: colors.textMuted, fontFamily: fonts.mono }}>
                                {otherNode.type}
                              </span>
                            )}
                          </div>
                        </div>
                        <span style={{
                          ...typeScale.caption, color: colors.textMuted, fontSize: 7,
                          padding: "1px 5px", borderRadius: 3,
                          background: "rgba(255,255,255,0.03)",
                        }}>
                          {l.label.replace(/_/g, " ")}
                        </span>
                      </div>
                    );
                  })}
              </div>
            </div>

            {/* Warnings for incident/ticket nodes */}
            {(selectedNode.type === "incident" || selectedNode.type === "ticket") && (
              <div style={{
                marginTop: spacing.lg, padding: spacing.md, borderRadius: radius.md,
                background: "rgba(245,158,11,0.05)",
                border: "1px solid rgba(245,158,11,0.15)",
                display: "flex", alignItems: "center", gap: 8,
              }}>
                <AlertTriangle size={14} color="#F59E0B" />
                <span style={{ ...typeScale.small, color: "#F59E0B" }}>
                  {selectedNode.details?.severity === "critical" ? "Critical severity" : "Review recommended"}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Global styles */}
      <style>{`
        @keyframes graphSpin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
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
