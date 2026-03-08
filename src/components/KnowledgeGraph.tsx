/**
 * KnowledgeGraph — Floating top-right panel showing a real-time perception
 * decision graph. Central EGO node connected to AI processing nodes
 * (Perception, Prediction, Planning, Risk) and detected objects. Animated
 * data-flow particles along edges. Color-coded by risk.
 */

import { useRef, useEffect, useState, useCallback } from "react";
import { useStore } from "../store";
import { colors, fonts } from "../theme";
import { BOX_TYPE_COLORS } from "../mockData";
import type { BBox3D } from "../mockData";

const W = 310;
const H = 260;
const CX = W / 2;
const CY = H / 2;

interface GNode {
  id: string;
  label: string;
  x: number;
  y: number;
  r: number;
  color: string;
  ring?: number;
  glow?: boolean;
  type: "ego" | "module" | "object";
}

interface GEdge {
  from: string;
  to: string;
  risk: number; // 0-1
  dashed?: boolean;
}

const AI_MODULES: Omit<GNode, "x" | "y">[] = [
  { id: "percept", label: "Perception", r: 14, color: colors.accentBlue, ring: 55, type: "module" },
  { id: "predict", label: "Prediction", r: 14, color: "#7B6FFF", ring: 55, type: "module" },
  { id: "plan",    label: "Planning",   r: 14, color: colors.accent,     ring: 55, type: "module" },
  { id: "risk",    label: "Risk Eval",  r: 14, color: "#FF9E00",         ring: 55, type: "module" },
];

function computeRisk(box: BBox3D): number {
  const dist = Math.sqrt(box.cx ** 2 + box.cy ** 2);
  const closingSpeed = box.speed;
  if (dist < 5) return 1.0;
  if (closingSpeed > 0) {
    const ttc = dist / closingSpeed;
    if (ttc < 2) return 0.95;
    if (ttc < 5) return 0.6;
    if (ttc < 10) return 0.3;
  }
  if (dist < 15) return 0.4;
  if (dist < 30) return 0.2;
  return 0.05;
}

function riskColor(risk: number): string {
  if (risk > 0.8) return "#FF3333";
  if (risk > 0.5) return "#FF9E00";
  if (risk > 0.2) return "#FFD700";
  return colors.accent;
}

function buildGraph(boxes: BBox3D[]): { nodes: GNode[]; edges: GEdge[] } {
  const nodes: GNode[] = [];
  const edges: GEdge[] = [];

  // Ego
  nodes.push({ id: "ego", label: "EGO", x: CX, y: CY, r: 18, color: colors.accent, glow: true, type: "ego" });

  // AI modules at ring 1
  AI_MODULES.forEach((m, i) => {
    const angle = (i / AI_MODULES.length) * Math.PI * 2 - Math.PI / 2;
    nodes.push({ ...m, x: CX + Math.cos(angle) * m.ring!, y: CY + Math.sin(angle) * m.ring! });
    edges.push({ from: "ego", to: m.id, risk: 0 });
  });

  // Top objects by proximity (max 12 to avoid clutter)
  const sorted = [...boxes]
    .map((b) => ({ box: b, dist: Math.sqrt(b.cx ** 2 + b.cy ** 2), risk: computeRisk(b) }))
    .sort((a, b) => a.dist - b.dist)
    .slice(0, 12);

  sorted.forEach((item, i) => {
    const { box, risk } = item;
    // Position in ring 2 based on the object's actual angle
    const angle = Math.atan2(box.cy, box.cx);
    const ringR = 90 + Math.min(item.dist * 0.8, 30);
    const x = CX + Math.cos(angle) * ringR;
    const y = CY + Math.sin(angle) * ringR;
    const clr = BOX_TYPE_COLORS[box.type] || "#6B7280";
    const nodeId = `obj_${i}`;

    nodes.push({
      id: nodeId,
      label: box.label || box.type,
      x: Math.max(14, Math.min(W - 14, x)),
      y: Math.max(14, Math.min(H - 14, y)),
      r: 8 + risk * 6,
      color: clr,
      type: "object",
    });

    // Connect to perception
    edges.push({ from: "percept", to: nodeId, risk });
    // High-risk: connect to risk module
    if (risk > 0.4) edges.push({ from: "risk", to: nodeId, risk, dashed: true });
    // Moving: connect to prediction
    if (box.speed > 0.5) edges.push({ from: "predict", to: nodeId, risk: risk * 0.5, dashed: true });
  });

  return { nodes, edges };
}

function drawGraph(
  ctx: CanvasRenderingContext2D,
  nodes: GNode[],
  edges: GEdge[],
  t: number,
) {
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = "rgba(8,12,22,0.95)";
  ctx.fillRect(0, 0, W, H);

  // Concentric rings
  for (const r of [55, 95, 120]) {
    ctx.strokeStyle = "rgba(255,255,255,0.03)";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.arc(CX, CY, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  // Edges
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  for (const edge of edges) {
    const from = nodeMap.get(edge.from);
    const to = nodeMap.get(edge.to);
    if (!from || !to) continue;

    const clr = riskColor(edge.risk);
    ctx.strokeStyle = clr + (edge.risk > 0.5 ? "60" : "30");
    ctx.lineWidth = 0.8 + edge.risk;

    if (edge.dashed) ctx.setLineDash([3, 3]);
    else ctx.setLineDash([]);

    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.setLineDash([]);

    // Animated particle along edge
    if (edge.risk > 0.15) {
      const speed = 0.5 + edge.risk;
      const p = ((t * speed) % 1);
      const px = from.x + (to.x - from.x) * p;
      const py = from.y + (to.y - from.y) * p;
      ctx.fillStyle = clr + "90";
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Nodes
  for (const node of nodes) {
    // Glow
    if (node.glow || node.type === "ego") {
      const grd = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, node.r * 2.5);
      grd.addColorStop(0, node.color + "30");
      grd.addColorStop(1, "transparent");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r * 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Body
    ctx.fillStyle = node.color + "25";
    ctx.strokeStyle = node.color + "90";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Inner dot
    ctx.fillStyle = node.color;
    ctx.beginPath();
    ctx.arc(node.x, node.y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Label
    ctx.fillStyle = node.type === "ego" ? colors.textPrimary : colors.textSecondary;
    ctx.font = node.type === "module" ? `bold 7px ${fonts.mono}` : `6px ${fonts.mono}`;
    ctx.textAlign = "center";
    ctx.fillText(node.label.toUpperCase().slice(0, 12), node.x, node.y + node.r + 10);
  }
}

export default function KnowledgeGraph() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [collapsed, setCollapsed] = useState(false);
  const animRef = useRef(0);
  const lastFrame = useRef(-1);
  const graphRef = useRef<{ nodes: GNode[]; edges: GEdge[] }>({ nodes: [], edges: [] });

  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const state = useStore.getState();
    if (state.currentFrameIndex !== lastFrame.current && state.currentFrame) {
      lastFrame.current = state.currentFrameIndex;
      graphRef.current = buildGraph(state.currentFrame.boxes);
    }

    const t = (Date.now() % 4000) / 4000;
    drawGraph(ctx, graphRef.current.nodes, graphRef.current.edges, t);
    animRef.current = requestAnimationFrame(render);
  }, []);

  useEffect(() => {
    if (!collapsed) {
      animRef.current = requestAnimationFrame(render);
      return () => cancelAnimationFrame(animRef.current);
    }
  }, [collapsed, render]);

  return (
    <div
      style={{
        position: "absolute",
        top: 78,
        right: 8,
        zIndex: 10,
        background: "rgba(8,12,22,0.7)",
        backdropFilter: "blur(16px)",
        borderRadius: 8,
        border: "1px solid rgba(255,255,255,0.05)",
        overflow: "hidden",
        userSelect: "none",
      }}
    >
      <div
        onClick={() => setCollapsed((v) => !v)}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "4px 8px",
          cursor: "pointer",
          borderBottom: collapsed ? "none" : "1px solid rgba(255,255,255,0.04)",
        }}
      >
        <span style={{ fontSize: 9, fontWeight: 700, color: colors.textDim, fontFamily: fonts.mono, letterSpacing: "1px" }}>
          🧠 KNOWLEDGE GRAPH
        </span>
        <span style={{ fontSize: 8, color: colors.textDim }}>{collapsed ? "▶" : "▼"}</span>
      </div>

      {!collapsed && (
        <canvas
          ref={canvasRef}
          width={W}
          height={H}
          style={{ width: W, height: H, display: "block" }}
        />
      )}
    </div>
  );
}
