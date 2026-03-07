/**
 * Mock data generator — creates realistic self-driving perception data mimicking
 * Waymo Open Dataset style:
 *   • LiDAR point cloud with radial scan-line ring pattern (64-beam spinner)
 *   • 3D bounding boxes for vehicles, pedestrians, cyclists, signs
 *   • Per-frame poses (timeline animation at 10 Hz)
 */

import * as THREE from "three";

// ── Types ──────────────────────────────────────────────────────────

export type ActorType = "vehicle" | "pedestrian" | "cyclist" | "sign";

export interface BBox3D {
  id: string;
  type: ActorType;
  cx: number; cy: number; cz: number;       // centre
  sx: number; sy: number; sz: number;       // size (length, width, height)
  heading: number;                          // yaw radians
  speed: number;
  label: string;
  trackId: number;
}

export interface FrameData {
  timestamp: number;
  egoPosition: [number, number, number];
  egoYaw: number;
  boxes: BBox3D[];
  pointPositions: Float32Array;   // xyz interleaved (3 floats per pt)
  pointAttributes: Float32Array;  // intensity, range, elongation (3 floats per pt)
  pointCount: number;
}

export interface SceneData {
  frames: FrameData[];
  fps: number;
  totalSeconds: number;
  totalFrames: number;
}

// ── Constants ──────────────────────────────────────────────────────

const NUM_FRAMES = 198;
const FPS = 10;

const LIDAR_BEAMS = 64;           // vertical beams
const LIDAR_COLUMNS = 2650;       // horizontal resolution per spin
const MAX_RANGE = 75;             // metres
const BEAM_V_MIN = -25;           // degrees — lowest beam
const BEAM_V_MAX = 2;             // degrees — highest beam

// ── Helpers ────────────────────────────────────────────────────────

function rand(lo: number, hi: number) { return lo + Math.random() * (hi - lo); }

function gaussRand() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ── Realistic LiDAR generation with scan-line rings ────────────────

interface SimpleOccupancy {
  boxes: BBox3D[];
}

/**
 * Trace a single LiDAR ray and return hit point, or null if no return.
 * Checks ground plane + box occupancies.
 */
function traceRay(
  azimuth: number,    // horizontal angle rad
  elevation: number,  // vertical angle rad
  occ: SimpleOccupancy,
): { x: number; y: number; z: number; intensity: number; range: number; elongation: number } | null {
  const cosE = Math.cos(elevation);
  const sinE = Math.sin(elevation);
  const cosA = Math.cos(azimuth);
  const sinA = Math.sin(azimuth);

  // Ray direction (Waymo: X=forward, Y=left, Z=up)
  const dx = cosE * cosA;
  const dy = cosE * sinA;
  const dz = sinE;

  let bestT = MAX_RANGE;
  let hitType: "ground" | "object" | "none" = "none";
  let hitBox: BBox3D | null = null;

  // Ground hit (z = 0 plane, sensor at ~2.0m height)
  const sensorZ = 2.0;
  if (dz < -0.001) {
    const tGround = -sensorZ / dz;
    const gx = dx * tGround;
    const gy = dy * tGround;
    const gr = Math.sqrt(gx * gx + gy * gy);
    if (gr < MAX_RANGE && tGround < bestT) {
      bestT = tGround;
      hitType = "ground";
    }
  }

  // Box hits (simple AABB after inverse rotation)
  for (const box of occ.boxes) {
    const cosH = Math.cos(-box.heading);
    const sinH = Math.sin(-box.heading);
    // Transform ray origin to box-local frame
    const ox = -box.cx;
    const oy = -box.cy;
    const oz = sensorZ - box.cz;
    const lox = ox * cosH - oy * sinH;
    const loy = ox * sinH + oy * cosH;
    const ldx = dx * cosH - dy * sinH;
    const ldy = dx * sinH + dy * cosH;

    const halfX = box.sx / 2;
    const halfY = box.sy / 2;
    const halfZ = box.sz / 2;

    // Slab intersection
    let tMin = -1e9, tMax = 1e9;
    // X slab
    if (Math.abs(ldx) > 1e-8) {
      const t1 = (-halfX - lox) / ldx;
      const t2 = (halfX - lox) / ldx;
      tMin = Math.max(tMin, Math.min(t1, t2));
      tMax = Math.min(tMax, Math.max(t1, t2));
    } else if (Math.abs(lox) > halfX) continue;
    // Y slab
    if (Math.abs(ldy) > 1e-8) {
      const t1 = (-halfY - loy) / ldy;
      const t2 = (halfY - loy) / ldy;
      tMin = Math.max(tMin, Math.min(t1, t2));
      tMax = Math.min(tMax, Math.max(t1, t2));
    } else if (Math.abs(loy) > halfY) continue;
    // Z slab
    if (Math.abs(dz) > 1e-8) {
      const t1 = (-halfZ - oz) / dz;
      const t2 = (halfZ - oz) / dz;
      tMin = Math.max(tMin, Math.min(t1, t2));
      tMax = Math.min(tMax, Math.max(t1, t2));
    } else if (Math.abs(oz) > halfZ) continue;

    if (tMin <= tMax && tMax > 0) {
      const tHit = tMin > 0 ? tMin : tMax;
      if (tHit < bestT && tHit > 0.5) {
        bestT = tHit;
        hitType = "object";
        hitBox = box;
      }
    }
  }

  if (hitType === "none") return null;

  const px = dx * bestT;
  const py = dy * bestT;
  const pz = sensorZ + dz * bestT;
  const range = bestT;

  // Intensity: ground=low, objects=high with some variation
  let intensity: number;
  let elongation: number;
  if (hitType === "ground") {
    intensity = 0.15 + 0.25 * (1 - range / MAX_RANGE) + gaussRand() * 0.05;
    elongation = rand(0.0, 0.1);
  } else {
    intensity = 0.5 + rand(0, 0.5);
    elongation = rand(0.0, 0.3);
  }
  intensity = Math.max(0, Math.min(1, intensity));

  // Add tiny noise to simulate real sensor
  return {
    x: px + gaussRand() * 0.02,
    y: py + gaussRand() * 0.02,
    z: pz + gaussRand() * 0.01,
    intensity,
    range,
    elongation: Math.max(0, Math.min(1, elongation)),
  };
}

function generateLidarFrame(boxes: BBox3D[]): {
  positions: Float32Array;
  attributes: Float32Array;
  count: number;
} {
  const occ: SimpleOccupancy = { boxes };
  const maxPts = LIDAR_BEAMS * LIDAR_COLUMNS;
  const positions = new Float32Array(maxPts * 3);
  const attributes = new Float32Array(maxPts * 3);
  let count = 0;

  // Subsample columns for performance (every ~4th → ~42K points)
  const colStep = 4;
  for (let beam = 0; beam < LIDAR_BEAMS; beam++) {
    const elevDeg = BEAM_V_MIN + (BEAM_V_MAX - BEAM_V_MIN) * (beam / (LIDAR_BEAMS - 1));
    const elevRad = elevDeg * (Math.PI / 180);

    for (let col = 0; col < LIDAR_COLUMNS; col += colStep) {
      const azimuthRad = (col / LIDAR_COLUMNS) * Math.PI * 2 - Math.PI;

      // Drop-out probability (realistic: ~5% of rays miss)
      if (Math.random() < 0.05) continue;

      const hit = traceRay(azimuthRad, elevRad, occ);
      if (!hit) continue;

      const i3 = count * 3;
      positions[i3] = hit.x;
      positions[i3 + 1] = hit.y;
      positions[i3 + 2] = hit.z;
      attributes[i3] = hit.intensity;
      attributes[i3 + 1] = hit.range;
      attributes[i3 + 2] = hit.elongation;
      count++;
    }
  }

  return {
    positions: positions.subarray(0, count * 3),
    attributes: attributes.subarray(0, count * 3),
    count,
  };
}

// ── Actor trajectories ─────────────────────────────────────────────

interface ActorDef {
  id: string;
  type: ActorType;
  size: [number, number, number];  // sx, sy, sz (length, width, height)
  label: string;
  trackId: number;
  trajectory: (t: number) => { x: number; y: number; heading: number; speed: number };
}

function makeActors(): ActorDef[] {
  const actors: ActorDef[] = [];

  // ── Moving vehicles ──
  actors.push({
    id: "v1", type: "vehicle", size: [4.8, 2.1, 1.8], label: "Car", trackId: 1,
    trajectory: (t) => ({ x: 12 + t * 6.5, y: -1.8, heading: 0, speed: 6.5 }),
  });
  actors.push({
    id: "v2", type: "vehicle", size: [4.5, 2.0, 1.6], label: "Car", trackId: 2,
    trajectory: (t) => ({ x: 25 + t * 5, y: -1.5, heading: 0, speed: 5 }),
  });
  actors.push({
    id: "v3", type: "vehicle", size: [5.2, 2.2, 2.0], label: "SUV", trackId: 3,
    trajectory: (t) => ({ x: -8 + t * 8.5, y: 2.0, heading: 0, speed: 8.5 }),
  });
  // Oncoming
  actors.push({
    id: "v4", type: "vehicle", size: [4.6, 2.0, 1.7], label: "Car", trackId: 4,
    trajectory: (t) => ({ x: 60 - t * 9, y: 5.5, heading: Math.PI, speed: 9 }),
  });
  // Turning at intersection
  actors.push({
    id: "v5", type: "vehicle", size: [4.4, 1.9, 1.5], label: "Car", trackId: 5,
    trajectory: (t) => {
      const tr = 5;
      if (t < tr) return { x: 40, y: -20 + t * 7, heading: Math.PI / 2, speed: 7 };
      const dt = t - tr;
      const angle = Math.PI / 2 - dt * 0.25;
      const r = 12;
      return {
        x: 40 + r * (1 - Math.cos(Math.PI / 2 - angle)),
        y: -20 + tr * 7 + r * Math.sin(Math.PI / 2 - angle),
        heading: angle, speed: 6,
      };
    },
  });
  // Behind ego
  actors.push({
    id: "v9", type: "vehicle", size: [4.6, 2.0, 1.7], label: "Car", trackId: 9,
    trajectory: (t) => ({ x: -15 + t * 7, y: -5.5, heading: 0, speed: 7 }),
  });
  actors.push({
    id: "v10", type: "vehicle", size: [4.3, 1.9, 1.5], label: "Car", trackId: 10,
    trajectory: (t) => ({ x: 50 + t * 4, y: -1.8, heading: 0, speed: 4 }),
  });

  // ── Parked vehicles ──
  actors.push({
    id: "v6", type: "vehicle", size: [4.8, 2.1, 1.8], label: "Parked", trackId: 6,
    trajectory: () => ({ x: 18, y: 8.5, heading: 0, speed: 0 }),
  });
  actors.push({
    id: "v7", type: "vehicle", size: [4.5, 2.0, 1.6], label: "Parked", trackId: 7,
    trajectory: () => ({ x: 30, y: 8.5, heading: 0.05, speed: 0 }),
  });
  actors.push({
    id: "v8", type: "vehicle", size: [5.0, 2.2, 2.2], label: "Truck", trackId: 8,
    trajectory: () => ({ x: 45, y: 8.5, heading: 0.0, speed: 0 }),
  });
  actors.push({
    id: "v11", type: "vehicle", size: [4.6, 2.0, 1.7], label: "Parked", trackId: 11,
    trajectory: () => ({ x: 55, y: -8.5, heading: Math.PI, speed: 0 }),
  });
  actors.push({
    id: "v12", type: "vehicle", size: [4.8, 2.1, 1.8], label: "Parked", trackId: 12,
    trajectory: () => ({ x: 65, y: -8.5, heading: Math.PI, speed: 0 }),
  });

  // ── Pedestrians ──
  actors.push({
    id: "p1", type: "pedestrian", size: [0.8, 0.8, 1.8], label: "Ped", trackId: 20,
    trajectory: (t) => {
      const start = 2;
      if (t < start) return { x: 20, y: -6, heading: Math.PI / 2, speed: 0 };
      const dt = t - start;
      return { x: 20, y: -6 + dt * 1.3, heading: Math.PI / 2, speed: 1.3 };
    },
  });
  actors.push({
    id: "p2", type: "pedestrian", size: [0.7, 0.7, 1.7], label: "Ped", trackId: 21,
    trajectory: (t) => ({ x: 35 - t * 0.8, y: -8, heading: Math.PI, speed: 0.8 }),
  });
  actors.push({
    id: "p3", type: "pedestrian", size: [0.8, 0.8, 1.75], label: "Ped", trackId: 22,
    trajectory: () => ({ x: 15, y: 6.5, heading: 0, speed: 0 }),
  });
  actors.push({
    id: "p4", type: "pedestrian", size: [0.7, 0.7, 1.65], label: "Ped", trackId: 23,
    trajectory: (t) => ({ x: 28, y: 10 - t * 1.0, heading: -Math.PI / 2, speed: 1.0 }),
  });
  actors.push({
    id: "p5", type: "pedestrian", size: [0.8, 0.8, 1.8], label: "Ped", trackId: 24,
    trajectory: () => ({ x: 42, y: -7.5, heading: 0, speed: 0 }),
  });
  actors.push({
    id: "p6", type: "pedestrian", size: [0.75, 0.75, 1.7], label: "Ped", trackId: 25,
    trajectory: (t) => {
      const start = 8;
      if (t < start) return { x: 50, y: 7, heading: -Math.PI / 2, speed: 0 };
      return { x: 50, y: 7 - (t - start) * 1.2, heading: -Math.PI / 2, speed: 1.2 };
    },
  });

  // ── Cyclists ──
  actors.push({
    id: "c1", type: "cyclist", size: [1.8, 0.7, 1.7], label: "Cyclist", trackId: 30,
    trajectory: (t) => ({ x: 5 + t * 4.5, y: 3.5, heading: 0, speed: 4.5 }),
  });
  actors.push({
    id: "c2", type: "cyclist", size: [1.8, 0.7, 1.7], label: "Cyclist", trackId: 31,
    trajectory: (t) => ({ x: 55 - t * 3.5, y: -3.5, heading: Math.PI, speed: 3.5 }),
  });

  // ── Signs ──
  actors.push({
    id: "s1", type: "sign", size: [0.1, 0.8, 1.2], label: "Sign", trackId: 40,
    trajectory: () => ({ x: 25, y: 10, heading: 0, speed: 0 }),
  });
  actors.push({
    id: "s2", type: "sign", size: [0.1, 0.6, 0.8], label: "Sign", trackId: 41,
    trajectory: () => ({ x: 48, y: -10, heading: Math.PI / 2, speed: 0 }),
  });

  return actors;
}

// ── Public API ─────────────────────────────────────────────────────

let cachedScene: SceneData | null = null;

export function generateSceneData(): SceneData {
  if (cachedScene) return cachedScene;

  const actors = makeActors();
  const totalSeconds = NUM_FRAMES / FPS;
  const frames: FrameData[] = [];

  for (let fi = 0; fi < NUM_FRAMES; fi++) {
    const t = fi / FPS;

    // Ego drives forward
    const egoX = t * 8;
    const egoY = 0;
    const egoYaw = 0;

    const boxes: BBox3D[] = [];
    for (const actor of actors) {
      const pose = actor.trajectory(t);
      const relX = pose.x - egoX;
      const relY = pose.y - egoY;
      if (Math.abs(relX) > MAX_RANGE || Math.abs(relY) > MAX_RANGE) continue;

      boxes.push({
        id: actor.id,
        type: actor.type,
        cx: relX,
        cy: relY,
        cz: actor.size[2] / 2,  // bottom at ground
        sx: actor.size[0],
        sy: actor.size[1],
        sz: actor.size[2],
        heading: pose.heading - egoYaw,
        speed: pose.speed,
        label: actor.label,
        trackId: actor.trackId,
      });
    }

    // Generate realistic LiDAR with ray-tracing
    const lidar = generateLidarFrame(boxes);

    frames.push({
      timestamp: t,
      egoPosition: [egoX, egoY, 0],
      egoYaw,
      boxes,
      pointPositions: lidar.positions,
      pointAttributes: lidar.attributes,
      pointCount: lidar.count,
    });
  }

  cachedScene = { frames, fps: FPS, totalSeconds, totalFrames: NUM_FRAMES };
  return cachedScene;
}

// ── Color constants (matching Waymo Perception Studio theme) ──────

export const BOX_TYPE_COLORS: Record<ActorType, string> = {
  vehicle: "#FF9E00",     // orange
  pedestrian: "#CCFF00",  // lemon-lime
  cyclist: "#DC143C",     // crimson
  sign: "#FF44FF",        // magenta
};

export const HIGHLIGHT_COLOR = "#00E5FF";

export function actorColor(type: ActorType): string {
  return BOX_TYPE_COLORS[type] ?? "#6B7280";
}
