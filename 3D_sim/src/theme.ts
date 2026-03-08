/**
 * Waymo-inspired design tokens — matches the Waymo Perception Studio palette
 */

export const colors = {
  accent: "#00E89D",
  accentDim: "rgba(0, 232, 157, 0.3)",
  accentGlow: "rgba(0, 232, 157, 0.15)",
  accentBlue: "#00C9DB",

  bgDeep: "#0C0F1A",
  bgBase: "#111628",
  bgSurface: "#1A1F35",
  bgOverlay: "#232940",
  bgHover: "#2D3350",

  border: "#2A3050",
  borderSubtle: "#1E2440",

  textPrimary: "#E8ECF4",
  textSecondary: "#8892A8",
  textDim: "#5A6378",

  sensorTop: "#00E89D",
  sensorFront: "#00C9DB",
  sensorSideL: "#4DA8FF",
  sensorSideR: "#7B6FFF",
  sensorRear: "#B490FF",

  boxVehicle: "#FF9E00",
  boxPedestrian: "#CCFF00",
  boxSign: "#FF44FF",
  boxCyclist: "#DC143C",
  boxUnknown: "#6B7280",

  gridMajor: "#2E3550",
  gridMinor: "#252B42",
  vehicleMarker: "#00E89D",

  gizmoX: "#FF5757",
  gizmoY: "#00E89D",
  gizmoZ: "#4DA8FF",
} as const;

export const fonts = {
  sans: "-apple-system, 'system-ui', 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
  mono: "'SF Mono', 'Cascadia Code', 'JetBrains Mono', Consolas, monospace",
} as const;

export const radius = {
  sm: "4px",
  md: "8px",
  lg: "12px",
  pill: "999px",
} as const;

export const gradients = {
  accent: `linear-gradient(90deg, #00E89D, #00C9DB)`,
  surface: `linear-gradient(180deg, #1A1F35 0%, #111628 100%)`,
} as const;
