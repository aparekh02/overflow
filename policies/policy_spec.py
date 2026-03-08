"""
Policy data input specifications — formal contracts for observation, action, and ticket data.

This module defines the exact data shapes, normalization ranges, and semantic meaning
of every field consumed by OpenENV policies. Use this as the reference when:

  1. Building a new environment that targets these policies
  2. Writing a bridge/adapter from a different simulator
  3. Implementing a new policy that must interoperate with the existing set

All policies share the same raw observation layout (EGO + ticket matrix).
Specialized policies (ThreatAvoidance, SystemFailure) select subsets internally.

Example usage:
    from openenv.policies.policy_spec import ObsSpec, ActionSpec, validate_obs

    spec = ObsSpec()
    obs  = my_env.get_observation()
    validate_obs(obs, spec)  # raises ValueError on shape/range mismatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Ego state specification ──────────────────────────────────────────────────

EGO_STATE_DIM = 11

@dataclass(frozen=True)
class EgoField:
    """Description of a single ego state field."""
    index:       int
    name:        str
    unit:        str
    raw_range:   Tuple[float, float]   # physical range before normalization
    norm_divisor: float                # obs_value = raw_value / norm_divisor
    description: str

EGO_FIELDS: List[EgoField] = [
    EgoField(0,  "x",             "m",     (-5000, 5000),  1000.0,  "Forward displacement from episode start"),
    EgoField(1,  "y",             "m",     (-6.0, 6.0),    3.7,     "Lateral displacement (0 = lane center, + = left)"),
    EgoField(2,  "z",             "m",     (-10, 10),      10.0,    "Vertical position (flat road = 0)"),
    EgoField(3,  "vx",            "m/s",   (-20, 20),      20.0,    "Forward velocity in world frame"),
    EgoField(4,  "vy",            "m/s",   (-20, 20),      20.0,    "Lateral velocity in world frame"),
    EgoField(5,  "vz",            "m/s",   (0, 0),         1.0,     "Vertical velocity (always 0 on flat road)"),
    EgoField(6,  "heading_sin",   "rad",   (-1, 1),        1.0,     "sin(heading angle), 0 = forward"),
    EgoField(7,  "heading_cos",   "rad",   (-1, 1),        1.0,     "cos(heading angle), 1 = forward"),
    EgoField(8,  "speed",         "m/s",   (0, 20),        20.0,    "Scalar speed = sqrt(vx^2 + vy^2)"),
    EgoField(9,  "steer",         "norm",  (-1, 1),        1.0,     "Current steering command [-1=full left, 1=full right]"),
    EgoField(10, "net_drive",     "norm",  (-1, 1),        1.0,     "throttle - brake [-1=full brake, 1=full throttle]"),
]


# ── Ticket vector specification ──────────────────────────────────────────────

TICKET_VECTOR_DIM = 37   # 18 fixed + 14 type one-hot + 5 entity one-hot
MAX_TICKETS = 16

# Ticket types (14 total) — one-hot encoded starting at index 18
TICKET_TYPES = [
    "collision_risk", "sudden_brake", "side_impact", "head_on",
    "merge_cut", "rear_end_risk",
    "pedestrian_crossing", "cyclist_lane",
    "tire_blowout", "brake_fade", "steering_loss", "sensor_occlusion",
    "road_hazard", "weather_visibility",
]

# Entity types (5 total) — one-hot encoded after ticket types
ENTITY_TYPES = ["vehicle", "pedestrian", "cyclist", "obstacle", "system"]

# Verify dimension
assert 18 + len(TICKET_TYPES) + len(ENTITY_TYPES) == TICKET_VECTOR_DIM, (
    f"Ticket vector dim mismatch: 18 + {len(TICKET_TYPES)} + {len(ENTITY_TYPES)} "
    f"!= {TICKET_VECTOR_DIM}"
)

@dataclass(frozen=True)
class TicketField:
    """Description of a single ticket vector field."""
    offset:       int            # index within the TICKET_VECTOR_DIM vector
    length:       int            # number of floats
    name:         str
    unit:         str
    raw_range:    Tuple[float, float]
    norm_divisor: float
    description:  str

TICKET_FIELDS: List[TicketField] = [
    TicketField(0,  1, "severity_weight",   "norm",   (0, 1),       1.0,   "Severity: 0.25=LOW, 0.5=MED, 0.75=HIGH, 1.0=CRITICAL"),
    TicketField(1,  1, "ttl_norm",          "s",      (0, 10),     10.0,   "Time-to-live remaining, clamped to [0,1]"),
    TicketField(2,  1, "pos_x",             "m",      (-100, 100), 100.0,  "Ego-relative X (forward positive)"),
    TicketField(3,  1, "pos_y",             "m",      (-50, 50),    50.0,  "Ego-relative Y (left positive)"),
    TicketField(4,  1, "pos_z",             "m",      (-10, 10),    10.0,  "Ego-relative Z (up positive)"),
    TicketField(5,  1, "vel_x",             "m/s",    (-30, 30),    30.0,  "Entity velocity X in world frame"),
    TicketField(6,  1, "vel_y",             "m/s",    (-30, 30),    30.0,  "Entity velocity Y in world frame"),
    TicketField(7,  1, "vel_z",             "m/s",    (-10, 10),    10.0,  "Entity velocity Z in world frame"),
    TicketField(8,  1, "heading_sin",       "rad",    (-1, 1),      1.0,   "sin(entity heading relative to ego)"),
    TicketField(9,  1, "heading_cos",       "rad",    (-1, 1),      1.0,   "cos(entity heading relative to ego)"),
    TicketField(10, 1, "size_length",       "m",      (0, 10),     10.0,   "Entity bounding box length"),
    TicketField(11, 1, "size_width",        "m",      (0, 5),       5.0,   "Entity bounding box width"),
    TicketField(12, 1, "size_height",       "m",      (0, 4),       4.0,   "Entity bounding box height"),
    TicketField(13, 1, "distance_norm",     "m",      (0, 100),   100.0,   "Euclidean distance to ego, clamped to [0,1]"),
    TicketField(14, 1, "ttc_norm",          "s",      (0, 30),     30.0,   "Time-to-collision, clamped to [0,1]. 1.0 = no collision"),
    TicketField(15, 1, "bearing_sin",       "rad",    (-1, 1),      1.0,   "sin(bearing angle from ego forward axis)"),
    TicketField(16, 1, "bearing_cos",       "rad",    (-1, 1),      1.0,   "cos(bearing angle from ego forward axis)"),
    TicketField(17, 1, "confidence",        "norm",   (0, 1),       1.0,   "Perception confidence [0=unreliable, 1=certain]"),
    TicketField(18, len(TICKET_TYPES), "type_onehot",   "bool", (0, 1), 1.0, "One-hot ticket type"),
    TicketField(18 + len(TICKET_TYPES), len(ENTITY_TYPES), "entity_onehot", "bool", (0, 1), 1.0, "One-hot entity type"),
]


# ── Full observation specification ───────────────────────────────────────────

OBS_DIM = EGO_STATE_DIM + MAX_TICKETS * TICKET_VECTOR_DIM   # 11 + 16*37 = 603

@dataclass(frozen=True)
class ObsSpec:
    """Complete observation space specification."""
    ego_dim:         int = EGO_STATE_DIM
    ticket_dim:      int = TICKET_VECTOR_DIM
    max_tickets:     int = MAX_TICKETS
    total_dim:       int = OBS_DIM
    dtype:           str = "float32"
    value_range:     Tuple[float, float] = (-1.0, 1.0)

    # Layout: obs[0:ego_dim] = ego state
    #         obs[ego_dim:] reshaped to (max_tickets, ticket_dim)
    # Tickets are sorted by severity desc, distance asc. Zero-padded rows = empty slots.


# ── Action specification ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ActionField:
    index:       int
    name:        str
    raw_range:   Tuple[float, float]
    description: str

ACTION_DIM = 3

ACTION_FIELDS: List[ActionField] = [
    ActionField(0, "steer",    (-1.0, 1.0), "Steering command. -1=full left, +1=full right. Scaled by MAX_STEER=0.6 rad"),
    ActionField(1, "throttle", (-1.0, 1.0), "Throttle command. Only positive values used (clipped to [0,1]). Scaled by MAX_ACCEL=4.0 m/s^2"),
    ActionField(2, "brake",    (-1.0, 1.0), "Brake command. Only positive values used (clipped to [0,1]). Scaled by MAX_BRAKE=8.0 m/s^2"),
]

@dataclass(frozen=True)
class ActionSpec:
    """Action space specification."""
    dim:          int = ACTION_DIM
    dtype:        str = "float32"
    value_range:  Tuple[float, float] = (-1.0, 1.0)


# ── Policy input requirements ────────────────────────────────────────────────

@dataclass(frozen=True)
class PolicyInputSpec:
    """Describes what a specific policy reads from the observation."""
    name:              str
    reads_ego:         bool
    ego_indices:       Tuple[int, ...]           # which ego fields are used
    reads_tickets:     bool
    ticket_filter:     Optional[str]             # None = all, or "kinematic" / "failure"
    max_tickets_used:  int                       # how many ticket slots the policy actually reads
    requires_history:  bool                      # whether GRU/recurrent hidden state is needed
    description:       str

POLICY_SPECS: Dict[str, PolicyInputSpec] = {
    "SurvivalPolicy": PolicyInputSpec(
        name="SurvivalPolicy",
        reads_ego=True,
        ego_indices=tuple(range(EGO_STATE_DIM)),
        reads_tickets=False,
        ticket_filter=None,
        max_tickets_used=0,
        requires_history=False,
        description="Stage 1 baseline. Reads only ego state (first 11 dims). "
                    "Ticket portion of obs is ignored entirely.",
    ),
    "FlatMLPPolicy": PolicyInputSpec(
        name="FlatMLPPolicy",
        reads_ego=True,
        ego_indices=tuple(range(EGO_STATE_DIM)),
        reads_tickets=True,
        ticket_filter=None,
        max_tickets_used=MAX_TICKETS,
        requires_history=False,
        description="Sanity-check baseline. Reads full flat observation (ego + all tickets "
                    "concatenated). No attention or structure.",
    ),
    "TicketAttentionPolicy": PolicyInputSpec(
        name="TicketAttentionPolicy",
        reads_ego=True,
        ego_indices=tuple(range(EGO_STATE_DIM)),
        reads_tickets=True,
        ticket_filter=None,
        max_tickets_used=MAX_TICKETS,
        requires_history=False,
        description="Main policy (Stage 2+). Cross-attention: ego queries ticket set. "
                    "Order-invariant over tickets. Padding mask on zero-rows.",
    ),
    "ThreatAvoidancePolicy": PolicyInputSpec(
        name="ThreatAvoidancePolicy",
        reads_ego=True,
        ego_indices=tuple(range(EGO_STATE_DIM)),
        reads_tickets=True,
        ticket_filter="kinematic",
        max_tickets_used=1,
        requires_history=False,
        description="Specialist for kinematic threats (collision_risk, sudden_brake, "
                    "side_impact, head_on, merge_cut, rear_end_risk). Extracts the "
                    "highest-severity kinematic ticket and gates between brake/evade branches.",
    ),
    "SystemFailurePolicy": PolicyInputSpec(
        name="SystemFailurePolicy",
        reads_ego=True,
        ego_indices=tuple(range(EGO_STATE_DIM)),
        reads_tickets=True,
        ticket_filter="failure",
        max_tickets_used=1,
        requires_history=False,
        description="Specialist for onboard failures (tire_blowout, brake_fade, steering_loss). "
                    "Mixture-of-experts with one expert per failure type. Initialized with "
                    "domain-correct response priors.",
    ),
    "RecurrentPolicy": PolicyInputSpec(
        name="RecurrentPolicy",
        reads_ego=True,
        ego_indices=tuple(range(EGO_STATE_DIM)),
        reads_tickets=True,
        ticket_filter=None,
        max_tickets_used=MAX_TICKETS,
        requires_history=True,
        description="GRU-based policy for partial observability (Stage 4+). Carries hidden "
                    "state across timesteps. Requires h_prev to be tracked by caller.",
    ),
}


# ── Validation helpers ───────────────────────────────────────────────────────

def validate_obs(obs: np.ndarray, spec: Optional[ObsSpec] = None) -> None:
    """
    Validate an observation array against the spec.
    Raises ValueError with a descriptive message on any mismatch.
    """
    spec = spec or ObsSpec()
    if obs.ndim != 1:
        raise ValueError(f"Observation must be 1D, got shape {obs.shape}")
    if obs.shape[0] != spec.total_dim:
        raise ValueError(
            f"Observation dim mismatch: expected {spec.total_dim}, got {obs.shape[0]}. "
            f"Check ego_dim ({spec.ego_dim}) + max_tickets ({spec.max_tickets}) "
            f"* ticket_dim ({spec.ticket_dim})"
        )
    if obs.dtype != np.float32:
        raise ValueError(f"Observation dtype must be float32, got {obs.dtype}")


def validate_action(action: np.ndarray) -> None:
    """Validate an action array."""
    if action.shape != (ACTION_DIM,):
        raise ValueError(f"Action shape mismatch: expected ({ACTION_DIM},), got {action.shape}")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ValueError(f"Action values must be in [-1, 1], got min={action.min()}, max={action.max()}")


def build_obs(
    ego_x: float, ego_y: float, ego_z: float,
    ego_vx: float, ego_vy: float,
    heading: float, speed: float,
    steer: float, throttle: float, brake: float,
    ticket_vectors: Optional[np.ndarray] = None,
    max_tickets: int = MAX_TICKETS,
) -> np.ndarray:
    """
    Build a valid observation vector from raw values.

    This is the primary entry point for external environments that want to
    produce observations compatible with OpenENV policies.

    Parameters
    ----------
    ego_x       : forward displacement from episode start (metres)
    ego_y       : lateral displacement from lane center (metres, + = left)
    ego_z       : vertical position (metres)
    ego_vx      : forward velocity (m/s)
    ego_vy      : lateral velocity (m/s)
    heading     : heading angle (radians, 0 = forward)
    speed       : scalar speed (m/s)
    steer       : current steering command [-1, 1]
    throttle    : current throttle command [0, 1]
    brake       : current brake command [0, 1]
    ticket_vectors : (N, TICKET_VECTOR_DIM) array of ticket vectors, or None.
                     Use EventTicket.to_vector() or build_ticket_vector() to create these.
    max_tickets : number of ticket slots (must match policy expectation, default 16)

    Returns
    -------
    obs : np.ndarray of shape (EGO_STATE_DIM + max_tickets * TICKET_VECTOR_DIM,)
    """
    import math

    ego = np.array([
        ego_x / 1000.0,
        ego_y / 3.7,        # ROAD_HALF_WIDTH
        ego_z / 10.0,
        ego_vx / 20.0,      # MAX_SPEED
        ego_vy / 20.0,
        0.0,                 # vz (flat road)
        math.sin(heading),
        math.cos(heading),
        speed / 20.0,
        steer,
        throttle - brake,    # net drive signal
    ], dtype=np.float32)

    ticket_matrix = np.zeros((max_tickets, TICKET_VECTOR_DIM), dtype=np.float32)
    if ticket_vectors is not None:
        n = min(len(ticket_vectors), max_tickets)
        ticket_matrix[:n] = ticket_vectors[:n]

    return np.concatenate([ego, ticket_matrix.flatten()])


def build_ticket_vector(
    severity_weight: float,
    ttl: float,
    pos_x: float, pos_y: float, pos_z: float,
    vel_x: float, vel_y: float, vel_z: float,
    heading: float,
    size_length: float, size_width: float, size_height: float,
    distance: float,
    time_to_collision: Optional[float],
    bearing: float,
    ticket_type: str,
    entity_type: str,
    confidence: float = 1.0,
) -> np.ndarray:
    """
    Build a single ticket vector from raw values without needing the full
    EventTicket class. Use this when adapting a different simulator.

    Parameters
    ----------
    severity_weight    : 0.25 (LOW), 0.5 (MEDIUM), 0.75 (HIGH), 1.0 (CRITICAL)
    ttl                : seconds remaining until ticket expires
    pos_x/y/z          : ego-relative position (metres)
    vel_x/y/z          : entity velocity in world frame (m/s)
    heading            : entity heading relative to ego (radians)
    size_length/width/height : entity bounding box (metres)
    distance           : euclidean distance to ego (metres)
    time_to_collision  : seconds until collision, or None if no collision course
    bearing            : angle from ego forward axis (radians)
    ticket_type        : one of TICKET_TYPES (e.g., "collision_risk")
    entity_type        : one of ENTITY_TYPES (e.g., "vehicle")
    confidence         : perception confidence [0, 1]

    Returns
    -------
    vec : np.ndarray of shape (TICKET_VECTOR_DIM,) = (37,)
    """
    import math

    ttc_norm = min((time_to_collision if time_to_collision is not None else 30.0) / 30.0, 1.0)

    type_oh = [0.0] * len(TICKET_TYPES)
    entity_oh = [0.0] * len(ENTITY_TYPES)

    if ticket_type in TICKET_TYPES:
        type_oh[TICKET_TYPES.index(ticket_type)] = 1.0
    else:
        raise ValueError(f"Unknown ticket_type '{ticket_type}'. Must be one of {TICKET_TYPES}")

    if entity_type in ENTITY_TYPES:
        entity_oh[ENTITY_TYPES.index(entity_type)] = 1.0
    else:
        raise ValueError(f"Unknown entity_type '{entity_type}'. Must be one of {ENTITY_TYPES}")

    vec = [
        severity_weight,
        min(ttl / 10.0, 1.0),
        pos_x / 100.0,
        pos_y / 50.0,
        pos_z / 10.0,
        vel_x / 30.0,
        vel_y / 30.0,
        vel_z / 10.0,
        math.sin(heading),
        math.cos(heading),
        size_length / 10.0,
        size_width / 5.0,
        size_height / 4.0,
        min(distance / 100.0, 1.0),
        ttc_norm,
        math.sin(bearing),
        math.cos(bearing),
        confidence,
        *type_oh,
        *entity_oh,
    ]
    return np.array(vec, dtype=np.float32)
