import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import simulate_gbm_paths
from parameters import BARRIERS, S0, T, mu_star, n_steps, seed, sigma

# Styling constants to keep the plot readable and consistent.
BACKGROUND_COLOR = "rgba(0,0,0,0.10)"
TOUCHED_BELOW_COLOR = "#858585"  # dark gray instead of black
UNTOUCHED_COLOR = "#55ef8b"
TOUCHED_ABOVE_COLOR = "#C66262"
BARRIER_COLOR = "#14509d"  # dark blue
BARRIER_WIDTH = 3  # thicker line for better visibility
MAIN_LINE_WIDTH = 1.6
BACKGROUND_LINE_WIDTH = 1


def _split_path_segments_indices(
    path: np.ndarray, barrier: float
) -> List[Tuple[bool, int, int]]:
    """
    Split a trajectory into contiguous below/above segments using index ranges.
    Ensures each segment has at least two points for visibility.
    """
    above_mask = path >= barrier
    segments: List[Tuple[bool, int, int]] = []

    start_idx = 0
    current_state = above_mask[0]

    for idx in range(1, len(path)):
        if above_mask[idx] == current_state:
            continue

        end_idx = idx + 1
        if end_idx - start_idx == 1 and start_idx > 0:
            start_idx -= 1
        segments.append((current_state, start_idx, end_idx))

        start_idx = idx
        current_state = above_mask[idx]

    end_idx = len(path)
    if end_idx - start_idx == 1 and start_idx > 0:
        start_idx -= 1
    segments.append((current_state, start_idx, end_idx))

    return segments


def _build_barrier_traces(
    time_grid: np.ndarray,
    S_paths_main: np.ndarray,
    barrier: float,
    R_paths_main: int,
) -> Tuple[List[go.Scatter], List[Tuple[int, int] | None], List[Optional[int]]]:
    traces: List[go.Scatter] = []
    trace_ranges: List[Tuple[int, int] | None] = []
    trace_parents: List[Optional[int]] = []
    untouched_legend_shown = False
    below_legend_shown = False
    above_legend_shown = False

    for i in range(R_paths_main):
        path = S_paths_main[i]
        touched = (path >= barrier).any()

        if not touched:
            show_leg = not untouched_legend_shown
            traces.append(
                go.Scatter(
                    x=time_grid,
                    y=path,
                    mode="lines",
                    line=dict(color=UNTOUCHED_COLOR, width=MAIN_LINE_WIDTH),
                    name="nie dotknęła bariery" if show_leg else None,
                    showlegend=show_leg,
                )
            )
            trace_ranges.append((0, len(time_grid)))
            trace_parents.append(i)
            untouched_legend_shown = untouched_legend_shown or show_leg
            continue

        for is_above, start_idx, end_idx in _split_path_segments_indices(path, barrier):
            x_seg = time_grid[start_idx:end_idx]
            y_seg = path[start_idx:end_idx]
            if is_above:
                show_leg = not above_legend_shown
                traces.append(
                    go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode="lines",
                        line=dict(color=TOUCHED_ABOVE_COLOR, width=MAIN_LINE_WIDTH),
                        name="dotknęła: powyżej" if show_leg else None,
                        showlegend=show_leg,
                        hoverinfo="skip",
                    )
                )
                trace_ranges.append((start_idx, end_idx))
                trace_parents.append(i)
                above_legend_shown = above_legend_shown or show_leg
            else:
                show_leg = not below_legend_shown
                traces.append(
                    go.Scatter(
                        x=x_seg,
                        y=y_seg,
                        mode="lines",
                        line=dict(color=TOUCHED_BELOW_COLOR, width=MAIN_LINE_WIDTH),
                        name="dotknęła: poniżej" if show_leg else None,
                        showlegend=show_leg,
                        hoverinfo="skip",
                    )
                )
                trace_ranges.append((start_idx, end_idx))
                trace_parents.append(i)
                below_legend_shown = below_legend_shown or show_leg

    traces.append(
        go.Scatter(
            x=[time_grid[0], time_grid[-1]],
            y=[barrier, barrier],
            mode="lines",
            line=dict(color=BARRIER_COLOR, width=BARRIER_WIDTH, dash="dash"),
            name=f"bariera {barrier}",
            hoverinfo="skip",
        )
    )
    trace_ranges.append(None)
    trace_parents.append(None)

    return traces, trace_ranges, trace_parents


def _build_background_traces(
    time_grid: np.ndarray, S_paths_bg: np.ndarray
) -> Tuple[List[go.Scatter], List[Tuple[int, int] | None], List[Optional[int]]]:
    traces = []
    ranges: List[Tuple[int, int] | None] = []
    parents: List[Optional[int]] = []
    for i in range(len(S_paths_bg)):
        traces.append(
            go.Scatter(
                x=time_grid,
                y=S_paths_bg[i],
                mode="lines",
                line=dict(color=BACKGROUND_COLOR, width=BACKGROUND_LINE_WIDTH),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        ranges.append(None)
        parents.append(None)
    return traces, ranges, parents


def _build_visibility(
    total_traces: int,
    bg_count: int,
    barrier_trace_ranges: List[Tuple[int, int]],
) -> List[List[bool]]:
    visibility_states: List[List[bool]] = []

    for start, end in barrier_trace_ranges:
        visibility = [True] * bg_count + [False] * (total_traces - bg_count)
        for idx in range(start, end):
            visibility[idx] = True
        visibility_states.append(visibility)

    return visibility_states


def build_gbm_trajectories_figure(
    R_paths_main: int = 8,
    background_multiplier: int = 3,
) -> go.Figure:
    """
    Build an interactive Plotly figure showing GBM trajectories with barriers.
    """
    R_paths_bg = R_paths_main * background_multiplier

    S_paths_main = simulate_gbm_paths(
        S0, mu_star, sigma, T, n_steps, R_paths_main, seed
    )
    S_paths_bg = simulate_gbm_paths(
        S0,
        mu_star,
        sigma,
        T,
        n_steps,
        R_paths_bg,
        seed + 1 if seed is not None else None,
    )
    time_grid = np.linspace(T / n_steps, T, n_steps)

    traces: List[go.Scatter] = []
    trace_ranges: List[Tuple[int, int] | None] = []
    trace_parents: List[Optional[int]] = []

    bg_traces, bg_ranges, bg_parents = _build_background_traces(time_grid, S_paths_bg)
    traces.extend(bg_traces)
    trace_ranges.extend(bg_ranges)
    trace_parents.extend(bg_parents)

    barrier_trace_ranges: List[Tuple[int, int]] = []
    for barrier in BARRIERS:
        start = len(traces)
        barrier_traces, barrier_ranges, barrier_parents = _build_barrier_traces(
            time_grid, S_paths_main, barrier, R_paths_main
        )
        traces.extend(barrier_traces)
        trace_ranges.extend(barrier_ranges)
        trace_parents.extend(barrier_parents)
        end = len(traces)
        barrier_trace_ranges.append((start, end))

    fig = go.Figure(data=traces)

    base_xy = [(np.array(tr.x), np.array(tr.y)) for tr in fig.data]

    bg_count = R_paths_bg
    visibility_states = _build_visibility(
        total_traces=len(traces),
        bg_count=bg_count,
        barrier_trace_ranges=barrier_trace_ranges,
    )

    buttons = []
    for visibility, barrier in zip(visibility_states, BARRIERS):
        buttons.append(
            dict(
                label=f"bariera {barrier}",
                method="update",
                args=[
                    {"visible": visibility},
                    {"title": f"Przykładowe trajektorie GBM - bariera {barrier}"},
                ],
            )
        )

    # Build animation frames with time proportional to step length per path.
    step_lengths = np.sqrt(
        np.diff(time_grid) ** 2 + np.diff(S_paths_main, axis=1) ** 2
    )
    median_length = np.median(step_lengths)
    path_repeats = np.clip(
        np.round(step_lengths / median_length).astype(int), 1, 6
    )

    path_frame_steps: List[List[int]] = []
    for repeats in path_repeats:
        steps = [1]
        for step_idx, rep in enumerate(repeats, start=2):
            steps.extend([step_idx] * rep)
        path_frame_steps.append(steps)

    max_frames = max(len(s) for s in path_frame_steps) if path_frame_steps else 0

    def _slice_trace(trace_idx: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
        rng = trace_ranges[trace_idx]
        base_x, base_y = base_xy[trace_idx]
        if rng is None:
            return base_x, base_y

        start, _ = rng
        rel = step - start
        if rel <= 0:
            return np.array([]), np.array([])
        rel = min(rel, len(base_x))
        return base_x[:rel], base_y[:rel]

    frames = []
    for frame_i in range(max_frames):
        frame_data = []
        for trace_idx in range(len(traces)):
            parent = trace_parents[trace_idx]
            if parent is None:
                x_slice, y_slice = base_xy[trace_idx]
            else:
                schedule = path_frame_steps[parent]
                step = schedule[min(frame_i, len(schedule) - 1)]
                x_slice, y_slice = _slice_trace(trace_idx, step)
            frame_data.append(go.Scatter(x=x_slice, y=y_slice))
        frames.append(go.Frame(name=f"f{frame_i}", data=frame_data))

    fig.frames = frames

    if frames:
        init_data = frames[0].data
        for trace, frame_state in zip(fig.data, init_data):
            trace.x = frame_state.x
            trace.y = frame_state.y
    frame_names = [f.name for f in frames]

    fig.update_layout(
        title=dict(
            text=f"Przykładowe trajektorie GBM - bariera {BARRIERS[0]}",
            x=0.5,
        ),
        xaxis_title="czas",
        yaxis_title="cena aktywa",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=1.02),
        margin=dict(l=60, r=200, t=60, b=80),
        width=980,
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                buttons=buttons,
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                showactive=True,
            ),
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="▶",
                        method="animate",
                        args=[
                            frame_names,
                            {
                                "mode": "immediate",
                                "frame": {"duration": 45, "redraw": False},
                                "transition": {"duration": 0},
                            }
                        ],
                    ),
                ],
                x=0.9,
                y=-0.08,
                xanchor="right",
                yanchor="top",
            ),
        ],
    )

    initial_visibility = visibility_states[0]
    for trace, vis in zip(fig.data, initial_visibility):
        trace.visible = vis

    return fig


if __name__ == "__main__":
    build_gbm_trajectories_figure().show()
