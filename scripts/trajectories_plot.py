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


def _split_segments_with_crossings(
    time_grid: np.ndarray, path: np.ndarray, barrier: float
) -> List[Tuple[bool, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split a path into below/above segments, inserting a crossing point when a segment
    crosses the barrier so that colors switch exactly at the barrier level.
    Returns a list of tuples (is_above, xs, ys, step_positions).
    step_positions are 1-based indices aligned with time_grid so we can animate by step.
    """
    segments: List[Tuple[bool, np.ndarray, np.ndarray, np.ndarray]] = []

    current_above = path[0] >= barrier
    xs = [time_grid[0]]
    ys = [path[0]]
    steps = [1]  # 1-based to match animation steps

    for idx in range(1, len(path)):
        x_prev, y_prev = time_grid[idx - 1], path[idx - 1]
        x_curr, y_curr = time_grid[idx], path[idx]
        next_above = y_curr >= barrier

        crosses = (y_prev - barrier) * (y_curr - barrier) < 0

        if crosses:
            # Linear interpolation to find crossing with the barrier.
            t = (barrier - y_prev) / (y_curr - y_prev)
            x_cross = x_prev + t * (x_curr - x_prev)
            y_cross = barrier

            xs.append(x_cross)
            ys.append(y_cross)
            steps.append(idx + 1)  # crossing revealed when reaching current point
            segments.append(
                (current_above, np.array(xs), np.array(ys), np.array(steps))
            )

            # Start new segment from the crossing point.
            xs = [x_cross, x_curr]
            ys = [y_cross, y_curr]
            steps = [idx + 1, idx + 1]
            current_above = next_above
        else:
            if next_above == current_above:
                xs.append(x_curr)
                ys.append(y_curr)
                steps.append(idx + 1)
            else:
                # No numeric crossing (touch at a point).
                xs.append(x_curr)
                ys.append(y_curr)
                steps.append(idx + 1)
                segments.append(
                    (current_above, np.array(xs), np.array(ys), np.array(steps))
                )
                xs = [x_curr]
                ys = [y_curr]
                steps = [idx + 1]
                current_above = next_above

    segments.append((current_above, np.array(xs), np.array(ys), np.array(steps)))
    return segments


def _build_barrier_traces(
    time_grid: np.ndarray,
    S_paths_main: np.ndarray,
    barrier: float,
    R_paths_main: int,
) -> Tuple[List[go.Scatter], List[Optional[np.ndarray]], List[Optional[int]]]:
    traces: List[go.Scatter] = []
    trace_steps: List[Optional[np.ndarray]] = []
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
            trace_steps.append(np.arange(1, len(time_grid) + 1))
            trace_parents.append(i)
            untouched_legend_shown = untouched_legend_shown or show_leg
            continue

        for is_above, x_seg, y_seg, step_pos in _split_segments_with_crossings(
            time_grid, path, barrier
        ):
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
                trace_steps.append(step_pos)
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
                trace_steps.append(step_pos)
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
    trace_steps.append(None)
    trace_parents.append(None)

    return traces, trace_steps, trace_parents


def _build_background_traces(
    time_grid: np.ndarray, S_paths_bg: np.ndarray
) -> Tuple[List[go.Scatter], List[Optional[np.ndarray]], List[Optional[int]]]:
    traces = []
    steps: List[Optional[np.ndarray]] = []
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
        steps.append(None)
        parents.append(None)
    return traces, steps, parents


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
    trace_steps: List[Optional[np.ndarray]] = []
    trace_parents: List[Optional[int]] = []

    bg_traces, bg_steps, bg_parents = _build_background_traces(time_grid, S_paths_bg)
    traces.extend(bg_traces)
    trace_steps.extend(bg_steps)
    trace_parents.extend(bg_parents)

    barrier_trace_ranges: List[Tuple[int, int]] = []
    for barrier in BARRIERS:
        start = len(traces)
        barrier_traces, barrier_steps, barrier_parents = _build_barrier_traces(
            time_grid, S_paths_main, barrier, R_paths_main
        )
        traces.extend(barrier_traces)
        trace_steps.extend(barrier_steps)
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
        step_positions = trace_steps[trace_idx]
        base_x, base_y = base_xy[trace_idx]
        if step_positions is None:
            return base_x, base_y

        count = np.searchsorted(step_positions, step, side="right")
        return base_x[:count], base_y[:count]

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
