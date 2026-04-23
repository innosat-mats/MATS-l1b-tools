"""Streamlit quick-look GUI for MATS L1b zarr data.

Run with:
    streamlit run scripts/quickview.py
"""
from __future__ import annotations

import io
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, time, timedelta
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from mats_l1b_tools.fetch_data import fetch_MATS_l1b_data

CHANNELS = ["IR1", "IR2", "IR3", "IR4", "UV1", "UV2"]
EXTEND_TIME = timedelta(hours=1)
EXTEND_DEG = 10.0
MAX_IMAGES_WARN = 200
CROSS_CHANNEL_TOL_S = 6.0
CMAP = "magma"

# Panel layout for the 6-channel orbit-view movie (row, col).
PANEL_LAYOUT = {
    "IR1": (0, 0), "IR3": (0, 1), "UV1": (0, 2),
    "IR2": (1, 0), "IR4": (1, 1), "UV2": (1, 2),
}

SIMPLE_VARS = ["ImageCalibrated", "TPlat", "TPlon"]
ORBIT_VIEW_VARS = SIMPLE_VARS + ["TPheight", "TPsza", "TPssa",
                                 "nadir_sza", "satlat", "satlon", "satheight"]


# ---------- data access ----------

@st.cache_resource(show_spinner="Opening zarr archive...")
def open_channel(channel: str) -> xr.Dataset:
    return fetch_MATS_l1b_data(channel)


@st.cache_data(show_spinner="Loading index...")
def load_index(channel: str) -> dict:
    """Tiny per-image scalars, pulled once and reused for every filter op."""
    ds = open_channel(channel)
    return {
        "time": ds["time"].values,
        "TPlat": np.asarray(ds["TPlat"].values, dtype=float),
        "TPlon": np.asarray(ds["TPlon"].values, dtype=float),
    }


@st.cache_data(show_spinner=False)
def channel_extent(channel: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    idx = load_index(channel)
    t = idx["time"]
    return pd.Timestamp(t[0]), pd.Timestamp(t[-1])


@st.cache_data(show_spinner=False)
def image_units(channel: str) -> str:
    ds = open_channel(channel)
    return str(ds["ImageCalibrated"].attrs.get("units", "")).strip()


def query_indices(idx, t0, t1, lat0, lat1, lon0, lon1) -> np.ndarray:
    t, la, lo = idx["time"], idx["TPlat"], idx["TPlon"]
    t0_64, t1_64 = np.datetime64(t0, "ns"), np.datetime64(t1, "ns")
    mask = ((t >= t0_64) & (t <= t1_64)
            & (la >= lat0) & (la <= lat1)
            & (lo >= lon0) & (lo <= lon1))
    return np.where(mask)[0]


def closest_time_in_latlon(idx, t0, t1, lat0, lat1, lon0, lon1):
    t, la, lo = idx["time"], idx["TPlat"], idx["TPlon"]
    t_ext0 = np.datetime64(t0 - EXTEND_TIME, "ns")
    t_ext1 = np.datetime64(t1 + EXTEND_TIME, "ns")
    m = ((t >= t_ext0) & (t <= t_ext1)
         & (la >= lat0) & (la <= lat1)
         & (lo >= lon0) & (lo <= lon1))
    if not m.any():
        return None
    ts = t[m]
    t0_64, t1_64 = np.datetime64(t0, "ns"), np.datetime64(t1, "ns")
    before = t0_64 - ts
    after = ts - t1_64
    dist = np.maximum(np.maximum(before, after), np.timedelta64(0, "ns"))
    j = int(np.argmin(dist))
    gidx = np.where(m)[0][j]
    return {
        "time": pd.to_datetime(t[gidx]),
        "lat": float(la[gidx]),
        "lon": float(lo[gidx]),
        "offset_seconds": float(dist[j] / np.timedelta64(1, "s")),
    }


def closest_latlon_in_time(idx, t0, t1, lat0, lat1, lon0, lon1):
    t, la, lo = idx["time"], idx["TPlat"], idx["TPlon"]
    t0_64, t1_64 = np.datetime64(t0, "ns"), np.datetime64(t1, "ns")
    m = (t >= t0_64) & (t <= t1_64)
    if not m.any():
        return None
    la_s, lo_s, t_s = la[m], lo[m], t[m]
    la_d = np.clip(np.maximum(lat0 - la_s, la_s - lat1), 0, None)
    lo_d = np.clip(np.maximum(lon0 - lo_s, lo_s - lon1), 0, None)
    dist = np.sqrt(la_d**2 + lo_d**2)
    j = int(np.argmin(dist))
    return {
        "time": pd.to_datetime(t_s[j]),
        "lat": float(la_s[j]),
        "lon": float(lo_s[j]),
        "distance_deg": float(dist[j]),
    }


def extended_count(idx, t0, t1, lat0, lat1, lon0, lon1) -> int:
    inds = query_indices(
        idx,
        t0 - EXTEND_TIME, t1 + EXTEND_TIME,
        lat0 - EXTEND_DEG, lat1 + EXTEND_DEG,
        lon0 - EXTEND_DEG, lon1 + EXTEND_DEG,
    )
    return int(len(inds))


def fetch_subset(channel: str, inds: np.ndarray, needed_vars: list[str]) -> xr.Dataset:
    """Load only the given vars at the given time indices. Slicing a
    contiguous range first keeps access aligned with zarr chunks."""
    ds = open_channel(channel)
    keep = [v for v in needed_vars if v in ds.data_vars]
    ds = ds[keep]
    if inds.size == 0:
        return ds.isel(time=slice(0, 0))
    lo, hi = int(inds.min()), int(inds.max())
    sub = ds.isel(time=slice(lo, hi + 1))
    local = np.asarray(inds, dtype=int) - lo
    return sub.isel(time=local).load()


def fetch_time_matched(channel: str, target_times: np.ndarray, tol_s: float,
                       needed_vars: list[str]) -> tuple[xr.Dataset, np.ndarray]:
    """For each time in `target_times`, find the nearest image in `channel`
    within tolerance. Returns (dataset of matched rows, match-index-per-target
    array with -1 where no match).

    The per-channel zarr time array is sorted, so we use a single vectorized
    searchsorted instead of a per-target argmin. The matched indices are then
    loaded with one contiguous zarr slice.
    """
    idx = load_index(channel)
    t = idx["time"]
    n_t = t.size
    n_targets = len(target_times)
    per_target = np.full(n_targets, -1, dtype=int)

    if n_t == 0 or n_targets == 0:
        ds = open_channel(channel)
        keep = [v for v in needed_vars if v in ds.data_vars]
        return ds[keep].isel(time=slice(0, 0)), per_target

    tt = np.asarray(target_times, dtype="datetime64[ns]")
    tol = np.timedelta64(int(tol_s * 1e9), "ns")
    j = np.searchsorted(t, tt)
    j_left = np.clip(j - 1, 0, n_t - 1)
    j_right = np.clip(j, 0, n_t - 1)
    d_left = np.abs(t[j_left] - tt)
    d_right = np.abs(t[j_right] - tt)
    take_right = d_right < d_left
    best = np.where(take_right, j_right, j_left)
    best_d = np.where(take_right, d_right, d_left)
    within = best_d <= tol
    per_target = np.where(within, best, -1).astype(int)

    good = per_target[per_target >= 0]
    if good.size == 0:
        ds = open_channel(channel)
        keep = [v for v in needed_vars if v in ds.data_vars]
        return ds[keep].isel(time=slice(0, 0)), per_target

    unique_sel = np.unique(good)
    sub = fetch_subset(channel, unique_sel, needed_vars)
    pos_lookup = {int(u): i for i, u in enumerate(unique_sel)}
    remapped = np.array(
        [pos_lookup[int(p)] if p >= 0 else -1 for p in per_target], dtype=int,
    )
    return sub, remapped


# ---------- plotting helpers ----------

def resolve_limits(arr, nstd, vmin, vmax):
    if vmin is not None and vmax is not None:
        return float(vmin), float(vmax)
    mean, std = float(np.nanmean(arr)), float(np.nanstd(arr))
    return mean - nstd * std, mean + nstd * std


def _title(ts, lat, lon, channel) -> str:
    t = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    return f"{channel}  {t}  TP=({lat:.2f}, {lon:.2f})"


def plot_simple(image, ts, lat, lon, channel, nstd=2.0, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 3))
    lo, hi = resolve_limits(image, nstd, vmin, vmax)
    im = ax.imshow(image, origin="lower", aspect="auto",
                   cmap=CMAP, vmin=lo, vmax=hi)
    ax.set_title(_title(ts, lat, lon, channel), fontsize=10)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    cb = fig.colorbar(im, ax=ax)
    units = image_units(channel)
    if units:
        cb.set_label(units)
    fig.tight_layout()
    return fig


def make_animation_gif(images, metas, channel, fps=4,
                       nstd=2.0, vmin=None, vmax=None) -> tuple[bytes, str]:
    stack = np.stack(images)
    lo, hi = resolve_limits(stack, nstd, vmin, vmax)

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(images[0], origin="lower", aspect="auto",
                   cmap=CMAP, vmin=lo, vmax=hi)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    title = ax.set_title(_title(*metas[0], channel), fontsize=10)
    cb = fig.colorbar(im, ax=ax)
    units = image_units(channel)
    if units:
        cb.set_label(units)
    fig.tight_layout()

    def update(i):
        im.set_array(images[i])
        title.set_text(_title(*metas[i], channel))
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(images), interval=1000 // fps, blit=False,
    )
    data = _anim_to_movie_bytes(anim, fps)
    plt.close(fig)
    return data


def _anim_to_movie_bytes(anim, fps) -> tuple[bytes, str]:
    """Produce an MP4 (click-to-pause via st.video) if ffmpeg is available,
    otherwise fall back to an animated GIF."""
    if animation.FFMpegWriter.isAvailable():
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            anim.save(path, writer=animation.FFMpegWriter(fps=fps))
            return Path(path).read_bytes(), "mp4"
        except Exception:
            pass
        finally:
            Path(path).unlink(missing_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        path = f.name
    try:
        anim.save(path, writer=animation.PillowWriter(fps=fps))
        return Path(path).read_bytes(), "gif"
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- orbit view movie ----------

def _scalar(row, name):
    if name in row:
        v = row[name].values
        return float(v) if v.ndim == 0 else v
    return None


def build_orbit_frames(target_times: np.ndarray,
                       matched: dict[str, tuple[xr.Dataset, np.ndarray]]):
    """Build a list of frames. Each frame has, per channel, either a panel
    dict or None (hold previous / blank)."""
    frames = []
    last_per_channel: dict[str, dict | None] = {ch: None for ch in CHANNELS}
    for k, t in enumerate(target_times):
        panels: dict[str, dict | None] = {}
        for ch in CHANNELS:
            ds, remap = matched.get(ch, (None, None))
            panel = None
            if ds is not None and remap is not None:
                local = int(remap[k])
                if local >= 0:
                    row = ds.isel(time=local)
                    panel = {
                        "image": np.asarray(row["ImageCalibrated"].values),
                        "time": row["time"].values,
                        "TPlat": _scalar(row, "TPlat"),
                        "TPlon": _scalar(row, "TPlon"),
                        "satlat": _scalar(row, "satlat"),
                        "satlon": _scalar(row, "satlon"),
                        "TPsza": _scalar(row, "TPsza"),
                        "TPssa": _scalar(row, "TPssa"),
                        "nadir_sza": _scalar(row, "nadir_sza"),
                        "TPheight": _scalar(row, "TPheight"),
                    }
                    last_per_channel[ch] = panel
            if panel is None:
                panel = last_per_channel[ch]
            panels[ch] = panel
        frames.append({"time": t, "panels": panels})
    return frames


def _channel_color_limits(frames, q):
    """Global vmin/vmax per channel so the movie doesn't flicker."""
    vlims = {}
    for ch in CHANNELS:
        imgs = [f["panels"][ch]["image"] for f in frames
                if f["panels"][ch] is not None]
        if imgs:
            vlims[ch] = resolve_limits(np.stack(imgs),
                                       q["nstd"], q["vmin"], q["vmax"])
    return vlims


def _reference_panel(frame):
    """Pick the panel to drive the map/info line — IR1 if available else
    first non-None panel."""
    p = frame["panels"].get("IR1")
    if p is not None:
        return p
    for ch in CHANNELS:
        if frame["panels"].get(ch) is not None:
            return frame["panels"][ch]
    return None


def _format_info(frame):
    t_str = pd.Timestamp(frame["time"]).strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"time      : {t_str}"]
    p = _reference_panel(frame)
    if p is None:
        return "\n".join(lines + ["(no data)"])
    if p["TPlat"] is not None:
        lines.append(f"TPlat     : {p['TPlat']:7.2f}°")
    if p["TPlon"] is not None:
        lines.append(f"TPlon     : {p['TPlon']:7.2f}°")
    if p["TPheight"] is not None:
        lines.append(f"TPheight  : {p['TPheight']/1e3:7.1f} km")
    if p["TPsza"] is not None:
        lines.append(f"TPsza     : {p['TPsza']:7.2f}°")
    if p["TPssa"] is not None:
        lines.append(f"TPssa     : {p['TPssa']:7.2f}°")
    if p["nadir_sza"] is not None:
        lines.append(f"nadir_sza : {p['nadir_sza']:7.2f}°")
    if p["satlat"] is not None and p["satlon"] is not None:
        lines.append(f"sat       : ({p['satlat']:6.2f}, {p['satlon']:6.2f})")
    return "\n".join(lines)


def render_orbit_view_movie(frames, q, fps=4) -> tuple[bytes, str]:
    try:
        import cartopy.crs as ccrs
        have_cartopy = True
    except Exception:
        have_cartopy = False

    vlims = _channel_color_limits(frames, q)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.5])

    ax_panels = {ch: fig.add_subplot(gs[r, c])
                 for ch, (r, c) in PANEL_LAYOUT.items()}
    if have_cartopy:
        ax_map = fig.add_subplot(gs[2, 0:2], projection=ccrs.PlateCarree())
    else:
        ax_map = fig.add_subplot(gs[2, 0:2])
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis("off")

    # Initialize each panel with the first non-None frame for that channel.
    ims: dict[str, any] = {}
    titles: dict[str, any] = {}
    for ch, ax in ax_panels.items():
        first = next((f["panels"][ch] for f in frames
                      if f["panels"][ch] is not None), None)
        if first is None:
            ax.text(0.5, 0.5, f"{ch}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ims[ch] = None
            titles[ch] = None
            continue
        lo, hi = vlims.get(ch, (0, 1))
        im = ax.imshow(first["image"], origin="lower", aspect="auto",
                       cmap=CMAP, vmin=lo, vmax=hi)
        ax.set_xticks([])
        ax.set_yticks([])
        titles[ch] = ax.set_title(ch, fontsize=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        units = image_units(ch)
        if units:
            cb.set_label(units, fontsize=7, rotation=270, labelpad=10)
        cb.ax.tick_params(labelsize=7)
        ims[ch] = im

    # Map setup.
    if have_cartopy:
        ax_map.set_global()
        ax_map.coastlines(linewidth=0.5)
        gl = ax_map.gridlines(draw_labels=True, alpha=0.3, linewidth=0.4)
        gl.top_labels = False
        gl.right_labels = False
        tp_marker = ax_map.scatter(
            [], [], s=30, color="red", label="TP",
            transform=ccrs.PlateCarree(), zorder=5,
        )
        sat_marker = ax_map.scatter(
            [], [], s=30, color="blue", label="sat",
            transform=ccrs.PlateCarree(), zorder=5,
        )
    else:
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-90, 90)
        ax_map.grid(alpha=0.3)
        ax_map.set_xlabel("longitude [°]")
        ax_map.set_ylabel("latitude [°]")
        tp_marker = ax_map.scatter([], [], s=30, color="red", label="TP", zorder=5)
        sat_marker = ax_map.scatter([], [], s=30, color="blue", label="sat", zorder=5)
    ax_map.legend(fontsize=9, loc="lower left")

    info_text = ax_info.text(
        0.0, 0.98, "", transform=ax_info.transAxes,
        fontsize=10, va="top", family="monospace",
    )

    fig.tight_layout()

    def update(i):
        frame = frames[i]
        for ch, im in ims.items():
            if im is None:
                continue
            p = frame["panels"].get(ch)
            if p is None:
                continue
            im.set_array(p["image"])
            if titles[ch] is not None:
                ts = pd.Timestamp(p["time"]).strftime("%Y-%m-%d %H:%M:%S")
                titles[ch].set_text(f"{ch}  {ts}")
        ref = _reference_panel(frame)
        if ref is not None:
            tp_marker.set_offsets([[ref["TPlon"], ref["TPlat"]]])
            if ref["satlat"] is not None and ref["satlon"] is not None:
                sat_marker.set_offsets([[ref["satlon"], ref["satlat"]]])
        info_text.set_text(_format_info(frame))
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=False,
    )
    data = _anim_to_movie_bytes(anim, fps)
    plt.close(fig)
    return data


def dataset_to_netcdf_bytes(ds: xr.Dataset) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as f:
        path = f.name
    try:
        ds.to_netcdf(path)
        return Path(path).read_bytes()
    finally:
        Path(path).unlink(missing_ok=True)


# ---------- UI ----------

def sidebar_inputs():
    st.sidebar.header("Query")

    try:
        ds_t0, ds_t1 = channel_extent("IR1")
        default_d0 = ds_t0.date()
        default_t0 = (int(ds_t0.hour), int(ds_t0.minute), int(ds_t0.second))
        end_dt = ds_t0 + pd.Timedelta(hours=24)
        default_d1 = end_dt.date()
        default_t1 = (int(end_dt.hour), int(end_dt.minute), int(end_dt.second))
        extent_hint = (f"Dataset (IR1): {ds_t0:%Y-%m-%d %H:%M:%S} → "
                       f"{ds_t1:%Y-%m-%d %H:%M:%S} UTC")
    except Exception as exc:
        today = date.today()
        default_d0 = today - timedelta(days=1)
        default_d1 = today
        default_t0 = default_t1 = (0, 0, 0)
        extent_hint = f"(could not read dataset extent: {exc})"

    st.sidebar.caption(extent_hint)

    def hms(label, dh, dm, ds):
        st.sidebar.markdown(f"**{label} time** (UTC, h / m / s)")
        a, b, c = st.sidebar.columns(3)
        with a:
            h = st.number_input("h", 0, 23, dh, key=f"{label}_h",
                                label_visibility="collapsed")
        with b:
            m = st.number_input("m", 0, 59, dm, key=f"{label}_m",
                                label_visibility="collapsed")
        with c:
            s = st.number_input("s", 0, 59, ds, key=f"{label}_s",
                                label_visibility="collapsed")
        return time(int(h), int(m), int(s))

    d0 = st.sidebar.date_input("Start date (UTC)", value=default_d0, key="d0")
    t0 = hms("Start", *default_t0)
    d1 = st.sidebar.date_input("End date (UTC)", value=default_d1, key="d1")
    t1 = hms("End", *default_t1)

    st.sidebar.markdown("**Tangent-point latitude**")
    lat0, lat1 = st.sidebar.slider(
        "lat", -90.0, 90.0, (-90.0, 90.0), step=1.0, key="lat",
        label_visibility="collapsed",
    )
    st.sidebar.markdown("**Tangent-point longitude**")
    lon0, lon1 = st.sidebar.slider(
        "lon", -180.0, 180.0, (-180.0, 180.0), step=1.0, key="lon",
        label_visibility="collapsed",
    )

    channels = st.sidebar.multiselect(
        "Primary channels (drive the lat/lon filter)",
        CHANNELS, default=["IR1"], key="channels",
    )
    plot_mode = st.sidebar.radio(
        "Plot style",
        ["simple (fast)", "orbit view (6-channel movie)"],
        key="plot_mode",
    )
    color_mode = st.sidebar.radio(
        "Color range", ["auto (N std)", "manual"],
        key="color_mode", horizontal=True,
    )
    if color_mode == "auto (N std)":
        nstd = st.sidebar.slider("N std", 0.5, 5.0, 2.0, 0.5, key="nstd")
        vmin = vmax = None
    else:
        nstd = 2.0
        cA, cB = st.sidebar.columns(2)
        with cA:
            vmin = st.number_input("vmin", value=0.0, format="%g", key="vmin")
        with cB:
            vmax = st.number_input("vmax", value=1e14, format="%g", key="vmax")

    fps = st.sidebar.slider("Movie fps", 1, 20, 4, key="fps")

    return dict(
        t_start=datetime.combine(d0, t0),
        t_end=datetime.combine(d1, t1),
        lat0=lat0, lat1=lat1, lon0=lon0, lon1=lon1,
        channels=channels,
        plot_mode=plot_mode, fps=fps,
        nstd=nstd, vmin=vmin, vmax=vmax,
    )


def describe_closest(ch, idx, q) -> str:
    it = closest_time_in_latlon(idx, q["t_start"], q["t_end"],
                                q["lat0"], q["lat1"], q["lon0"], q["lon1"])
    ill = closest_latlon_in_time(idx, q["t_start"], q["t_end"],
                                 q["lat0"], q["lat1"], q["lon0"], q["lon1"])
    out = [f"**{ch}: no data in selected window.**"]
    if it is not None:
        out.append(
            f"- Closest time with TP in lat/lon box (±1 h): "
            f"{it['time']} at TP=({it['lat']:.2f}, {it['lon']:.2f}), "
            f"offset {it['offset_seconds']:.0f} s"
        )
    else:
        out.append("- No data with TP in the lat/lon box within ±1 h.")
    if ill is not None:
        out.append(
            f"- Closest TP in time window: ({ill['lat']:.2f}, {ill['lon']:.2f}) "
            f"at {ill['time']}, {ill['distance_deg']:.2f}° away."
        )
    else:
        out.append("- No data in the time window.")
    if it is None and ill is None:
        n = extended_count(idx, q["t_start"], q["t_end"],
                           q["lat0"], q["lat1"], q["lon0"], q["lon1"])
        out.append(
            "- No data in ±1 h, ±10° either." if n == 0
            else f"- Extended search (±1 h, ±10°) finds **{n}** images."
        )
    return "\n".join(out)


def run_preview(q) -> dict:
    """Index-only probe: returns per-channel indices and their times/lats/lons
    without touching image data."""
    preview = {}
    for ch in q["channels"]:
        idx = load_index(ch)
        inds = query_indices(idx, q["t_start"], q["t_end"],
                             q["lat0"], q["lat1"], q["lon0"], q["lon1"])
        preview[ch] = {
            "inds": inds,
            "times": idx["time"][inds],
            "TPlat": idx["TPlat"][inds],
            "TPlon": idx["TPlon"][inds],
        }
    return preview


def render_preview(preview, q):
    total = sum(len(p["inds"]) for p in preview.values())
    cols = st.columns(len(preview)) if preview else []
    for col, (ch, info) in zip(cols, preview.items()):
        col.metric(f"{ch}", f"{len(info['inds'])} img")

    for ch, info in preview.items():
        if len(info["inds"]) == 0:
            st.warning(describe_closest(ch, load_index(ch), q))

    has_any = any(len(info["inds"]) > 0 for info in preview.values())
    if has_any:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
        for ch, info in preview.items():
            if len(info["inds"]) == 0:
                continue
            ax1.scatter(info["times"], info["TPlat"], s=8, label=ch)
            ax2.scatter(info["times"], info["TPlon"], s=8, label=ch)
        for ax, ylabel in ((ax1, "TP lat [°]"), (ax2, "TP lon [°]")):
            ax.set_xlabel("time (UTC)")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            for lab in ax.get_xticklabels():
                lab.set_rotation(30)
                lab.set_ha("right")
        fig.tight_layout()
        st.pyplot(fig)
    return total


def run_query(q, messages: list, preview: dict | None = None) -> dict:
    """Execute the query. Returns a dict with:
      - 'primary': {channel: xr.Dataset} for user-selected channels (lat/lon
        filter applied).
      - 'orbit': {channel: (xr.Dataset, remap_array)} for all 6 channels
        (orbit-view mode only). remap_array[k] is the row in the dataset
        corresponding to frame k of the primary timeline, or -1 for none.
      - 'timeline': np.ndarray of datetime64 driving the movie frames.
    """
    primary: dict[str, xr.Dataset] = {}
    missing: list[str] = []
    needed_vars = (ORBIT_VIEW_VARS
                   if q["plot_mode"].startswith("orbit view") else SIMPLE_VARS)

    # Resolve indices per channel (fast, in-memory).
    per_ch_inds: dict[str, np.ndarray] = {}
    for ch in q["channels"]:
        if preview is not None and ch in preview:
            inds = preview[ch]["inds"]
        else:
            idx = load_index(ch)
            inds = query_indices(idx, q["t_start"], q["t_end"],
                                 q["lat0"], q["lat1"], q["lon0"], q["lon1"])
        if len(inds) == 0:
            missing.append(ch)
            continue
        if len(inds) > MAX_IMAGES_WARN:
            messages.append(("warning",
                f"{ch}: {len(inds)} images — truncating to first {MAX_IMAGES_WARN}."))
            inds = inds[:MAX_IMAGES_WARN]
        per_ch_inds[ch] = inds

    # Parallel fetch across channels — HTTP I/O overlaps via s3fs.
    if per_ch_inds:
        progress = st.progress(0.0, text="Loading primary channels in parallel...")
        n_primary = len(per_ch_inds)
        done = 0
        with ThreadPoolExecutor(max_workers=min(6, n_primary)) as ex:
            futures = {ex.submit(fetch_subset, ch, inds, needed_vars): ch
                       for ch, inds in per_ch_inds.items()}
            for fut in as_completed(futures):
                ch = futures[fut]
                primary[ch] = fut.result()
                done += 1
                progress.progress(done / n_primary,
                                  text=f"{ch}: loaded ({done}/{n_primary})")
                messages.append(("success",
                    f"**{ch}: {len(per_ch_inds[ch])} image(s)** match."))
        progress.empty()

    for ch in missing:
        messages.append(("warning", describe_closest(ch, load_index(ch), q)))

    out = {"primary": primary}

    if q["plot_mode"].startswith("orbit view") and primary:
        timeline = np.unique(np.concatenate(
            [ds["time"].values for ds in primary.values()]
        ))
        orbit: dict[str, tuple[xr.Dataset, np.ndarray]] = {}
        prog = st.progress(0.0, text="Time-matching channels for movie...")
        done = 0
        with ThreadPoolExecutor(max_workers=len(CHANNELS)) as ex:
            futures = {
                ex.submit(fetch_time_matched, ch, timeline,
                          CROSS_CHANNEL_TOL_S, ORBIT_VIEW_VARS): ch
                for ch in CHANNELS
            }
            for fut in as_completed(futures):
                ch = futures[fut]
                sub, remap = fut.result()
                orbit[ch] = (sub, remap)
                done += 1
                prog.progress(done / len(CHANNELS),
                              text=f"{ch}: matched ({done}/{len(CHANNELS)})")
                n_hits = int((remap >= 0).sum())
                if n_hits > 0:
                    messages.append(
                        ("info", f"orbit view: {ch} has {n_hits}/{len(timeline)} "
                                 f"frames within ±{CROSS_CHANNEL_TOL_S:.0f} s.")
                    )
        prog.empty()
        out["orbit"] = orbit
        out["timeline"] = timeline

    return out


def render_simple(primary: dict[str, xr.Dataset], q):
    for ch, hit in primary.items():
        n = int(hit.sizes["time"])
        st.subheader(f"{ch}  —  {n} image(s)")
        images = np.stack(hit["ImageCalibrated"].values)
        times = hit["time"].values
        lats = [float(x) for x in hit["TPlat"].values]
        lons = [float(x) for x in hit["TPlon"].values]
        if n == 1:
            st.pyplot(plot_simple(
                images[0], times[0], lats[0], lons[0], ch,
                nstd=q["nstd"], vmin=q["vmin"], vmax=q["vmax"],
            ))
        else:
            metas = list(zip(times, lats, lons))
            with st.spinner(f"Rendering {ch} movie ({n} frames)..."):
                data, fmt = make_animation_gif(
                    images, metas, ch, fps=q["fps"],
                    nstd=q["nstd"], vmin=q["vmin"], vmax=q["vmax"],
                )
            if fmt == "mp4":
                st.video(data)
            else:
                st.image(data)


def render_orbit_view(query_out, q):
    timeline = query_out["timeline"]
    orbit = query_out["orbit"]
    n = len(timeline)
    st.subheader(f"Orbit view  —  {n} frame(s)")
    frames = build_orbit_frames(timeline, orbit)
    with st.spinner(f"Rendering orbit view movie ({n} frames)..."):
        data, fmt = render_orbit_view_movie(frames, q, fps=q["fps"])
    if fmt == "mp4":
        st.video(data)
    else:
        st.image(data)


def render_downloads(primary, q):
    st.markdown("**Downloads (primary channels)**")
    for ch, hit in primary.items():
        data = dataset_to_netcdf_bytes(hit)
        st.download_button(
            f"Download {ch} ({hit.sizes['time']} imgs) as netCDF",
            data=data,
            file_name=(f"MATS_L1b_{ch}_"
                       f"{q['t_start']:%Y%m%dT%H%M%S}_"
                       f"{q['t_end']:%Y%m%dT%H%M%S}.nc"),
            mime="application/x-netcdf",
            key=f"dl_{ch}",
        )


def render_results(query_out, q):
    st.divider()
    primary = query_out["primary"]
    if not primary:
        return
    if q["plot_mode"].startswith("orbit view"):
        render_orbit_view(query_out, q)
    else:
        render_simple(primary, q)
    st.divider()
    render_downloads(primary, q)
    st.info(
        "Primary datasets are in `st.session_state['last_results']['primary']` "
        "as a `{channel: xarray.Dataset}` dict."
    )


def main():
    st.set_page_config(
        page_title="MATS L1b Quick-look",
        page_icon="🛰️",
        layout="wide",
    )
    st.title("🛰️ MATS L1b Quick-look")
    st.markdown(
        "MATS data is free to use but we encourage to contact the MATS team "
        "at MISU ([linda@misu.su.se](mailto:linda@misu.su.se)) for scientific use."
    )

    q = sidebar_inputs()

    ok = True
    if q["t_end"] <= q["t_start"]:
        st.error("End time must be after start time.")
        ok = False
    if not q["channels"]:
        st.warning("Select at least one channel.")
        ok = False

    b1, b2, b3 = st.sidebar.columns(3)
    with b1:
        preview_btn = st.button("Preview", type="primary",
                                use_container_width=True, disabled=not ok)
    with b2:
        plot_btn = st.button("Plot", use_container_width=True, disabled=not ok)
    with b3:
        clear = st.button("Clear", use_container_width=True)

    if clear:
        for k in ("last_preview", "last_preview_query", "last_results",
                  "last_query", "last_messages", "last_ran_at", "n_sub"):
            st.session_state.pop(k, None)
        st.rerun()

    if preview_btn and ok:
        for k in ("last_results", "last_messages"):
            st.session_state.pop(k, None)
        try:
            preview = run_preview(q)
        except Exception as exc:
            st.error(
                f"Failed to read index: {type(exc).__name__}: {exc}\n\n"
                "This usually means the Bolin S3 endpoint is unreachable."
            )
            return
        st.session_state["last_preview"] = preview
        st.session_state["last_preview_query"] = q

    preview = st.session_state.get("last_preview")
    preview_q = st.session_state.get("last_preview_query")

    if preview is not None:
        st.subheader("Preview")
        total = render_preview(preview, preview_q)
        n_sub = 1
        if total > 50:
            default_n = max(1, round(total / 50))
            n_sub = int(st.number_input(
                f"Total {total} images. Use every nth image:",
                min_value=1, value=default_n, step=1, key="n_sub",
            ))
            plotted = sum(len(p["inds"][::n_sub]) for p in preview.values())
            st.caption(f"Will plot {plotted} image(s) with n={n_sub}.")
        elif total > 0:
            st.caption(f"Will plot all {total} image(s).")

        if plot_btn and ok:
            subsampled = {
                ch: {"inds": info["inds"][::n_sub]}
                for ch, info in preview.items()
            }
            messages: list[tuple[str, str]] = []
            try:
                results = run_query(q, messages, preview=subsampled)
            except Exception as exc:
                st.error(
                    f"Failed to fetch data: {type(exc).__name__}: {exc}\n\n"
                    "This usually means the Bolin S3 endpoint is unreachable."
                )
                return
            st.session_state["last_results"] = results
            st.session_state["last_query"] = q
            st.session_state["last_messages"] = messages
            st.session_state["last_ran_at"] = datetime.now().strftime("%H:%M:%S")
    elif plot_btn and ok:
        st.warning("Click **Preview** first to inspect what data is available.")

    results = st.session_state.get("last_results")
    q_stored = st.session_state.get("last_query", q)
    messages = st.session_state.get("last_messages", [])
    ran_at = st.session_state.get("last_ran_at")

    if results is None:
        if preview is None:
            st.info("Set the query in the sidebar and click **Preview**. "
                    "(Scroll down if needed.)")
        return

    st.divider()
    st.caption(
        f"Last plot ran at {ran_at}  ·  "
        f"window {q_stored['t_start']:%Y-%m-%d %H:%M:%S} → "
        f"{q_stored['t_end']:%Y-%m-%d %H:%M:%S}  ·  "
        f"lat [{q_stored['lat0']:.1f}, {q_stored['lat1']:.1f}]  ·  "
        f"lon [{q_stored['lon0']:.1f}, {q_stored['lon1']:.1f}]"
    )
    for level, text in messages:
        getattr(st, level)(text)

    if not results.get("primary"):
        st.warning("No matching data found for this query.")
        return
    # Use live `q` so color/fps changes re-render without re-fetching.
    render_results(results, q)


if __name__ == "__main__":
    main()
