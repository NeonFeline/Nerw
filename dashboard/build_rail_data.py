#!/usr/bin/env python3
"""Pack the Polish rail GTFS feed into one script the rail map dashboard loads.

The page places every scheduled passenger train on its track from the timetable
alone, so it needs three things: track geometry, each trip's stop times as a
distance along that geometry, and the service days each trip runs on. Bus
replacement services (route_type 3) are dropped.

    uv run python dashboard/build_rail_data.py --gtfs polish_trains \
        --countries ne_10m_admin_0_countries.geojson --out dashboard/rail_data.js

The GTFS feed is Mikołaj Kuranowski's build of the PKP PLK open timetable
(https://mkuran.pl/gtfs/, polish_trains.zip); unzip it into polish_trains/.
Without --countries the Natural Earth 1:10m country file is downloaded.

Geometry is quantised to 1e-5 degrees and delta-encoded. Distances along a
shape are measured on that quantised geometry with the equirectangular formula
in `cumulative_metres`; the page repeats the same formula, so stop positions
written here land on the same vertices there.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
    "geojson/ne_10m_admin_0_countries.geojson"
)
COORD_SCALE = 1e5  # degrees -> integer units, ~1 m
M_PER_DEG = 111_320.0
VIEW_BBOX = (11.0, 47.0, 27.5, 57.0)  # lon_min, lat_min, lon_max, lat_max

# Train classes, in the order the page assigns colours.
CLASSES = ["long_distance", "regional", "suburban"]
LONG_DISTANCE_AGENCIES = {"IC", "LEO", "RJ"}
SUBURBAN_ROUTE = re.compile(r"^(SKM_|SKMT_|KML_SKA|PR_SKA|KW_PKM|PR_PKM|LKA)")
# plk_train_name often carries a line code ("S1", "PKM2", "F7/D18") rather than a name.
LINE_CODE = re.compile(r"^[A-ZŁŚ]{1,4} ?\d{1,3}[A-Z]?(/[A-ZŁŚ]{0,4} ?\d{0,3}[A-Z]?)*$")


def read_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def gtfs_seconds(hms: str) -> int:
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def simplify(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker keep-mask, iterative so long shapes cannot hit the recursion limit."""
    n = len(x)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        dx, dy = x[b] - x[a], y[b] - y[a]
        px, py = x[a + 1 : b] - x[a], y[a + 1 : b] - y[a]
        l2 = dx * dx + dy * dy
        if l2 > 0:
            t = np.clip((px * dx + py * dy) / l2, 0.0, 1.0)
            d = np.hypot(px - t * dx, py - t * dy)
        else:
            d = np.hypot(px, py)
        i = int(np.argmax(d))
        if d[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return keep


def to_metres(lat: np.ndarray, lon: np.ndarray):
    k = np.cos(np.radians(lat.mean()))
    return lon * M_PER_DEG * k, lat * M_PER_DEG


def cumulative_metres(qlat: np.ndarray, qlon: np.ndarray) -> np.ndarray:
    lat, lon = qlat / COORD_SCALE, qlon / COORD_SCALE
    mid = np.radians((lat[1:] + lat[:-1]) / 2)
    seg = np.hypot(np.diff(lat) * M_PER_DEG, np.diff(lon) * M_PER_DEG * np.cos(mid))
    return np.concatenate([[0.0], np.cumsum(seg)])


def delta_encode(qlat: np.ndarray, qlon: np.ndarray) -> list[int]:
    q = np.stack([qlat, qlon], axis=1)
    return np.diff(q, axis=0, prepend=np.zeros((1, 2), dtype=q.dtype)).ravel().tolist()


def train_class(route: dict) -> int:
    if route["agency_id"] in LONG_DISTANCE_AGENCIES:
        return CLASSES.index("long_distance")
    if SUBURBAN_ROUTE.match(route["route_id"]):
        return CLASSES.index("suburban")
    return CLASSES.index("regional")


def load_shapes(path: Path, wanted: set[str], tol: float):
    """shape_id -> (encoded coords, original km at kept vertices, metres at kept vertices)."""
    pts = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            sid = row[col["shape_id"]]
            if sid in wanted:
                pts[sid].append(
                    (
                        int(row[col["shape_pt_sequence"]]),
                        float(row[col["shape_pt_lat"]]),
                        float(row[col["shape_pt_lon"]]),
                        float(row[col["shape_dist_traveled"]] or "nan"),
                    )
                )
    shapes = {}
    for sid, rows in pts.items():
        arr = np.array(sorted(rows))
        lat, lon, km = arr[:, 1], arr[:, 2], arr[:, 3]
        if np.isnan(km).any():
            x, y = to_metres(lat, lon)
            km = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))]) / 1000
        keep = simplify(*to_metres(lat, lon), tol)
        qlat = np.round(lat[keep] * COORD_SCALE).astype(np.int64)
        qlon = np.round(lon[keep] * COORD_SCALE).astype(np.int64)
        shapes[sid] = (
            delta_encode(qlat, qlon),
            np.maximum.accumulate(km[keep]),
            cumulative_metres(qlat, qlon),
        )
    return shapes


def load_countries(path: Path | None, tol: float):
    if path is None:
        print(f"downloading {NATURAL_EARTH_URL}")
        with urllib.request.urlopen(NATURAL_EARTH_URL, timeout=120) as resp:
            geo = json.load(resp)
    else:
        geo = json.loads(path.read_text(encoding="utf-8"))
    x0, y0, x1, y1 = VIEW_BBOX
    out = []
    for feat in geo["features"]:
        props, geom = feat["properties"], feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        rings, inside = [], []
        for poly in polys:
            outer = np.asarray(poly[0])
            if outer[:, 0].max() < x0 or outer[:, 0].min() > x1 or outer[:, 1].max() < y0 or outer[:, 1].min() > y1:
                continue
            for ring in poly:
                ring = np.asarray(ring)
                keep = simplify(*to_metres(ring[:, 1], ring[:, 0]), tol)
                if keep.sum() < 4:
                    continue
                qlat = np.round(ring[keep, 1] * COORD_SCALE).astype(np.int64)
                qlon = np.round(ring[keep, 0] * COORD_SCALE).astype(np.int64)
                rings.append(delta_encode(qlat, qlon))
                m = (ring[:, 0] >= x0) & (ring[:, 0] <= x1) & (ring[:, 1] >= y0) & (ring[:, 1] <= y1)
                inside.append(ring[m])
        if not rings:
            continue
        lx, ly = props["LABEL_X"], props["LABEL_Y"]
        if not (x0 + 1 <= lx <= x1 - 1 and y0 + 0.5 <= ly <= y1 - 0.5):
            pts = np.concatenate(inside)
            lx, ly = (pts.mean(axis=0).tolist() if len(pts) else (None, None))
        out.append(
            {
                "iso": props["ADM0_A3"],
                "name": props.get("NAME_PL") or props["NAME_EN"],
                "label": None if lx is None else [round(ly, 3), round(lx, 3)],
                "rings": rings,
            }
        )
    return out


def build(args) -> dict:
    gtfs = Path(args.gtfs)
    feed = next(read_csv(gtfs / "feed_info.txt"))
    agencies = list(read_csv(gtfs / "agency.txt"))
    agency_index = {a["agency_id"]: i for i, a in enumerate(agencies)}

    routes = {r["route_id"]: r for r in read_csv(gtfs / "routes.txt") if r["route_type"] == "2"}
    route_ids = sorted(routes)
    route_index = {rid: i for i, rid in enumerate(route_ids)}

    # Service days, from the day before the feed starts (after-midnight runs of
    # that day's trains are still on the rails when the feed begins).
    base = date.fromisoformat(feed["feed_start_date"][:4] + "-" + feed["feed_start_date"][4:6] + "-" + feed["feed_start_date"][6:]) - timedelta(days=1)
    service_days: dict[str, set[int]] = defaultdict(set)
    for row in read_csv(gtfs / "calendar_dates.txt"):
        if row["exception_type"] != "1":
            continue
        d = row["date"]
        offset = (date(int(d[:4]), int(d[4:6]), int(d[6:])) - base).days
        if offset >= 0:
            service_days[row["service_id"]].add(offset)
    n_days = max(max(v) for v in service_days.values()) + 1

    trips = [
        t
        for t in read_csv(gtfs / "trips.txt")
        if t["route_id"] in routes and t["shape_id"] and service_days.get(t["service_id"])
    ]
    trip_by_id = {t["trip_id"]: t for t in trips}

    stop_rows = defaultdict(list)
    for row in read_csv(gtfs / "stop_times.txt"):
        if row["trip_id"] in trip_by_id:
            stop_rows[row["trip_id"]].append(row)

    stops = {s["stop_id"]: s for s in read_csv(gtfs / "stops.txt")}
    station_index: dict[str, int] = {}
    stations = []

    def station_of(stop_id: str) -> int:
        stop = stops[stop_id]
        sid = stop["parent_station"] or stop_id
        if sid not in station_index:
            st = stops[sid]
            station_index[sid] = len(stations)
            stations.append(
                {
                    "name": st["stop_name"],
                    "lat": round(float(st["stop_lat"]) * COORD_SCALE),
                    "lon": round(float(st["stop_lon"]) * COORD_SCALE),
                    "calls": 0,
                }
            )
        return station_index[sid]

    shape_ids = sorted({t["shape_id"] for t in trips})
    shape_index = {sid: i for i, sid in enumerate(shape_ids)}
    shapes = load_shapes(gtfs / "shapes.txt", set(shape_ids), args.tolerance)

    services = sorted({t["service_id"] for t in trips})
    service_index = {s: i for i, s in enumerate(services)}

    patterns: dict[tuple, int] = {}
    cols = {k: [] for k in ("pattern", "start", "route", "service", "category", "number", "name")}
    overshoot = 0
    for t in trips:
        rows = sorted(stop_rows[t["trip_id"]], key=lambda r: int(r["stop_sequence"]))
        if len(rows) < 2:
            continue
        _, km_keep, m_keep = shapes[t["shape_id"]]
        km = np.array([float(r["shape_dist_traveled"]) for r in rows])
        overshoot += int((km > km_keep[-1] + 0.05).sum())
        metres = np.round(np.interp(km, km_keep, m_keep)).astype(int)
        metres = np.maximum.accumulate(metres)
        arr = [gtfs_seconds(r["arrival_time"] or r["departure_time"]) for r in rows]
        dep = [gtfs_seconds(r["departure_time"] or r["arrival_time"]) for r in rows]
        start = arr[0]
        flat = []
        prev_m, prev_dep = 0, start
        for r, m, a, d in zip(rows, metres.tolist(), arr, dep):
            st = station_of(r["stop_id"])
            stations[st]["calls"] += 1
            # station, metres since previous stop, running time, dwell
            flat += [st, m - prev_m, max(a - prev_dep, 0), max(d - a, 0)]
            prev_m, prev_dep = m, max(d, a)
        key = (shape_index[t["shape_id"]], tuple(flat))
        if key not in patterns:
            patterns[key] = len(patterns)
        name = t["plk_train_name"].strip()
        route = routes[t["route_id"]]
        if name == route["route_short_name"] or LINE_CODE.match(name):
            name = ""
        cols["pattern"].append(patterns[key])
        cols["start"].append(start)
        cols["route"].append(route_index[t["route_id"]])
        cols["service"].append(service_index[t["service_id"]])
        cols["category"].append(t["plk_category_code"] or route["route_short_name"])
        cols["number"].append(t["plk_train_number"] or t["trip_short_name"].split(" ")[0])
        cols["name"].append(name)

    # Every GTFS time here is on a whole minute; ship minutes to save bytes.
    unit = 60 if all(v % 60 == 0 for key in patterns for v in key[1][2::4] + key[1][3::4]) and all(
        s % 60 == 0 for s in cols["start"]
    ) else 1
    pattern_list = [None] * len(patterns)
    for (shape, flat), i in patterns.items():
        enc = list(flat)
        if unit != 1:
            for j in range(2, len(enc), 4):
                enc[j] //= unit
                enc[j + 1] //= unit
        pattern_list[i] = [shape, enc]
    cols["start"] = [s // unit for s in cols["start"]]

    print(
        f"{len(cols['pattern'])} trips, {len(patterns)} stop patterns, {len(shape_ids)} shapes, "
        f"{sum(len(shapes[s][0]) // 2 for s in shape_ids)} vertices, {len(stations)} stations, "
        f"{len(services)} services over {n_days} days; {overshoot} stops past their shape end"
    )

    return {
        "feed": {
            "version": feed["feed_version"],
            "start": feed["feed_start_date"],
            "end": feed["feed_end_date"],
            "publisher": feed["feed_publisher_name"],
            "baseDate": base.isoformat(),
            "days": n_days,
        },
        "coordScale": COORD_SCALE,
        "timeUnit": unit,
        "classes": CLASSES,
        "agencies": [{"id": a["agency_id"], "name": a["agency_name"]} for a in agencies],
        "routes": [
            {
                "short": routes[r]["route_short_name"],
                "long": routes[r]["route_long_name"],
                "agency": agency_index[routes[r]["agency_id"]],
                "cls": train_class(routes[r]),
            }
            for r in route_ids
        ],
        "services": ["".join("1" if d in service_days[s] else "0" for d in range(n_days)) for s in services],
        "shapes": [shapes[s][0] for s in shape_ids],
        "patterns": pattern_list,
        "stations": stations,
        "trips": cols,
        "countries": load_countries(Path(args.countries) if args.countries else None, args.border_tolerance),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gtfs", default="polish_trains", help="directory holding the GTFS .txt files")
    ap.add_argument("--countries", help="Natural Earth admin-0 countries GeoJSON (downloaded when omitted)")
    ap.add_argument("--out", default="dashboard/rail_data.js")
    ap.add_argument("--tolerance", type=float, default=30.0, help="track simplification tolerance, metres")
    ap.add_argument("--border-tolerance", type=float, default=250.0, help="border simplification tolerance, metres")
    args = ap.parse_args()

    data = build(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    out.write_text(f"window.RAIL_DATA={payload};\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
