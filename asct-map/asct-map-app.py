"""
Interactive map of the ASTC Travel Passport Program's US museums.

Reads astc_museums_202609.tsv (see parse_astc_pdf.py) and geocodes each museum
through three tiers, falling through only when the previous one misses:

1. US Census Bureau batch geocoder (free, no key) - matches a full street
   address against actual TIGER road-range data. ~87% hit rate; misses
   institutional addresses that aren't in that range data (campus/building
   names, PO boxes, highway-exit addresses, etc).
2. Photon (free, no key, OpenStreetMap-based) - queried by museum name +
   city + state first, since named institutions are usually tagged as
   their own point-of-interest in OSM even when Census has no road-range
   entry for their street. A result is only kept if it lands within ~100
   miles of the zip-code centroid (below), as a sanity check against a
   same-named place in the wrong state.
3. pgeocode zip-code centroid (offline, always succeeds) - last resort.

Results are cached to geocode_cache_202609.csv next to this script so the
network calls only happen once.

Renders a Folium/Leaflet map (street or satellite tiles, no API key)
centered on the US: click a museum icon to open a popup, right at that
marker, with its name, address, city/state, phone number, and
requirements.

Run:
    uv run streamlit run asct-map/asct-map-app.py
"""

import csv
import html
import io
import re
import time
from pathlib import Path

import folium
import pandas as pd
import pgeocode
import requests
import streamlit as st
from streamlit_folium import st_folium

SCRIPT_DIR = Path(__file__).parent
TSV_PATH = SCRIPT_DIR / "astc_museums_202609.tsv"
GEOCODE_CACHE_PATH = SCRIPT_DIR / "geocode_cache_202609.csv"
CENSUS_BATCH_URL = "https://geocoding.geo.census.gov/geocoder/locations/addressbatch"
PHOTON_URL = "https://photon.komoot.io/api/"
PHOTON_SANITY_RADIUS_DEG = 1.5  # ~100 miles; reject a Photon hit further than this from the zip centroid

# Geographic center of the contiguous US - fixed so the map always opens
# framing the country, instead of auto-centering on the mean lat/lon of
# whatever's currently filtered (which drifts out into the ocean once
# AK/HI/PR are averaged in with the mainland).
USA_CENTER = [39.8283, -98.5795]
USA_ZOOM = 4

# A couple of addresses have no zip code at all (see parse_astc_pdf.md,
# "genuine data gaps") - hardcode a representative zip so they still plot.
MANUAL_ZIP_FALLBACK = {
    "Children's Creativity Museum": "94103",  # San Francisco, CA
    "EcoExploratorio": "00901",  # San Juan, PR
}

ZIP_RE = re.compile(r"(\d{5})(?!\d)")


def extract_zip(row):
    matches = ZIP_RE.findall(row["address"])
    if matches:
        return matches[-1]
    return MANUAL_ZIP_FALLBACK.get(row["museum_name"])


@st.cache_resource
def get_zip_geocoders():
    return {"us": pgeocode.Nominatim("us"), "pr": pgeocode.Nominatim("pr")}


def load_geocode_cache():
    if GEOCODE_CACHE_PATH.exists():
        cached = pd.read_csv(GEOCODE_CACHE_PATH, dtype=str, keep_default_na=False)
        return {
            r["address"]: (float(r["lat"]), float(r["lon"]), r["precision"])
            for _, r in cached.iterrows()
        }
    return {}


def save_geocode_cache(cache):
    pd.DataFrame(
        [{"address": addr, "lat": lat, "lon": lon, "precision": prec} for addr, (lat, lon, prec) in cache.items()]
    ).to_csv(GEOCODE_CACHE_PATH, index=False)


def geocode_via_census(addresses):
    """Batch, address-level geocoding (free, no API key). Returns
    {address: (lat, lon)} only for addresses Census's road-range data
    could actually match - callers must handle the rest themselves."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    for i, addr in enumerate(addresses):
        writer.writerow([i, addr, "", "", ""])

    resp = requests.post(
        CENSUS_BATCH_URL,
        files={"addressFile": ("addresses.csv", buf.getvalue().encode("utf-8"), "text/csv")},
        data={"benchmark": "Public_AR_Current"},
        timeout=180,
    )
    resp.raise_for_status()

    matched = {}
    for fields in csv.reader(io.StringIO(resp.text)):
        if len(fields) >= 6 and fields[2] == "Match":
            lon, lat = fields[5].split(",")
            matched[addresses[int(fields[0])]] = (float(lat), float(lon))
    return matched


def geocode_via_zip(rows):
    """Fallback: zip-code centroid via pgeocode (offline, always succeeds
    for a row with a zip). Returns {address: (lat, lon)}."""
    geocoders = get_zip_geocoders()
    result = {}
    zips = rows.copy()
    zips["_zip"] = zips.apply(extract_zip, axis=1)
    zips["_country"] = zips["state"].apply(lambda s: "pr" if s == "PR" else "us")
    for country, group in zips.groupby("_country"):
        coords = geocoders[country].query_postal_code(group["_zip"].tolist())
        for addr, lat, lon in zip(group["address"], coords["latitude"], coords["longitude"]):
            if pd.notna(lat) and pd.notna(lon):
                result[addr] = (lat, lon)
    return result


# Photon 403s a bare "python-requests" User-Agent - needs something
# identifying, same courtesy Nominatim's usage policy asks for.
PHOTON_HEADERS = {"User-Agent": "astc-museums-dashboard/1.0 (personal streamlit project)"}
# osm_key values that mean "this is a real place/venue", worth preferring
# over an incidental street/highway/building-shell match.
PHOTON_VENUE_KEYS = {"tourism", "amenity", "leisure"}


def _photon_query(q):
    try:
        resp = requests.get(PHOTON_URL, params={"q": q, "limit": 1}, headers=PHOTON_HEADERS, timeout=10)
        resp.raise_for_status()
        features = resp.json().get("features", [])
    except (requests.RequestException, ValueError):
        return None
    if not features:
        return None
    props = features[0]["properties"]
    lon, lat = features[0]["geometry"]["coordinates"]
    return lat, lon, props.get("osm_key")


def geocode_via_photon(rows, zip_centroids):
    """Second-tier fallback for addresses Census's road-range data missed.
    Tries the raw address text and a "name + city + state" query (OSM's
    tagged name for an institution sometimes differs from ours, so
    neither query alone is reliable) and prefers whichever result is
    tagged as an actual venue (tourism/amenity/leisure) over an
    incidental street/building match. Rejects anything that lands
    implausibly far from the zip-code centroid (a same-named place in
    the wrong state), since a wrong precise-looking pin is worse than an
    honest approximate one. Returns {address: (lat, lon)}."""
    result = {}
    for _, row in rows.iterrows():
        centroid = zip_centroids.get(row["address"])
        candidates = []
        for query in (row["address"], f"{row['museum_name']} {row['city']} {row['state']}"):
            hit = _photon_query(query)
            time.sleep(0.3)
            if hit is None:
                continue
            lat, lon, osm_key = hit
            if centroid is not None:
                if abs(lat - centroid[0]) > PHOTON_SANITY_RADIUS_DEG or abs(lon - centroid[1]) > PHOTON_SANITY_RADIUS_DEG:
                    continue
            candidates.append((lat, lon, osm_key))
        if not candidates:
            continue
        lat, lon, _ = next((c for c in candidates if c[2] in PHOTON_VENUE_KEYS), candidates[0])
        result[row["address"]] = (lat, lon)
    return result


@st.cache_data
def load_museums():
    df = pd.read_csv(TSV_PATH, sep="\t", dtype=str, keep_default_na=False)

    cache = load_geocode_cache()
    to_geocode = [a for a in df["address"].unique() if a not in cache]
    if to_geocode:
        for addr, (lat, lon) in geocode_via_census(to_geocode).items():
            cache[addr] = (lat, lon, "address")

        remaining = df[df["address"].isin([a for a in to_geocode if a not in cache])]
        if len(remaining):
            zip_hits = geocode_via_zip(remaining)
            for addr, (lat, lon) in geocode_via_photon(remaining, zip_hits).items():
                cache[addr] = (lat, lon, "poi")
            for addr, (lat, lon) in zip_hits.items():
                if addr not in cache:
                    cache[addr] = (lat, lon, "zip")

        save_geocode_cache(cache)

    df["lat"] = df["address"].map(lambda a: cache[a][0] if a in cache else float("nan"))
    df["lon"] = df["address"].map(lambda a: cache[a][1] if a in cache else float("nan"))
    df["geocode_precision"] = df["address"].map(lambda a: cache[a][2] if a in cache else "")
    return df


def popup_html(row):
    parts = [
        f"<div style='font-family:sans-serif;min-width:220px'>",
        f"<b>{html.escape(row['museum_name'])}</b><br>",
        f"📍 {html.escape(row['address'])}<br>",
        f"🏙️ {html.escape(row['city'])}, {html.escape(row['state'])}<br>",
        f"☎️ {html.escape(row['phone_number']) if row['phone_number'] else '—'}<br>",
    ]
    if row["requirements"]:
        parts.append(f"⚠️ {html.escape(row['requirements'])}<br>")
    if row["geocode_precision"] == "zip":
        parts.append("<i>Location approximate (zip-code center)</i>")
    parts.append("</div>")
    return "".join(parts)


def build_map(rows):
    m = folium.Map(location=USA_CENTER, zoom_start=USA_ZOOM, tiles=None)
    # CARTO Voyager: roads, state/country borders, city labels/dots baked
    # into the tiles themselves - closest free look-and-feel to Google Maps'
    # default street view (plain OpenStreetMap tiles read much flatter/barer).
    folium.TileLayer(
        tiles="https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name="Street",
        max_zoom=20,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri, Maxar, Earthstar Geographics",
        name="Satellite",
    ).add_to(m)
    # Labels/roads/borders overlay, so Satellite mode can also read like
    # Google's "hybrid" view instead of bare, unlabeled imagery. Off by
    # default (Street already has its own labels from Voyager) - toggle it
    # on from the layer switcher when using Satellite.
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Labels & roads (for Satellite)",
        overlay=True,
        show=False,
    ).add_to(m)

    for _, row in rows.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            tooltip=row["museum_name"],
            popup=folium.Popup(popup_html(row), max_width=300),
            icon=folium.Icon(icon="university", prefix="fa", color="red"),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


st.set_page_config(page_title="ASTC Museums Map", page_icon="🎫", layout="wide")

# Global look: soft gradient background, floating white sidebar/card panels,
# a gradient hero header, and pill-shaped stat chips - a more modern take on
# astc.org's red/teal brand (colors still sampled from their live CSS)
# layered with a cleaner Poppins/Inter font pairing. Selectors are Streamlit's
# stable data-testid hooks, plus the auto-added `.st-key-<key>` class for the
# map (see st_folium(..., key="museum_map") below).
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #F4F5FA 0%, #FAFAFC 45%, #FFFFFF 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        box-shadow: 2px 0 24px rgba(20, 20, 43, 0.06);
    }
    [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
    [data-testid="stExpander"] {
        border: none;
        box-shadow: 0 2px 16px rgba(20, 20, 43, 0.06);
        border-radius: 1rem;
    }
    .st-key-museum_map iframe {
        border-radius: 1.25rem;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(20, 20, 43, 0.10);
    }
    .astc-hero {
        background: linear-gradient(120deg, #EE2D33 0%, #C81E45 60%, #7E2757 100%);
        border-radius: 1.5rem;
        padding: 2rem 2.25rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 12px 32px rgba(238, 45, 51, 0.22);
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .astc-hero-badge {
        width: 60px; height: 60px; min-width: 60px;
        border-radius: 50%;
        background: rgba(255,255,255,0.18);
        display: flex; align-items: center; justify-content: center;
        font-size: 28px;
    }
    .astc-hero-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.1rem;
        color: #FFFFFF;
        line-height: 1.15;
    }
    .astc-hero-subtitle {
        font-family: 'Inter', sans-serif;
        color: rgba(255,255,255,0.85);
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .astc-chip-row { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.5rem; }
    .astc-chip {
        background: #FFFFFF;
        border-radius: 999px;
        padding: 0.45rem 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        color: #26272B;
        box-shadow: 0 2px 10px rgba(20, 20, 43, 0.07);
    }
    .astc-chip b { color: #EE2D33; font-weight: 700; }
    </style>

    <div class="astc-hero">
        <div class="astc-hero-badge">🎫</div>
        <div>
            <div class="astc-hero-title">ASTC Travel Passport Program</div>
            <div class="astc-hero-subtitle">US Member Museums Map</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

df = load_museums()
geocoded = df.dropna(subset=["lat", "lon"])
missing = len(df) - len(geocoded)
n_address = (geocoded["geocode_precision"] == "address").sum()
n_poi = (geocoded["geocode_precision"] == "poi").sum()
n_zip = (geocoded["geocode_precision"] == "zip").sum()
st.markdown(
    f"""
    <div class="astc-chip-row">
        <div class="astc-chip">🏛️ <b>{len(df)}</b> museums</div>
        <div class="astc-chip">📍 <b>{n_address}</b> exact address</div>
        <div class="astc-chip">🗺️ <b>{n_poi}</b> OpenStreetMap listing</div>
        <div class="astc-chip">〰️ <b>{n_zip}</b> zip-code approx.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if missing:
    st.caption(f"{missing} could not be geocoded at all and are omitted from the map.")

with st.sidebar:
    st.markdown("## 🔎 Filters")
    picked_states = st.multiselect(
        "State", sorted(geocoded["state"].unique()), placeholder="All states"
    )
    name_query = st.text_input("Search by name", placeholder="e.g. Science Center")
    proof_only = st.checkbox("Proof of residence required only")

filtered = geocoded
if picked_states:
    filtered = filtered[filtered["state"].isin(picked_states)]
if name_query:
    filtered = filtered[filtered["museum_name"].str.contains(name_query, case=False, na=False)]
if proof_only:
    filtered = filtered[filtered["requirements"] != ""]

with st.sidebar:
    st.divider()
    st.metric("Museums shown", len(filtered), f"of {len(geocoded)} total")

if filtered.empty:
    st.warning("No museums match the current filters.")
    st.stop()

st.caption(
    "Click a museum icon for its details. Use the layer switcher (top right of the map) "
    "for satellite view, and to overlay roads/labels on top of it."
)
st_folium(
    build_map(filtered),
    height=650,
    use_container_width=True,
    returned_objects=[],
    key="museum_map",
)

st.divider()

with st.expander(f"All {len(filtered)} museums (table)"):
    st.dataframe(
        filtered[["museum_name", "address", "phone_number", "email", "website", "requirements"]],
        use_container_width=True,
        hide_index=True,
    )
