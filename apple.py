import os
import re
import time
import random

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
import folium
import streamlit as st
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim, Photon, ArcGIS
from geopy.point import Point as GeoPoint
from pyproj import Transformer
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping
from shapely.ops import unary_union, substring, transform

# set_page_config должен быть ПЕРВОЙ командой Streamlit
st.set_page_config(page_title="15-минутный Костанай", layout="wide", page_icon="🏙️")

# --- 0. КОНФИГУРАЦИЯ ---

WALK_SPEED_M_PER_MIN = 75       # ~4.5 км/ч
ISO_TIMES = (5, 10, 15)         # минуты
EDGE_BUFFER_M = 60              # буфер вокруг улиц
CLOSING_M = 120                 # "замыкание": заполняет промежутки между параллельными улицами
MAX_SNAP_DISTANCE_M = 300       # максимум от адреса до ближайшего узла сети
BOUNDS_MARGIN_DEG = 0.02        # запас вокруг графа (~2 км)
BIZ_RADIUS_M = 1200

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

GRAPH_FILE = os.path.join(BASE_DIR, "kostanay_full_graph.graphml")
POI_FILE = os.path.join(BASE_DIR, "kostanay_pois.geojson")

CATEGORY_ORDER = ["Магазины", "Медицина", "Школы", "Детсады", "Парки"]
WGS84 = "EPSG:4326"
KOSTANAY_UTM_CRS = "EPSG:32641"  # UTM zone 41N (60–66° в.д.) — Костанай ~63.6°

SHOP_TAGS = ['supermarket', 'convenience', 'mall', 'department_store', 'grocery']
MED_TAGS = ['pharmacy', 'clinic', 'hospital', 'doctors']

CATEGORY_RULES = {
    'Магазины': ('shop', SHOP_TAGS),
    'Медицина': ('amenity', MED_TAGS),
    'Школы': ('amenity', ['school']),
    'Детсады': ('amenity', ['kindergarten', 'childcare']),
    'Парки': ('leisure', ['park', 'garden']),
}
CAT_COLORS = {'Магазины': '#e67e22', 'Медицина': '#c0392b', 'Школы': '#2980b9',
              'Детсады': '#8e44ad', 'Парки': '#27ae60'}

COUNT_BUFFER_M = 100        # объект считается доступным, если он не дальше 100 м от достижимой улицы
DEDUP_ANY_M = 15            # объекты одной категории ближе 15 м — один и тот же объект
DEDUP_SAME_NAME_M = 100     # с одинаковым названием ближе 100 м — тоже один объект
ONLY_NAMED = False          # False — объекты без названия («—») тоже считаются

_TO_UTM = Transformer.from_crs(WGS84, KOSTANAY_UTM_CRS, always_xy=True)
_TO_WGS = Transformer.from_crs(KOSTANAY_UTM_CRS, WGS84, always_xy=True)


# --- 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def parse_edge_length(data):
    """Безопасное извлечение длины ребра (м)."""
    length = data.get('length', 100)
    if isinstance(length, (list, tuple)):
        try:
            return sum(float(x) for x in length)
        except (ValueError, TypeError):
            return 100.0
    try:
        return float(length)
    except (ValueError, TypeError):
        return 100.0


def fill_poly_holes(geometry):
    """Удаляет дыры в полигоне."""
    if geometry is None or geometry.is_empty:
        return geometry
    if geometry.geom_type == 'Polygon':
        return Polygon(geometry.exterior)
    if geometry.geom_type == 'MultiPolygon':
        return MultiPolygon([Polygon(p.exterior) for p in geometry.geoms if not p.is_empty])
    return geometry


def haversine_m(lat1, lon1, lat2, lon2):
    """Векторизованное расстояние в метрах."""
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def validate_coordinates(lat, lon, bounds):
    """Проверка, что точка лежит в пределах графа (+ небольшой запас)."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return False, "Координаты вне допустимого диапазона"
    minx, miny, maxx, maxy = bounds
    m = BOUNDS_MARGIN_DEG
    if not (minx - m <= lon <= maxx + m and miny - m <= lat <= maxy + m):
        return False, "Точка вне области анализа (Костанай)"
    return True, "OK"


# --- 2. ЗАГРУЗКА ДАННЫХ ---

@st.cache_resource(show_spinner="Загрузка дорожной сети...")
def load_graph():
    """
    Возвращает (bundle, error).
    bundle: граф + индекс узлов (WGS84 и UTM) для быстрой привязки и построения изохрон.
    """
    if not os.path.exists(GRAPH_FILE):
        return None, f"Файл {os.path.basename(GRAPH_FILE)} не найден в папке проекта."

    try:
        G = ox.load_graphml(GRAPH_FILE)
        G_undir = G.to_undirected() if G.is_directed() else G.copy()

        if len(G_undir.nodes) == 0:
            return None, "Граф пуст"

        # Крупнейшая связная компонента
        largest_cc = max(nx.connected_components(G_undir), key=len)
        G_clean = G_undir.subgraph(largest_cc).copy()

        # Длина (м) и время пешком (мин)
        for _, _, _, data in G_clean.edges(data=True, keys=True):
            length_m = parse_edge_length(data)
            data['length'] = length_m
            data['time'] = length_m / WALK_SPEED_M_PER_MIN

        # Индекс узлов
        ids = list(G_clean.nodes)
        lats = np.array([float(G_clean.nodes[n]['y']) for n in ids])
        lons = np.array([float(G_clean.nodes[n]['x']) for n in ids])
        xs, ys = _TO_UTM.transform(lons, lats)
        xs, ys = np.asarray(xs), np.asarray(ys)

        bundle = {
            'G': G_clean,
            'ids': ids,
            'lats': lats,
            'lons': lons,
            'xy': {n: (float(x), float(y)) for n, x, y in zip(ids, xs, ys)},
            'bounds': (float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max())),
            'n_components': nx.number_connected_components(G_undir),
            'n_nodes_total': len(G_undir.nodes),
        }
        return bundle, None

    except Exception as e:
        return None, f"Ошибка загрузки графа: {e}"


@st.cache_data(show_spinner="Загрузка объектов инфраструктуры...")
def load_pois():
    """Возвращает (GeoDataFrame, is_demo)."""
    if os.path.exists(POI_FILE):
        try:
            pois = gpd.read_file(POI_FILE)
            if not pois.empty:
                return pois.to_crs(WGS84), False
        except Exception:
            pass

    # Резервная генерация (ДЕМО-данные)
    rng = random.Random(42)
    data = []
    amenities = ['school', 'kindergarten', 'pharmacy', 'clinic', 'hospital']
    shops = ['supermarket', 'convenience']
    for i in range(60):
        lat = 53.2144 + rng.uniform(-0.04, 0.04)
        lon = 63.6246 + rng.uniform(-0.05, 0.05)
        data.append({
            'geometry': Point(lon, lat),
            'name': f"Объект {i+1}",
            'amenity': rng.choice(amenities) if i % 2 == 0 else None,
            'shop': rng.choice(shops) if i % 2 != 0 else None,
            'leisure': 'park' if i % 5 == 0 else None,
        })
    return gpd.GeoDataFrame(data, crs=WGS84), True


# --- 3. ГЕОКОДИРОВАНИЕ ---

def parse_coords(text):
    """'53.2144, 63.6246' -> (lat, lon) либо None."""
    m = re.match(r"^\s*(-?\d{1,3}[.,]\d+)\s*[,; ]\s*(-?\d{1,3}[.,]\d+)\s*$", text or "")
    if not m:
        return None
    a, b = (float(x.replace(",", ".")) for x in m.groups())
    return a, b


_CITY_RE = re.compile(r"\bКостанай\b|\bҚостанай\b|\bKostanay\b")


def _geo_diag():
    log = st.session_state.get('_geo_log')
    return f" [Диагностика: {'; '.join(log)}]" if log else ""


def geocode_address(address_str, bounds):
    """Геокодирование с запасными сервисами: Nominatim -> Photon -> ArcGIS."""
    if not address_str or len(address_str.strip()) < 3:
        return None

    cache = st.session_state.setdefault("_geo_cache", {})
    key = address_str.strip().lower()
    if key in cache:
        return cache[key]

    UA = "kostanay_15min_v13"
    minx, miny, maxx, maxy = bounds
    m = BOUNDS_MARGIN_DEG
    viewbox = [GeoPoint(maxy + m, minx - m), GeoPoint(miny - m, maxx + m)]
    center = GeoPoint((miny + maxy) / 2, (minx + maxx) / 2)

    street = house = None
    match = re.match(r"^(.*?)[\s,]+(\d+[\w/\-]*)$", address_str.strip())
    if match:
        street, house = match.group(1).strip(), match.group(2)

    log = []

    def found(lat, lon, name, approx):
        cache[key] = {"lat": lat, "lon": lon, "name": name, "approx": approx}
        st.session_state['_geo_log'] = None
        return cache[key]

    # 1. Nominatim
    attempts = []
    if house:
        attempts.append(({"street": f"{house} {street}", "city": "Костанай"}, True, False))
    attempts.append((f"{address_str}, Костанай", True, False))
    attempts.append((f"{address_str}, Kostanay, Kazakhstan", True, False))
    if street:
        attempts.append((f"{street}, Костанай", True, True))
    attempts.append((f"{address_str}, Костанай, Казахстан", False, False))
    if street:
        attempts.append((f"{street}, Костанай, Казахстан", False, True))

    nominatim = Nominatim(user_agent=UA, timeout=10)
    for i, (q, bounded, approx) in enumerate(attempts):
        if i:
            time.sleep(1.1)
        kwargs = dict(exactly_one=False, limit=5, language="ru", country_codes="kz")
        if bounded:
            kwargs.update(viewbox=viewbox, bounded=True)
        try:
            results = nominatim.geocode(q, **kwargs)
        except Exception as e:
            log.append(f"Nominatim: {type(e).__name__}")
            break
        for loc in results or []:
            if bounded:
                ok, _ = validate_coordinates(loc.latitude, loc.longitude, bounds)
            else:
                ok = bool(_CITY_RE.search(loc.address))
            if ok:
                return found(loc.latitude, loc.longitude, loc.address, approx)
    else:
        log.append("Nominatim: нет результатов")

    # 2–3. Запасные сервисы (Photon, ArcGIS)
    def photon_fn(q):
        res = Photon(user_agent=UA, timeout=10).geocode(
            q, exactly_one=False, limit=5, location_bias=center)
        return [(r.latitude, r.longitude, r.address) for r in (res or [])]

    def arcgis_fn(q):
        res = ArcGIS(timeout=10).geocode(f"{q}, Казахстан", exactly_one=False)
        return [(r.latitude, r.longitude, r.address) for r in (res or [])]

    texts = [(f"{address_str}, Костанай", False)]
    if street:
        texts.append((f"{street}, Костанай", True))

    for name, fn in (("Photon", photon_fn), ("ArcGIS", arcgis_fn)):
        for q, approx in texts:
            try:
                results = fn(q)
            except Exception as e:
                log.append(f"{name}: {type(e).__name__}")
                break
            for lat, lon, addr in results:
                if validate_coordinates(lat, lon, bounds)[0]:
                    return found(lat, lon, f"{addr} [{name}]", approx)
        else:
            log.append(f"{name}: нет результатов")

    st.session_state['_geo_log'] = log
    return None


@st.cache_data(show_spinner="Определение адреса...", ttl=3600)
def reverse_geocode(lat, lon):
    """Обратное геокодирование: Nominatim, затем Photon."""
    try:
        geolocator = Nominatim(user_agent="kostanay_15min_reverse_v13", timeout=5)
        location = geolocator.reverse((lat, lon), exactly_one=True, language="ru")
        if location and "address" in location.raw:
            addr = location.raw["address"]
            road = addr.get("road") or addr.get("street") or addr.get("pedestrian")
            house = addr.get("house_number")
            if road and house:
                return f"{road}, {house}"
            elif road:
                return f"{road}"
    except Exception:
        pass
    try:
        loc = Photon(user_agent="kostanay_15min_reverse_v13", timeout=5).reverse(
            (lat, lon), exactly_one=True)
        props = ((loc.raw or {}).get("properties", {})) if loc else {}
        road = props.get("street") or props.get("name")
        house = props.get("housenumber")
        if road and house:
            return f"{road}, {house}"
        if road:
            return f"{road}"
    except Exception:
        pass
    return f"точка ({lat:.4f}, {lon:.4f})"


# --- 4. ПРОСТРАНСТВЕННЫЙ АНАЛИЗ ---

def get_nearest_node(bundle, lat, lon):
    """Ближайший узел сети."""
    d = haversine_m(lat, lon, bundle['lats'], bundle['lons'])
    i = int(np.argmin(d))
    return bundle['ids'][i], (float(bundle['lats'][i]), float(bundle['lons'][i])), float(d[i])


def _edge_line_utm(data, u, v, node_xy):
    """Геометрия ребра в UTM, ориентированная от узла u."""
    pu, pv = node_xy[u], node_xy[v]
    geom = data.get('geometry')
    if geom is not None and hasattr(geom, 'coords'):
        try:
            line = transform(_TO_UTM.transform, geom)
            c = list(line.coords)
            if Point(c[0]).distance(Point(pu)) > Point(c[-1]).distance(Point(pu)):
                line = LineString(c[::-1])
            return line
        except Exception:
            pass
    return LineString([pu, pv])


def _collect_segments(G, node_xy, reached, t):
    """Сегменты улиц, достижимые за t минут."""
    segs, seen = [], set()
    for u, du in reached.items():
        for v, keydict in G[u].items():
            pair = frozenset((u, v))
            if pair in seen:
                continue
            seen.add(pair)

            data = min(keydict.values(), key=lambda d: d.get('length', 1e9))
            length = data['length']
            line = _edge_line_utm(data, u, v, node_xy)
            if line.length == 0:
                continue

            if v in reached:
                segs.append(line)
            else:
                remaining = (t - du) * WALK_SPEED_M_PER_MIN
                if remaining <= 0 or length <= 0:
                    continue
                frac = remaining / length
                segs.append(line if frac >= 1 else substring(line, 0, frac, normalized=True))
    return segs


def build_isochrones(bundle, center_node):
    """Изохроны 5/10/15 минут."""
    G, node_xy = bundle['G'], bundle['xy']
    polys = {}
    if center_node not in G:
        return polys, 0

    dist = nx.single_source_dijkstra_path_length(
        G, center_node, cutoff=max(ISO_TIMES), weight='time'
    )

    for t in sorted(ISO_TIMES, reverse=True):
        reached = {n: d for n, d in dist.items() if d <= t}
        segs = _collect_segments(G, node_xy, reached, t)
        if not segs:
            continue
        merged = unary_union(segs)
        if t == max(ISO_TIMES):
            polys['count'] = transform(_TO_WGS.transform, merged.buffer(COUNT_BUFFER_M))
        area = merged.buffer(EDGE_BUFFER_M)
        area = area.buffer(CLOSING_M).buffer(-CLOSING_M)
        area = fill_poly_holes(area)
        if area is None or area.is_empty:
            continue
        polys[t] = transform(_TO_WGS.transform, area)

    return polys, len(dist)


def _filter(pois, column, values):
    if column in pois.columns:
        return pois[pois[column].isin(values)]
    return pois.iloc[0:0]


def _poi_name(row):
    v = row.get('name')
    return str(v).strip() if v is not None and pd.notna(v) else ''


def named_only(gdf):
    """Фильтрует безымянные объекты, если ONLY_NAMED == True."""
    if not ONLY_NAMED or 'name' not in gdf.columns:
        return gdf
    names = gdf['name'].fillna('').astype(str).str.strip()
    return gdf[names != '']


def dedupe_pois(gdf):
    """Убирает дубли одного и того же объекта."""
    n = len(gdf)
    if n < 2:
        return gdf
    pts = gdf.geometry.representative_point()
    x, y = _TO_UTM.transform(pts.x.values, pts.y.values)
    x, y = np.asarray(x), np.asarray(y)
    if 'name' in gdf.columns:
        names = gdf['name'].fillna('').astype(str).str.strip().str.lower().values
    else:
        names = np.array([''] * n)

    keep = []
    for i in range(n):
        if keep:
            k = np.array(keep)
            dd = np.hypot(x[k] - x[i], y[k] - y[i])
            same = (names[k] == names[i]) & (names[i] != '')
            thr = np.where(same, DEDUP_SAME_NAME_M, DEDUP_ANY_M)
            if np.any(dd <= thr):
                continue
        keep.append(i)
    return gdf.iloc[keep]


def categorize(pois):
    """{категория: GeoDataFrame без дублей}"""
    return {cat: dedupe_pois(named_only(_filter(pois, col, vals)))
            for cat, (col, vals) in CATEGORY_RULES.items()}


def calculate_accessibility(pois_gdf, poly_dict):
    """Индекс доступности + список объектов."""
    empty = {cat: 0 for cat in CATEGORY_ORDER}
    cols = ['Категория', 'Название', 'Тип', 'lat', 'lon']
    if pois_gdf.empty or 15 not in poly_dict:
        return 0, empty, pd.DataFrame(columns=cols)

    zone = poly_dict.get('count', poly_dict[15])
    pts_all = pois_gdf.geometry.representative_point()
    inside = pois_gdf[pts_all.intersects(zone).values]
    cats = categorize(inside)

    targets = {'Магазины': 3, 'Медицина': 2, 'Школы': 1, 'Детсады': 1, 'Парки': 1}
    counts, scores, rows = {}, [], []
    for cat in CATEGORY_ORDER:
        g = cats[cat]
        counts[cat] = len(g)
        scores.append(min(1.0, len(g) / targets[cat]))
        for _, r in g.iterrows():
            pp = r.geometry.representative_point()
            rows.append({'Категория': cat, 'Название': _poi_name(r) or 'без названия',
                         'Тип': _poi_label(r), 'lat': pp.y, 'lon': pp.x})

    details = pd.DataFrame(rows, columns=cols)
    return round(sum(scores) / len(scores) * 100), counts, details


def get_business_recommendation(user_lat, user_lon, pois_gdf):
    """Анализ коммерческого потенциала."""
    if pois_gdf.empty:
        return "🏪 Продуктовый магазин или Аптека", "Нет данных об инфраструктуре", 0, {}

    pts = pois_gdf.geometry.representative_point()
    d = haversine_m(user_lat, user_lon, pts.y.values, pts.x.values)
    nearby = pois_gdf[d <= BIZ_RADIUS_M]

    def cnt(column, values):
        return len(dedupe_pois(named_only(_filter(nearby, column, values))))

    shops = cnt('shop', SHOP_TAGS)
    pharmacies = cnt('amenity', ['pharmacy'])
    clinics = cnt('amenity', ['clinic', 'hospital', 'doctors'])
    schools = cnt('amenity', ['school'])
    kindergartens = cnt('amenity', ['kindergarten', 'childcare'])

    recs = []
    if shops == 0:
        recs.append("🛒 Продуктовый магазин")
    elif shops < 3:
        recs.append("🛍️ Специализированная розница")
    if pharmacies == 0:
        recs.append("💊 Аптека")
    if kindergartens == 0:
        recs.append("👶 Детский сад")

    if not recs:
        main_rec, details = "☕ Кафе / Барбершоп", "Район обеспечен инфраструктурой"
    else:
        main_rec = recs[0]
        details = " | ".join(recs[1:]) if len(recs) > 1 else "Минимальная конкуренция"

    shop_score = min(35, shops * 3.5)
    med_score = min(25, (pharmacies + clinics) * 5)
    edu_score = min(25, (schools + kindergartens) * 6.25)
    balance_bonus = 15 if (shops > 0 and pharmacies > 0 and schools > 0) else 5
    biz_score = round(min(100, shop_score + med_score + edu_score + balance_bonus))

    counts = {'Магазины': shops, 'Медицина': pharmacies + clinics, 'Образование': schools + kindergartens}
    return main_rec, details, biz_score, counts


def analyze_location(address, bundle, pois_gdf):
    """Анализ локации."""
    bounds = bundle['bounds']
    max_snap = st.session_state.get('max_snap', MAX_SNAP_DISTANCE_M)
    approx = False

    coords = parse_coords(address)
    if coords:
        lat, lon = coords
        if not validate_coordinates(lat, lon, bounds)[0] and validate_coordinates(lon, lat, bounds)[0]:
            lat, lon = lon, lat
        found_name = "координаты заданы вручную"
    else:
        geo = geocode_address(address, bounds)
        if geo is None:
            return None, (f"Адрес «{address}» не найден геокодером. "
                          "Введите координаты «широта, долгота» или выберите точку на карте." + _geo_diag())
        lat, lon, found_name, approx = geo['lat'], geo['lon'], geo['name'], geo['approx']

    ok, msg = validate_coordinates(lat, lon, bounds)
    if not ok:
        return None, f"{msg}: точка ({lat:.5f}, {lon:.5f}) за пределами сети."

    node, node_ll, snap_dist = get_nearest_node(bundle, lat, lon)
    if snap_dist > max_snap:
        return None, (f"Ближайший узел сети в {snap_dist:.0f} м от точки ({lat:.5f}, {lon:.5f}), "
                      f"допустимо {max_snap} м.")

    polys, n_reached = build_isochrones(bundle, node)
    if 15 not in polys:
        return None, "Не удалось построить зоны доступности"

    score, counts, details = calculate_accessibility(pois_gdf, polys)
    return {
        'address': address, 'found_name': found_name, 'approx': approx,
        'lat': lat, 'lon': lon,
        'node_coords': node_ll, 'snap_dist': snap_dist,
        'poly_dict': polys, 'n_reached': n_reached,
        'score': score, 'counts': counts, 'details': details,
    }, None


def resolve_point(text, bundle):
    bounds = bundle['bounds']
    coords = parse_coords(text)
    if coords:
        lat, lon = coords
        if not validate_coordinates(lat, lon, bounds)[0] and validate_coordinates(lon, lat, bounds)[0]:
            lat, lon = lon, lat
    else:
        geo = geocode_address(text, bounds)
        if geo is None:
            return None, f"Адрес «{text}» не найден." + _geo_diag()
        lat, lon = geo['lat'], geo['lon']
    ok, msg = validate_coordinates(lat, lon, bounds)
    if not ok:
        return None, msg
    return (lat, lon), None


def _poi_label(row):
    for c in ('shop', 'amenity', 'leisure'):
        v = row.get(c)
        if v is not None and pd.notna(v):
            return f"{c}: {v}"
    return "объект"


# --- 5. ВИЗУАЛИЗАЦИЯ ---

def add_isochrones_to_map(m, poly_dict, addr_latlon, node_latlon, label="", colors=None):
    if colors is None:
        colors = {5: '#2ecc71', 10: '#f39c12', 15: '#e74c3c'}

    for t in (15, 10, 5):
        poly = poly_dict.get(t)
        if poly is None:
            continue
        feature = {"type": "Feature", "geometry": mapping(poly), "properties": {}}
        folium.GeoJson(
            feature,
            style_function=lambda x, col=colors[t]: {
                'fillColor': col, 'color': col, 'weight': 2, 'fillOpacity': 0.25},
            tooltip=f"{label}Зона {t} мин",
        ).add_to(m)

    folium.Marker(
        addr_latlon, popup=f"{label}Адрес",
        icon=folium.Icon(color="red", icon="home", prefix="fa"),
    ).add_to(m)
    folium.CircleMarker(
        node_latlon, radius=5, color="blue", fill=True, fill_opacity=0.8,
        popup=f"{label}Привязка к сети",
    ).add_to(m)
    folium.PolyLine([addr_latlon, node_latlon], color="blue", weight=2, dash_array="4").add_to(m)


def add_network_layer(m, bundle, lat, lon, radius_m=2000):
    G = bundle['G']
    d = haversine_m(lat, lon, bundle['lats'], bundle['lons'])
    near = {bundle['ids'][i] for i in np.where(d <= radius_m)[0]}
    lines = [
        [[float(G.nodes[u]['x']), float(G.nodes[u]['y'])],
         [float(G.nodes[v]['x']), float(G.nodes[v]['y'])]]
        for u, v in G.subgraph(near).edges()
    ]
    if lines:
        folium.GeoJson(
            {"type": "Feature", "properties": {},
             "geometry": {"type": "MultiLineString", "coordinates": lines}},
            style_function=lambda x: {'color': '#8e44ad', 'weight': 2, 'opacity': 0.9},
            tooltip="Дорожная сеть (граф)",
        ).add_to(m)
    minx, miny, maxx, maxy = bundle['bounds']
    folium.Rectangle([[miny, minx], [maxy, maxx]], color='black', weight=2, fill=False,
                     dash_array='8', tooltip="Границы загруженного графа").add_to(m)


def _fill(key, text):
    st.session_state[key] = text


def map_picker(targets, bundle):
    minx, miny, maxx, maxy = bundle['bounds']
    with st.expander("🗺️ Адрес не находится? Выберите точку кликом по карте"):
        m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=12,
                       tiles="OpenStreetMap")
        folium.Rectangle([[miny, minx], [maxy, maxx]], color='black', weight=1, fill=False,
                         dash_array='6').add_to(m)
        picked = st.session_state.get("_picked")
        if picked:
            folium.Marker(picked, icon=folium.Icon(color="green")).add_to(m)
        out = st_folium(m, height=420, width=1000, key="picker", returned_objects=["last_clicked"])
        click = (out or {}).get("last_clicked")
        if click:
            picked = [click["lat"], click["lng"]]
            st.session_state["_picked"] = picked
        if picked:
            txt = f"{picked[0]:.6f}, {picked[1]:.6f}"
            st.code(txt)
            cols = st.columns(len(targets))
            for col, (label, key) in zip(cols, targets.items()):
                col.button(f"Подставить в «{label}»", key=f"fill_{key}",
                           on_click=_fill, args=(key, txt))


def fit_map(m, poly_dicts):
    b = [p[15].bounds for p in poly_dicts if 15 in p]
    if b:
        m.fit_bounds([[min(x[1] for x in b), min(x[0] for x in b)],
                      [max(x[3] for x in b), max(x[2] for x in b)]])


# --- 6. ИНТЕРФЕЙС ---

st.title("🏙️ 15-минутный Костанай: GIS-Аналитика")

bundle, err_msg = load_graph()
if bundle is None:
    st.error(f"❌ {err_msg}")
    st.info("Убедитесь, что файл kostanay_full_graph.graphml лежит в папке проекта.")
    st.stop()

pois_gdf, is_demo_pois = load_pois()
if is_demo_pois:
    st.warning("⚠️ Файл kostanay_pois.geojson не найден — показаны ДЕМО-данные (случайные точки).")

st.sidebar.header("🎯 Режим работы")
mode = st.sidebar.radio("Выберите инструмент:", ["Анализ адреса", "Сравнение 2-х районов"])

with st.sidebar.expander("🔧 Диагностика графа"):
    G = bundle['G']
    st.write(f"Узлов: {len(G.nodes)} (всего в файле: {bundle['n_nodes_total']})")
    st.write(f"Рёбер: {len(G.edges)}")
    st.write(f"Компонент связности в файле: {bundle['n_components']}")
    st.write(f"Границы (lon/lat): {tuple(round(x, 4) for x in bundle['bounds'])}")

st.sidebar.slider("Макс. расстояние привязки к сети, м", 50, 1000, 300, 50, key="max_snap")
show_net = st.sidebar.checkbox("🛣️ Показать дорожную сеть и границы графа")
show_pois = st.sidebar.checkbox("📍 Показать посчитанные объекты на карте", value=True)

st.session_state.setdefault('last_analysis', None)
st.session_state.setdefault('comparison', None)


# --- РЕЖИМ 1 ---

if mode == "Анализ адреса":
    st.sidebar.subheader("Параметры")
    address_input = st.sidebar.text_input("Адрес или координаты («широта, долгота»):",
                                          "проспект Аль-Фараби 65", key="addr_main")

    st.sidebar.subheader("Моделирование бизнеса")
    enable_siting = st.sidebar.checkbox("Оценить точку под бизнес")
    biz_input = st.sidebar.text_input(
        "Точка бизнеса (адрес или «широта, долгота»). Пусто — использовать адрес выше:",
        "", key="addr_biz")

    map_picker({"Адрес": "addr_main", "Точка бизнеса": "addr_biz"}, bundle)

    if st.sidebar.button("Анализировать", type="primary"):
        with st.spinner("Анализ..."):
            result, error = analyze_location(address_input, bundle, pois_gdf)
        if error:
            st.session_state.last_analysis = None
            st.error(f"❌ {error}")
        else:
            st.session_state.last_analysis = result

    data = st.session_state.last_analysis
    if data:
        st.subheader(f"📍 {data['address']}")
        st.caption(f"Найдено геокодером: {data['found_name']} · "
                   f"привязка к сети: {data['snap_dist']:.0f} м · "
                   f"достижимых узлов за 15 мин: {data['n_reached']}")

        if data.get('approx'):
            st.warning("⚠️ Точный дом не найден в OSM — использован центр улицы. "
                       "Для точности введите координаты или выберите точку на карте.")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Индекс доступности", f"{data['score']} / 100")
        c2.metric("Магазины", data['counts'].get('Магазины', 0))
        c3.metric("Медицина", data['counts'].get('Медицина', 0))
        c4.metric("Образование", data['counts'].get('Школы', 0) + data['counts'].get('Детсады', 0))

        with st.expander("📋 Какие объекты посчитаны (сверьте с картой)"):
            st.caption(f"Дубли объединены; учитываются объекты не дальше "
                       f"{COUNT_BUFFER_M} м от улиц, достижимых за 15 минут.")
            st.dataframe(data['details'][['Категория', 'Название', 'Тип']],
                         use_container_width=True, hide_index=True)

        m = folium.Map(location=[data['lat'], data['lon']], zoom_start=14, tiles="OpenStreetMap")
        add_isochrones_to_map(m, data['poly_dict'], [data['lat'], data['lon']], data['node_coords'])
        fit_map(m, [data['poly_dict']])
        if show_pois:
            for _, r in data['details'].iterrows():
                folium.CircleMarker([r['lat'], r['lon']], radius=5, color=CAT_COLORS[r['Категория']],
                                    fill=True, fill_opacity=0.9,
                                    tooltip=f"{r['Категория']}: {r['Название']}").add_to(m)
        if show_net:
            add_network_layer(m, bundle, data['lat'], data['lon'])

        if enable_siting:
            st.markdown("---")
            if biz_input.strip():
                with st.spinner("Поиск точки бизнеса..."):
                    pt, perr = resolve_point(biz_input, bundle)
            else:
                pt, perr = (data['lat'], data['lon']), None

            if perr:
                st.error(f"❌ Точка бизнеса: {perr}")
            else:
                biz_lat, biz_lon = pt
                address_name = reverse_geocode(round(biz_lat, 5), round(biz_lon, 5))
                rec_title, rec_desc, biz_score, biz_counts = get_business_recommendation(
                    biz_lat, biz_lon, pois_gdf)

                st.subheader(f"🎯 Коммерческий потенциал: {address_name}")
                st.caption(f"Координаты точки: {biz_lat:.5f}, {biz_lon:.5f}")
                b1, b2, b3 = st.columns(3)
                b1.metric("Оценка локации", f"{biz_score} / 100")
                b2.metric("Магазинов (1.2 км)", biz_counts.get('Магазины', 0))
                b3.metric("Медучреждений (1.2 км)", biz_counts.get('Медицина', 0))

                st.success(f"**Рекомендуемый бизнес:** {rec_title}")
                st.caption(f"💡 {rec_desc}")

                in_zone = 15 in data['poly_dict'] and data['poly_dict'][15].contains(Point(biz_lon, biz_lat))
                if in_zone:
                    st.success("✅ Точка внутри 15-минутной зоны")
                else:
                    st.info("ℹ️ Точка вне 15-минутной зоны")

                folium.Circle([biz_lat, biz_lon], radius=BIZ_RADIUS_M, color="orange",
                              weight=2, fill=False, dash_array="6",
                              tooltip=f"Радиус анализа {BIZ_RADIUS_M} м").add_to(m)
                pts = pois_gdf.geometry.representative_point()
                d = haversine_m(biz_lat, biz_lon, pts.y.values, pts.x.values)
                for cat, g in categorize(pois_gdf[d <= BIZ_RADIUS_M]).items():
                    for _, row in g.iterrows():
                        pp = row.geometry.representative_point()
                        folium.CircleMarker(
                            [pp.y, pp.x], radius=4, color=CAT_COLORS[cat], fill=True, fill_opacity=0.9,
                            tooltip=f"{cat}: {_poi_name(row) or _poi_label(row)}").add_to(m)

                folium.Marker(
                    [biz_lat, biz_lon], popup=f"Бизнес: {address_name}",
                    icon=folium.Icon(color="orange", icon="star", prefix="fa"),
                ).add_to(m)

        st_folium(m, width=1200, height=550, key="main_map", returned_objects=[])
    else:
        st.info("👈 Введите адрес и нажмите 'Анализировать'")


# --- РЕЖИМ 2 ---

else:
    st.subheader("📊 Сравнительный анализ двух локаций")

    col_a, col_b = st.columns(2)
    addr1 = col_a.text_input("Первый адрес или координаты:", "Алтынсарина 234", key="addr1")
    addr2 = col_b.text_input("Второй адрес или координаты:", "улица Маяковского 105", key="addr2")

    map_picker({"Первый адрес": "addr1", "Второй адрес": "addr2"}, bundle)

    if st.button("Сравнить", type="primary"):
        with st.spinner("Анализ..."):
            r1, e1 = analyze_location(addr1, bundle, pois_gdf)
            r2, e2 = analyze_location(addr2, bundle, pois_gdf) if not e1 else (None, None)
        if e1:
            st.session_state.comparison = None
            st.error(f"❌ {addr1}: {e1}")
        elif e2:
            st.session_state.comparison = None
            st.error(f"❌ {addr2}: {e2}")
        else:
            st.session_state.comparison = {'r1': r1, 'r2': r2}

    comp = st.session_state.comparison
    if comp:
        r1, r2 = comp['r1'], comp['r2']

        col_a, col_b = st.columns(2)
        col_a.metric(f"🏠 {r1['address']}", f"{r1['score']} / 100")
        col_b.metric(f"🏠 {r2['address']}", f"{r2['score']} / 100",
                     delta=f"{int(r2['score']) - int(r1['score']):+d}")
        col_a.caption(("⚠️ приблизительно (улица) · " if r1.get('approx') else "")
                      + f"Геокодер: {r1['found_name']} · привязка {r1['snap_dist']:.0f} м")
        col_b.caption(("⚠️ приблизительно (улица) · " if r2.get('approx') else "")
                      + f"Геокодер: {r2['found_name']} · привязка {r2['snap_dist']:.0f} м")

        df_comp = pd.DataFrame({
            "Категория": CATEGORY_ORDER,
            "Локация 1": [r1['counts'].get(c, 0) for c in CATEGORY_ORDER],
            "Локация 2": [r2['counts'].get(c, 0) for c in CATEGORY_ORDER],
        })
        st.table(df_comp)

        with st.expander("📋 Какие объекты посчитаны"):
            ca, cb = st.columns(2)
            ca.caption("Локация 1")
            ca.dataframe(r1['details'][['Категория', 'Название', 'Тип']],
                         use_container_width=True, hide_index=True)
            cb.caption("Локация 2")
            cb.dataframe(r2['details'][['Категория', 'Название', 'Тип']],
                         use_container_width=True, hide_index=True)

        m = folium.Map(location=[(r1['lat'] + r2['lat']) / 2, (r1['lon'] + r2['lon']) / 2],
                       zoom_start=13, tiles="OpenStreetMap")
        add_isochrones_to_map(m, r1['poly_dict'], [r1['lat'], r1['lon']], r1['node_coords'],
                              label="Локация 1: ")
        add_isochrones_to_map(m, r2['poly_dict'], [r2['lat'], r2['lon']], r2['node_coords'],
                              label="Локация 2: ",
                              colors={5: '#3498db', 10: '#2980b9', 15: '#5d6d7e'})
        fit_map(m, [r1['poly_dict'], r2['poly_dict']])

        if show_pois:
            for r_data, prefix in [(r1, "Локация 1: "), (r2, "Локация 2: ")]:
                for _, row in r_data['details'].iterrows():
                    folium.CircleMarker(
                        [row['lat'], row['lon']],
                        radius=5,
                        color=CAT_COLORS[row['Категория']],
                        fill=True,
                        fill_opacity=0.9,
                        tooltip=f"{prefix}{row['Категория']}: {row['Название']}"
                    ).add_to(m)

        if show_net:
            add_network_layer(m, bundle, r1['lat'], r1['lon'])
            add_network_layer(m, bundle, r2['lat'], r2['lon'])

        st_folium(m, width=1200, height=550, key="compare_map", returned_objects=[])
    else:
        st.info("👈 Введите два адреса и нажмите 'Сравнить'")
