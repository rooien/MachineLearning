import json
import math
import time
import urllib.error
import urllib.request

import pandas as pd

MELBOURNE_CBD = (-37.8136, 144.9631)
ROUTES_API_URL = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
PLACES_API_URL = "https://places.googleapis.com/v1/places:searchNearby"
SCHOOL_TYPES = {
    "preschool": "preschool",
    "primary_school": "primary_school",
    "secondary_school": "secondary_school",
}


def load_housing_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [column.strip() for column in df.columns]

    # The exported CSV includes a duplicated header row as data in row 0.
    df = df[pd.to_numeric(df["Latitude"], errors="coerce").notna()].copy()

    numeric_columns = [
        "Postcode",
        "Latitude",
        "Longitude",
        "Bedroom",
        "Bathroom",
        "Car Park space",
        "Land size",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_m = 6371000
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return radius_m * c


def add_straight_line_cbd_distance(
    df: pd.DataFrame,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    cbd_lat: float = MELBOURNE_CBD[0],
    cbd_lon: float = MELBOURNE_CBD[1],
    output_col: str = "cbd_straight_line_km",
) -> pd.DataFrame:
    enriched = df.copy()
    enriched[output_col] = enriched.apply(
        lambda row: haversine_distance_m(
            float(row[lat_col]),
            float(row[lon_col]),
            cbd_lat,
            cbd_lon,
        )
        / 1000,
        axis=1,
    )
    return enriched


def _post_json(url: str, body: dict, headers: dict) -> dict | list:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Google API request failed: {exc.code} {details}") from exc


def add_google_cbd_distance(
    df: pd.DataFrame,
    api_key: str,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    cbd_lat: float = MELBOURNE_CBD[0],
    cbd_lon: float = MELBOURNE_CBD[1],
    batch_size: int = 200,
    output_col: str = "cbd_drive_distance_km",
) -> pd.DataFrame:
    if batch_size < 1 or batch_size > 625:
        raise ValueError("batch_size must be between 1 and 625")

    enriched = df.copy()
    enriched[output_col] = pd.NA

    valid_rows = enriched[enriched[lat_col].notna() & enriched[lon_col].notna()]

    for start in range(0, len(valid_rows), batch_size):
        batch = valid_rows.iloc[start : start + batch_size]
        body = {
            "origins": [
                {
                    "waypoint": {
                        "location": {
                            "latLng": {
                                "latitude": float(row[lat_col]),
                                "longitude": float(row[lon_col]),
                            }
                        }
                    }
                }
                for _, row in batch.iterrows()
            ],
            "destinations": [
                {
                    "waypoint": {
                        "location": {
                            "latLng": {
                                "latitude": cbd_lat,
                                "longitude": cbd_lon,
                            }
                        }
                    }
                }
            ],
            "travelMode": "DRIVE",
            "routingPreference": "TRAFFIC_UNAWARE",
        }
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "originIndex,destinationIndex,distanceMeters,condition",
        }

        response = _post_json(ROUTES_API_URL, body, headers)
        for element in response:
            if element.get("condition") != "ROUTE_EXISTS":
                continue
            row_index = batch.index[element["originIndex"]]
            enriched.at[row_index, output_col] = element["distanceMeters"] / 1000

    return enriched


def find_nearest_schools(
    lat: float,
    lon: float,
    api_key: str,
    radius_m: int = 5000,
    max_result_count: int = 20,
) -> dict:
    body = {
        "includedTypes": list(SCHOOL_TYPES.values()),
        "maxResultCount": max_result_count,
        "rankPreference": "DISTANCE",
        "locationRestriction": {
            "circle": {
                "center": {"latitude": float(lat), "longitude": float(lon)},
                "radius": float(radius_m),
            }
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": (
            "places.displayName,places.formattedAddress,"
            "places.location,places.primaryType,places.types"
        ),
    }

    response = _post_json(PLACES_API_URL, body, headers)
    places = response.get("places", [])

    nearest_by_type = {
        school_type: {"name": pd.NA, "address": pd.NA, "distance_m": pd.NA}
        for school_type in SCHOOL_TYPES
    }

    for place in places:
        place_location = place.get("location", {})
        place_lat = place_location.get("latitude")
        place_lon = place_location.get("longitude")
        if place_lat is None or place_lon is None:
            continue

        distance_m = haversine_distance_m(lat, lon, place_lat, place_lon)
        place_types = set(place.get("types", []))
        display_name = place.get("displayName", {}).get("text") or pd.NA
        formatted_address = place.get("formattedAddress", pd.NA)

        for school_type in SCHOOL_TYPES:
            if school_type not in place_types:
                continue

            current_distance = nearest_by_type[school_type]["distance_m"]
            if pd.isna(current_distance) or distance_m < float(current_distance):
                nearest_by_type[school_type] = {
                    "name": display_name,
                    "address": formatted_address,
                    "distance_m": round(distance_m, 1),
                }

    return nearest_by_type


def add_nearest_school_columns(
    df: pd.DataFrame,
    api_key: str,
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    radius_m: int = 5000,
    max_rows: int | None = None,
    pause_seconds: float = 0.0,
) -> pd.DataFrame:
    enriched = df.copy()

    for school_type in SCHOOL_TYPES:
        enriched[f"nearest_{school_type}_name"] = pd.NA
        enriched[f"nearest_{school_type}_address"] = pd.NA
        enriched[f"nearest_{school_type}_distance_m"] = pd.NA

    target_index = enriched.index[:max_rows] if max_rows is not None else enriched.index

    for row_number, index in enumerate(target_index, start=1):
        lat = enriched.at[index, lat_col]
        lon = enriched.at[index, lon_col]
        if pd.isna(lat) or pd.isna(lon):
            continue

        nearby_schools = find_nearest_schools(
            lat=float(lat),
            lon=float(lon),
            api_key=api_key,
            radius_m=radius_m,
        )

        for school_type, place in nearby_schools.items():
            enriched.at[index, f"nearest_{school_type}_name"] = place["name"]
            enriched.at[index, f"nearest_{school_type}_address"] = place["address"]
            enriched.at[index, f"nearest_{school_type}_distance_m"] = place["distance_m"]

        if pause_seconds > 0:
            time.sleep(pause_seconds)

        if row_number % 25 == 0:
            print(f"Processed {row_number} properties")

    return enriched


def add_google_features(
    df: pd.DataFrame,
    api_key: str,
    max_rows: int | None = None,
    school_radius_m: int = 5000,
    pause_seconds: float = 0.0,
) -> pd.DataFrame:
    if max_rows is not None:
        working_df = df.head(max_rows).copy()
    else:
        working_df = df.copy()

    working_df = add_straight_line_cbd_distance(working_df)
    working_df = add_google_cbd_distance(working_df, api_key=api_key)
    working_df = add_nearest_school_columns(
        working_df,
        api_key=api_key,
        radius_m=school_radius_m,
        pause_seconds=pause_seconds,
    )
    return working_df
