import geopy.distance
from shapely.geometry import Point


def distance_in_meters(p1: Point, p2: Point) -> float:
    dist = geopy.distance.geodesic((p1.x, p1.y), (p2.x, p2.y))
    return float(dist.meters)
