from functools import partial
from typing import List, Union

import folium
import geopandas as gpd
import numpy as np
from cachetools import cached, LRUCache
from folium.features import FeatureGroup
from folium.plugins import MeasureControl
from matplotlib import colors
from pandas import DataFrame
from shapely import wkt
from shapely.geometry import Point
from shapely.wkt import loads

from common.geographic.folium_extensions import NoClickGeoJson
from common.geographic.geo_utils import geoms2bbox, geom2image_projection, meters2degrees


@cached(cache=LRUCache(maxsize=256), key=lambda *a: hash(hash(tuple(p)) for p in a))
def build_image_overlay(wkt_array, color, fill_color, fill_alpha, line_alpha, step):
    gds = gpd.GeoSeries([wkt.loads(s) for s in wkt_array])

    bbox = geoms2bbox(gds)
    min_lon, min_lat, max_lon, max_lat = bbox

    image = np.zeros(shape=(int((max_lon - min_lon) / step) + 1, int((max_lat - min_lat) / step) + 1, 4))
    for index, geom in enumerate(gds):
        # fill color
        cur_color = fill_color[index] if type(fill_color) is not str else fill_color
        cur_color = list(colors.to_rgb(cur_color)) + [fill_alpha]
        image = geom2image_projection(image, bbox, geom, color=cur_color, fill=True)

        # edge color
        cur_color = color[index] if type(color) is not str else color
        cur_color = list(colors.to_rgb(cur_color)) + [line_alpha]
        image = geom2image_projection(image, bbox, geom, color=cur_color, fill=False)

    return image


class GeoMap:
    """
    This class will handle all the function  to explore data related to map,
    functions like show on the map and load geometries on the map
    this class is depends on the modules: folium for the map and geomet to parse wkt
    """

    def __init__(self, start_location: Point, start_zoom: int = 11, layer_control=True, measure_control=False):
        """Build GeoMap

        Args:
            start_location
                the map will open with this geographic position as the center

            start_zoom (int, default=True)
                the map will be initialized with this zoom (range between 0-15)

            layer_control: weather to see a layer choice or not

            measure_control: weather to see a distance measure control or not
        """
        self.start_location = start_location
        self.start_zoom = start_zoom
        self.layer_control = layer_control
        self.measure_control = measure_control
        self._get_folium_map()

    def _get_folium_map(self):

        self.map = folium.Map(location=(self.start_location.y, self.start_location.x), zoom_start=self.start_zoom)

    def clear_map(self):
        """
        init new folium map
        """
        self._get_folium_map()

    def show(self):
        """Returns the map so jupyter notebook can do eval on this map and show it as a result.

        Beware that if you use this function from another function you should return the result of this function
        to the notebook or call the show function outside.

        Returns:
            self
        """
        if self.layer_control:
            folium.LayerControl().add_to(self.map)
        if self.measure_control:
            MeasureControl().add_to(self.map)

        return self.map

    def save(self, path: str):
        """
        Args:
            path (str): relative path or direct path to save the map as html (when you save the map save it as *.html)
        """
        self.map.save(path)

    def load_wkt_layer_from_dataframe(self, df: DataFrame, wkt_column_name: str,
                                      color: Union[str, List[str]] = '#0078d7',
                                      fill_color: Union[str, List[str]] = '#0048a7',
                                      fill_alpha: float = 0.2,
                                      group_name=None,
                                      change_bounds_on_click=False,
                                      pop_up: bool = True):
        """Loads additional layer to the map.

        Args:
            df (Pandas dataframe,default=False)
                dataframe with at least one geographic column

            wkt_column_name (str,default=False)
                the name of the geometry column to show
                the geometries should be in the format of wkt string
                if you are using oracle, select the sdo_geometry with the oracle function sdo_util.to_wktgeometry

            color (str,default=True)
                the color to use when drawing the geoms on the map
                examples - blue,white,#0078d7, #9999d9

            fill_color(str,default=True)
                the color to fill when its complex geometry as polygon

            fill_alpha(float)
                the opacity of the fill color, between 0 and 1

            group_name: (str,default=None)
                will take geometries and create a group in the LayerControl, gives a specific name for the group.
                If None then does'nt group the geometries.

            change_bounds_on_click: whether a mouse-click on change the map bounds to the object

            pop_up: whether a mouse-click on the object will pop-up
        """
        part_func = lambda x, color_index: {'color': color[color_index] if type(color) is not str else color,  # noqa
                                            'fillColor': fill_color[color_index] if type(
                                                fill_color) is not str else fill_color,
                                            'fillOpacity': fill_alpha,
                                            'markerColor': color[color_index] if type(color) is not str else color,
                                            'icon_color': color[color_index] if type(color) is not str else color}

        object_to_add = self.map
        if group_name is not None:
            object_to_add = FeatureGroup(name=group_name)

        for index, row in df.iterrows():
            geom_dict = wkt.loads(row[wkt_column_name])
            shp_geom = loads(row[wkt_column_name])
            row_formatted = ""
            for index_c, column in enumerate(row.index.values):
                if column == wkt_column_name:
                    pass
                else:
                    row_formatted += "<b>{}</b>: {} <br/>".format(column, row[column])
            row_formatted += "<b>{}</b>: {} <br/>".format("GEOM CENTROID", shp_geom.centroid)
            feature = (folium.GeoJson if change_bounds_on_click else NoClickGeoJson)(
                geom_dict, style_function=partial(part_func, color_index=index))
            if pop_up:
                popup = folium.Popup(row_formatted.replace("'", "\""))
                popup.add_to(feature)
            feature.add_to(object_to_add)

        if group_name is not None:
            object_to_add.add_to(self.map)

    def load_image_overlay_from_dataframe(self, df: DataFrame, wkt_column_name: str, step: float = 1,
                                          color: Union[str, List[str]] = '#0078d7', line_alpha: float = 1,
                                          fill_color: Union[str, List[str]] = '#0048a7',
                                          fill_alpha: float = 0.2, name=None):
        """Loads additional layer to the map.

        Args:
            df (Pandas dataframe,default=False)
                dataframe with at least one geographic column

            wkt_column_name (str,default=False)
                the name of the geometry column to show
                the geometries should be in the format of wkt string
                if you are using oracle, select the sdo_geometry with the oracle function sdo_util.to_wktgeometry

            step (float,default=1)
                resolution of sample in meters

            color (str,default=True)
                the color to use when drawing the geoms on the map
                examples - blue,white,#0078d7, #9999d9

            fill_color(str,default=True)
                the color to fill when its complex geometry as polygon

            fill_alpha(float)
                the opacity of the fill color, between 0 and 1

            name: (str,default=None)
                gives a specific name for the image_overlay

            line_alpha:
                the opacity of the line color, between 0 and 1
        """
        gds = gpd.GeoSeries(df[wkt_column_name].map(wkt.loads))

        step = meters2degrees(step)
        # change all to tuples for LRU_CACHE to work.
        image = build_image_overlay(df[wkt_column_name], color, fill_color, fill_alpha, line_alpha, step)

        min_lon, min_lat, max_lon, max_lat = geoms2bbox(gds)
        image_overlay = folium.raster_layers.ImageOverlay(image,
                                                          bounds=[[min_lat, min_lon], [max_lat, max_lon]],
                                                          origin='lower', name=name)

        image_overlay.add_to(self.map)

    def __call__(self):
        """To easily show the map just apply the object."""
        return self.show()
