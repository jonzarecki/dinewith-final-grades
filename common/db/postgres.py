import logging
from typing import List

import geopandas as gpd
import pandas as pd
import sqlalchemy as sa
from geoalchemy2 import WKTElement, Geography, Geometry
from geoalchemy2.types import _GISType
from tqdm import tqdm

from common.db.sqlalchemy_utils import drop_table, insert_into_table, get_temp_table_name


def save_geo_series_to_tmp_table(geo_series: gpd.GeoSeries, eng: sa.engine.Engine) -> str:
    """
    Save a geo series as a table in the db, for better performance
    Args:
        geo_series: The GeoSeries to be inserted into a db table
        eng: SQL Alchemy engine
    Returns:
        The name of the new table
    """
    geo_series = geo_series.rename('geom')
    gdf = gpd.GeoDataFrame(geo_series, columns=['geom'], geometry='geom')
    gdf['geom'] = gdf.geometry.apply(lambda x: WKTElement(x.wkt, srid=4326))
    gdf['geom_id'] = range(len(gdf))
    tbl_name = get_temp_table_name()
    insert_into_table(eng, gdf, tbl_name, dtypes={'geom': Geography(srid=4326), 'geom_id': sa.INT})
    add_postgis_index(eng, tbl_name, 'geom')
    return tbl_name


def get_index_str_for_unique(index_columns: List[str], dtypes: dict):
    return ",".join([f"ST_GeoHash({col})" if isinstance(dtypes[col], _GISType) else col
                     for col in index_columns])


def add_postgis_index(eng: sa.engine.Engine, table_name: str, geom_col: str):
    with eng.begin() as con:
        con.execute(f"create index {table_name}_{geom_col}_idx on {table_name} using gist ({geom_col});")


def oracle_to_postgres(oracle_con: sa.engine.Engine, postgres_con: sa.engine.Engine,
                       source_table_name: str, destination_table_name: str = None,
                       geom_col: str = None, drop_columns: List = None, limit: int = None):
    """
    Replicate table from oracle DB to postgres DB.
    Args:
        oracle_con: connector to the oracle DB
        postgres_con: connector to the postgres DB
        source_table_name: table name in the oracle DB
        destination_table_name: desired table name in the postgres DB. If None, then gives 'source_table_name'.
        geom_col: name of the geo column. if None then no column will be treated as a geo column
        drop_columns: list of all columns in the source table that you want to drop
        limit: number of rows to copy. If None then copies all rows.
    Returns:
        pushed table to the postgres table
    """
    limit_condition = f"where rownum <= {limit}" if limit is not None else ""
    destination_table_name = destination_table_name if destination_table_name is not None else source_table_name

    df = pd.read_sql(f"""select * from {source_table_name} {limit_condition}""", oracle_con)
    df = df.drop(drop_columns, axis=1)

    dtype = None
    if geom_col is not None:
        df = df.rename({geom_col: 'way'}, axis=1)
        dtype = {'way': Geometry('GEOMETRY', srid=4326)}
        df['way'] = df['way'].map(lambda geo: WKTElement(geo, srid=4326))

    drop_table(destination_table_name, postgres_con)

    for i in tqdm(range(len(df)), desc="Pushing rows to DB", unit="row"):
        try:
            df.iloc[[i]].to_sql(destination_table_name, postgres_con, if_exists='append', index=False,
                                dtype=dtype)
        except:  # noqa
            logging.info(f'error in sample {i}')

    if geom_col is not None:
        add_postgis_index(postgres_con, destination_table_name, 'way')
