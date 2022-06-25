import pandas as pd
import sqlalchemy as sa
from tqdm import tqdm

from common.db.sqlalchemy_utils import get_df


def add_sdo_geo_to_table(table_name: str, wkt_geo_column: str, geo_sdo_column: str, eng: sa.engine.Engine,
                         no_asserts=False, dispose_eng=False):
    """
    Adds a separate SDO_GEOMETRY column from an existing wkt/wkb column
    Args:
        table_name: The table we're working on
        wkt_geo_column: The name of the column containing the wkt/wkb
        geo_sdo_column: The name of the column we want to store the sdo_geometry object
        eng: An engine object connecting the db
        no_asserts: True if no asserts on columns are made (will override existing data)
        dispose_eng: Whether to dispose of the engine after the function
    Returns:
         None
    """
    df = get_df(f"SELECT * FROM {table_name} WHERE ROWNUM < 1", eng)  # fetch only one row
    if not no_asserts:
        assert wkt_geo_column in df.columns, f"{wkt_geo_column} not in table {table_name}"
        assert geo_sdo_column not in df.columns, f"{geo_sdo_column} already in table {table_name}"

    if geo_sdo_column not in df.columns and wkt_geo_column in df.columns:
        eng.execute(f"""
        ALTER TABLE {table_name}
        ADD {geo_sdo_column} SDO_GEOMETRY
                     """)
        eng.execute("COMMIT")

    # run for each feature seperetly
    feature_names = pd.read_sql(f"""select distinct {FEATURE_NAME} from {table_name}""", eng).iloc[:, 0]
    conn = eng.raw_connection()
    cur = conn.cursor()

    def add_sdo(feature_name):
        SELECT_SDO_GEO = f"""select SDO_GEOMETRY({wkt_geo_column}, 4326) as {geo_sdo_column}, ROWID as rid
                            from {table_name}
                            where {geo_sdo_column} IS NULL
                            and {FEATURE_NAME} = '{feature_name}'
                            """

        # TIP: when using weird SDO_UTIL functions its better to use the raw connection.
        # In this case no values were returned by the merge into. only with the
        cur.execute(f"""
        merge into {table_name} curr
                    using ({SELECT_SDO_GEO}) tmp
                    on (curr.ROWID = tmp.rid)
                    when matched then
                    update set curr.{geo_sdo_column} = tmp.{geo_sdo_column}
        """)
        conn.commit()

    [add_sdo(feature_name) for feature_name in tqdm(feature_names, desc='adding SDO to features', unit='feature')]
    cur.close()

    # fix coordinate system
    eng.execute(f"update {table_name} T set T.{geo_sdo_column}.SDO_SRID = 4326 WHERE T.{geo_sdo_column} is not null")

    # add spatial index and add to user_sdo_geom_metadata table
    usersdo_df = get_df("SELECT * FROM user_sdo_geom_metadata", eng)
    if (table_name, geo_sdo_column) not in [tuple(row) for row in usersdo_df[['TABLE_NAME', 'COLUMN_NAME']].values]:
        eng.execute(f"""
        INSERT INTO user_sdo_geom_metadata
        VALUES ('{table_name}', '{geo_sdo_column}', sdo_dim_array(sdo_dim_element('X', -100, 100, 0.000005),
                                                       sdo_dim_element('Y', -100, 100, 0.000005)), 4326)
        """)

    is_there_index = len(eng.execute(f"""
                    select index_name
                    from SYS.ALL_INDEXES
                    where table_name = '{table_name}'
                    """).fetchall()) > 0

    if not is_there_index:
        acronym_short_geo_sdo = ''.join([s[0] for s in geo_sdo_column.split('_')])  # first letter of each word
        eng.execute(f"""
                    CREATE INDEX {table_name}_{acronym_short_geo_sdo}_idx
                    ON {table_name} ({geo_sdo_column}) INDEXTYPE IS MDSYS.SPATIAL_INDEX
                    """)

    if dispose_eng:
        eng.dispose()
