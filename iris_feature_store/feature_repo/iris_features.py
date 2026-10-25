from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64
from feast.value_type import ValueType

# ENTITY: the "thing" features describe. join_keys = the column that identifies it.
iris = Entity(
    name="iris",
    join_keys=["iris_id"],
    value_type=ValueType.INT64,
    description="One iris plant, tracked over time",
)

# DATA SOURCE: where raw feature values live + which column says WHEN each was measured.
iris_source = FileSource(
    name="iris_source",
    path="data/iris_data.parquet",                 # relative to this feature repo
    timestamp_field="event_timestamp",             # when the measurement happened
    created_timestamp_column="created_timestamp",  # when the row was written (tie-breaker)
)

# FEATURE VIEW: which columns are features, for which entity, from which source.
iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365 * 5),                   # how old a value may be and still count
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width",  dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width",  dtype=Float64),
    ],
    online=True,                                   # allow materializing into the online store
    source=iris_source,
)
