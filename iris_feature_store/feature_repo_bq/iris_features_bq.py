from datetime import timedelta

from feast import BigQuerySource, Entity, FeatureView, Field
from feast.types import Float64
from feast.value_type import ValueType

iris = Entity(name="iris", join_keys=["iris_id"], value_type=ValueType.INT64,
              description="One iris plant, tracked over time")

# The ONLY real change vs feature_repo/iris_features.py: FileSource -> BigQuerySource
iris_source = BigQuerySource(
    name="iris_source_bq",
    table="arcane-sentinel-252018.feast_iris.iris_source",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365 * 5),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width",  dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width",  dtype=Float64),
    ],
    online=True,
    source=iris_source,
)
