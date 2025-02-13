import knime.extension as knext

category = knext.category(
    path="/community/",
    level_id="socialscience",
    name="KNIME Social Science Extension",
    description="Python Nodes for Statistical Analysis",
    icon="icons/Time_Series_Analysis.png",
)

from nodes.timeseries import (
    arima_node,  # noqa: F401
    arima_node_apply
    )
