import knime.extension as knext

category = knext.category(
    path="/community/",
    level_id="socialscience",
    name="KNIME Social Science Extension",
    description="Python Nodes for Statistical Analysis",
    icon="icons/Time_Series_Analysis.png",
)

from nodes import arima_learner, arima_predictor  # noqa: F401