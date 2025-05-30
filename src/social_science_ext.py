import knime.extension as knext

category = knext.category(
    path="/community/",
    level_id="socialscience",
    name="Social Science Extension",
    description="Nodes for Statistical Analysis",
    icon="icons/Time_Series_Analysis.png",
)

from nodes import arima_learner, arima_predictor  # noqa: F401