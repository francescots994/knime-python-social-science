import logging
import knime.extension as knext
from util import utils as kutil
from ..configs.models.arima_apply import SPredictorApplyParams
import numpy as np
import pickle

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="ARIMA Predictor (Labs)",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/models/SARIMA_Forecaster-Apply.png",
    category=kutil.category_timeseries,
    id="arima_apply",
)
@knext.input_binary(
    name="Model Input",
    description="Trained SARIMA model",
    id="sarima.model",
)
@knext.output_table(
    name="Forecast",
    description="Table containing forecasts for the configured column, the first value will be one timestamp ahead of the final training value used.",
)
class SarimaForcasterApply:
    """
    This node generates forecasts with a (S)ARIMA Model.

    Based on a trained SARIMA model given at the model input port of this node, the forecasts values are computed.
    """

    sarima_params = SPredictorApplyParams
    natural_log = sarima_params.natural_log
    dynamic_check = sarima_params.dynamic_check
    number_of_forecasts = sarima_params.number_of_forecasts

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema: knext.Schema,  # NOSONAR input_schema is necessary
    ):
        if self.natural_log and self.dynamic_check:
            configure_context.set_warning(
                "Enabling dynamic predictions with log transformation can create invalid predictions."
            )
        return knext.Column(knext.double(), "Forecasts")

    def execute(self, exec_context: knext.ExecutionContext, model_input):
        trained_model = pickle.loads(model_input)
        exec_context.set_progress(0.4)

        # make out-of-sample forecasts
        forecasts = trained_model.forecast(steps=self.number_of_forecasts).to_frame(
            name="Forecasts"
        )

        exec_context.set_progress(0.8)

        # reverse log transformation for forecasts
        if self.natural_log:
            forecasts = np.exp(forecasts)

        return knext.Table.from_pandas(forecasts)
