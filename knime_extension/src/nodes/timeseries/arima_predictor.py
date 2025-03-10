import logging
import knime.extension as knext
from util import utils as kutil
from ..configs.models.arima import SarimaPredictorParams
import numpy as np
import pandas as pd
import pickle

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="ARIMA Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/models/SARIMA_Forecaster-Apply.png",
    category=kutil.category_timeseries,
    id="arima_predictor",
)
# Added the input port for the time series to use to generate forecasts
# TODO everything else
@knext.input_table(
    name="Input Data",
    description="Table containing data to generate forecast with the trained SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
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
class SarimaForcaster:
    """
    This node generates forecasts with a (S)ARIMA Model.

    Based on a trained SARIMA model given at the model input port of this node, the forecasts values are computed.
    """

    sarima_params = SarimaPredictorParams()

    def configure(
            self,
            configure_context: knext.ConfigurationContext,
            input_schema: knext.Schema,  # NOSONAR input_schema is necessary
        ):

        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the
        # first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.sarima_params.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.sarima_params.input_column,
            input_schema,
            kutil.is_numeric,
        )

        # dynamic predictions on log transformed column can generate invalid output
        if (
            self.sarima_params.predictor_params.natural_log_predictor
            and self.sarima_params.predictor_params.dynamic_check
        ):
            configure_context.set_warning(
                "Enabling dynamic predictions on log transformed target column can generate invalid output."
            )

    # def configure(self, configure_context, input_schema):

    #     if self.natural_log and self.dynamic_check:
    #         configure_context.set_warning(
    #             "Enabling dynamic predictions with log transformation can create invalid predictions."
    #         )

        forecast_schema = knext.Column(knext.double(), "Forecasts")
        return forecast_schema




    def execute(self, exec_context: knext.ExecutionContext, model_input, data_input: knext.Table):
    
        df: pd.DataFrame
        df = data_input.to_pandas()
        target_col: pd.Series
        target_col = df[self.sarima_params.input_column]

        # check if log transformation is enabled
        if self.sarima_params.predictor_params.natural_log_predictor:
            num_negative_vals = kutil.count_negative_values(target_col)

            if num_negative_vals > 0:
                raise knext.InvalidParametersError(
                    f" There are '{num_negative_vals}' non-positive values in the target column."
                )
            target_col = np.log(target_col)
        exec_context.set_progress(0.1)

        # TODO understand how to call the two inputs

        trained_model = pickle.loads(model_input)
        exec_context.set_progress(0.4)

        # make out-of-sample forecasts
        # forecasts = trained_model.forecast(steps=self.sarima_params.predictor_params.number_of_forecasts).to_frame(
        #     name="Forecasts"
        # )
        
        # instead, use the method .apply to generate forecasts using the same model but with new data



        exec_context.set_progress(0.8)

        # reverse log transformation for forecasts
        if self.sarima_params.predictor_params.natural_log_predictor:
            forecasts = np.exp(forecasts)

        return knext.Table.from_pandas(forecasts)
