import logging
import knime.extension as knext
from util import utils as kutil
import numpy as np
import pandas as pd
import pickle

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Auto SARIMA Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/models/SARIMA_Forecaster-Apply.png",
    category=kutil.category_timeseries,
    id="auto_sarima_predictor",
)
# Added the input port for the time series to use to generate forecasts
@knext.input_table(
    name="Input Data",
    description="Table containing data to generate forecast with the trained SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
)
@knext.input_binary(
    name="Model Input",
    description="Trained SARIMA model",
    id="auto_sarima.model",
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


    number_of_forecasts = knext.IntParameter(
        label="Forecast",
        description="Forecasts of the given time series *h* period ahead of the training data.",
        default_value=1,
        min_value=1,
    )
    dynamic_check = knext.BoolParameter(
        label="Generate out-of-sample forecasts dynamically",
        description="Check this box to use in-sample prediction as lagged values. Otherwise use true values.",
        default_value=False,
    )
    natural_log = knext.BoolParameter(
        label="Reverse Log",
        description="Check this box if you applied the log transform inside the SARIMA Forecaster node while training your model. It will reverse the transform before generating forecasts.",
        default_value=False,
    )
    # target column for modelling
    input_column = knext.ColumnParameter(
        label="Target Column",
        description="Table containing training data for fitting the SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
        port_index=0,
        column_filter=kutil.is_numeric,
    )


    def configure(
            self,
            configure_context: knext.ConfigurationContext,
            input_schema: knext.Schema,
            input_model
        ):
        
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the
        # first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.input_column,
            input_schema,
            kutil.is_numeric,
        )

        # dynamic predictions on log transformed column can generate invalid output
        if (
            self.natural_log
            and self.dynamic_check
        ):
            configure_context.set_warning(
                "Enabling dynamic predictions on log transformed target column can generate invalid output."
            )

        if self.number_of_forecasts < 1:
            configure_context.set_warning(
                "At least one forecast should be made by the model."
            )

        forecast_schema = knext.Column(knext.double(), "Forecasts")

        return (
            forecast_schema
        )




    def execute(
            self, 
            exec_context: knext.ExecutionContext,
            data_input: knext.Table,
            input_model):

        df: pd.DataFrame
        df = data_input.to_pandas()
        target_col: pd.Series
        target_col = df[self.input_column]

        # check if log transformation is enabled
        if self.natural_log:
            num_negative_vals = kutil.count_negative_values(target_col)

            if num_negative_vals > 0:
                raise knext.InvalidParametersError(
                    f" There are '{num_negative_vals}' non-positive values in the target column."
                )
            target_col = np.log(target_col)
        exec_context.set_progress(0.1)

        trained_model = pickle.loads(input_model)
        exec_context.set_progress(0.4)

        # use the method .apply to generate forecasts using the same model but with new data
        new_trained_model = trained_model.apply(target_col)

        exec_context.set_progress(0.8)
        
        # make out-of-sample forecasts
        forecasts = new_trained_model.forecast(
            steps=self.number_of_forecasts
            ).to_frame(name="Forecasts")

        # reverse log transformation for forecasts
        if self.natural_log:
            forecasts = np.exp(forecasts)

        return knext.Table.from_pandas(forecasts)