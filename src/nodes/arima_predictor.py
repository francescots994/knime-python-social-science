import logging
import knime.extension as knext
from util import utils as kutil
import numpy as np
import pickle


LOGGER = logging.getLogger(__name__)


@knext.node(
    name="Auto SARIMA Predictor",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/models/SARIMA_Forecaster-Apply.png",
    category=kutil.category_timeseries,
    id="auto_sarima_predictor",
)
@knext.input_binary(
    name="Model Input",
    description="A pickled model, output from the Auto-SARIMA Learner node. This model contains all the information necessary to generate forecasts.",
    id="auto_sarima.model",
)
@knext.output_table(
    name="Forecasts",
    description="Table containing the generated forecast values. The forecast starts one period after the end of the data used to train the input model.",
)
class SarimaForcaster:
    """
    Generates future forecasts using a pre-trained SARIMA model, output of the node Auto-SARIMA Learner.

    This node takes the pre-trained SARIMA model and produces out-of-sample forecasts for a specified number of future periods.
    The forecasts are generated directly from the end of the training data used to fit the input model.

    **Key Parameters & Behavior:**

    -   `Model Input` is a pickled `SARIMAXResults` object from `statsmodels`, output of node Auto-SARIMA Learner. It is the model utilized for the forecasting.

    -   `Forecast periods` allows to chose the number of forecast periods, minimum is 1.

    -   The quality of the forecasts heavily depends on the quality and representativeness of the input model.

    -   If a log transformation was applied during training (in the Learner node), the "Reverse Log" option must be checked here to ensure forecasts are on the original scale.
    
    **Outputs:**

    1.  Table with the forecasted values.

    """

    number_of_forecasts = knext.IntParameter(
        label="Forecasts",
        description="Specifies the number of future time periods for which to generate forecasts. For example, if the training data ended at time T, a value of 12 here would forecast for T+1, T+2, ..., T+12.",
        default_value=1,
        min_value=1,
    )
    natural_log = knext.BoolParameter(
        label="Reverse Log",
        description="Select this option if the original data was log-transformed within the Auto-SARIMA Learner node during model training. This will apply an exponential function (np.exp) to the forecasted values to revert them to their original scale. Ensure this matches the transformation setting used in the Learner node.",
        default_value=False,
    )


    def configure(
            self,
            configure_context: knext.ConfigurationContext,
            input_model
        ):

        forecast_schema = knext.Column(knext.double(), "Forecasts")

        return (
            forecast_schema
        )


    def execute(
            self, 
            exec_context: knext.ExecutionContext,
            input_model):

        exec_context.set_progress(0.1)

        trained_model = pickle.loads(input_model)

        exec_context.set_progress(0.5)

        # make out-of-sample forecasts
        forecasts = trained_model.forecast(
            steps=self.number_of_forecasts
            ).to_frame(name="Forecasts")

        exec_context.set_progress(0.8)
        
        # reverse log transformation for forecasts
        if self.natural_log:
            forecasts = np.exp(forecasts)

        return knext.Table.from_pandas(forecasts)