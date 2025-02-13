import logging
import knime.extension as knext
from util import utils as kutil
from ..configs.models.arima import SarimaForecasterParams
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

LOGGER = logging.getLogger(__name__)


@knext.node(
    name="ARIMA Learner (Labs)",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/SARIMA_Forecaster.png",
    category=kutil.category_timeseries,
    id="arima",
)
@knext.input_table(
    name="Input Data",
    description="Table containing training data for fitting the SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
)
@knext.output_table(
    name="Forecast",
    description="Table containing forecasts for the configured column, the first value will be 1 timestamp ahead of the final training value used.",
)
@knext.output_table(
    name="Residuals",
    description="In sample model prediction values and residuals i.e. difference between observed value and the predicted output.",
)
@knext.output_table(
    name="Coefficients and Statistics",
    description="Table containing fitted model coefficients, variance of residuals (sigma2), and several model metrics along with their standard errors.",
)
@knext.output_binary(
    name="Model",
    description="Pickled model object that can be used by the SARIMA (Apply) node to generate different forecast lengths without refitting the model",
    id="sarima.model",
)
class SarimaForcaster:
    """
    Trains and generates a forecast with a (S)ARIMA Model

    Trains and generates a forecast using a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model. The SARIMA models captures temporal structures in time series data in the following components:

    - **AR (AutoRegressive):** Relationship between the current observation and a number (p) of lagged observations.
    - **I (Integrated):** Degree (d) of differencing required to make the time series stationary.
    - **MA (Moving Average):** Time series mean and the relationship between the current forecast error and a number (q) of lagged forecast errors.

    *Seasonal versions of these components operate similarly, with lag intervals equal to the seasonal period (S).*
    """

    sarima_params = SarimaForecasterParams()

    # merge in-samples and residuals (In-Samples & Residuals)
    def configure(self, configure_context, input_schema):
        self.sarima_params.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.sarima_params.input_column,
            input_schema,
            kutil.is_numeric,
        )

        if (
            self.sarima_params.learner_params.natural_log
            and self.sarima_params.predictor_params.dynamic_check
        ):
            configure_context.set_warning(
                "Enabling dynamic predictions on log transformed target column can generate invalid output."
            )

        forecast_schema = knext.Column(knext.double(), "Forecasts")
        insamp_res_schema = knext.Schema(
            [knext.double(), knext.double()], ["Residuals", "In-Samples"]
        )
        model_summary_schema = knext.Column(knext.double(), "Value")
        binary_model_schema = knext.BinaryPortObjectSpec("sarima.model")

        return (
            forecast_schema,
            insamp_res_schema,
            model_summary_schema,
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input: knext.Table):
        df: pd.DataFrame
        df = input.to_pandas()
        target_col: pd.Series
        target_col = df[self.sarima_params.input_column]

        # check if log transformation is enabled
        if self.sarima_params.learner_params.natural_log:
            num_negative_vals = kutil.count_negative_values(target_col)

            if num_negative_vals > 0:
                raise knext.InvalidParametersError(
                    f" There are '{num_negative_vals}' non-positive values in the target column."
                )
            target_col = np.log(target_col)
        exec_context.set_progress(0.1)

        self.__validate(target_col)
        exec_context.set_progress(0.3)

        # model initialization and training
        model = SARIMAX(
            target_col,
            order=(
                self.sarima_params.learner_params.ar_order_param,
                self.sarima_params.learner_params.i_order_param,
                self.sarima_params.learner_params.ma_order_param,
            ),
            seasonal_order=(
                self.sarima_params.learner_params.seasoanal_ar_order_param,
                self.sarima_params.learner_params.seasoanal_i_order_param,
                self.sarima_params.learner_params.seasoanal_ma_order_param,
                self.sarima_params.learner_params.seasonal_period_param,
            ),
        )
        exec_context.set_progress(0.5)
        trained_model = model.fit()
        exec_context.set_progress(0.8)

        in_samples = pd.Series(dtype=np.float64)

        preds_col = trained_model.predict(
            start=1, dynamic=self.sarima_params.predictor_params.dynamic_check
        )
        in_samples = pd.concat([in_samples, preds_col])

        # reverse log transformation for in-sample values
        if self.sarima_params.learner_params.natural_log:
            in_samples = np.exp(in_samples)

        # combine residuals and in-samples
        in_samps_residuals = pd.concat([trained_model.resid, in_samples], axis=1)
        in_samps_residuals.columns = ["Residuals", "In-Samples"]

        # make out-of-sample forecasts
        forecasts = trained_model.forecast(
            steps=self.sarima_params.predictor_params.number_of_forecasts
        ).to_frame(name="Forecasts")

        # reverse log transformation for forecasts
        if self.sarima_params.learner_params.natural_log:
            forecasts = np.exp(forecasts)

        # populate model coefficients
        coeffs_and_stats = self.get_coeffs_and_stats(trained_model)

        model_binary = pickle.dumps(trained_model)

        exec_context.set_progress(0.9)
        return (
            knext.Table.from_pandas(forecasts),
            knext.Table.from_pandas(in_samps_residuals),
            knext.Table.from_pandas(coeffs_and_stats),
            model_binary,
        )

    def __validate(self, column):
        if kutil.check_missing_values(column):
            missing_count = kutil.count_missing_values(column)
            raise knext.InvalidParametersError(
                f"""There are "{missing_count}" number of missing values in the target column."""
            )

        # validate that enough values are being provided to train the SARIMA model
        set_val = set(
            [
                # p
                self.sarima_params.learner_params.ar_order_param,
                # q
                self.sarima_params.learner_params.ma_order_param,
                # S * P
                self.sarima_params.learner_params.seasonal_period_param
                * self.sarima_params.learner_params.seasoanal_ar_order_param,
                # S*Q
                self.sarima_params.learner_params.seasonal_period_param
                * self.sarima_params.learner_params.seasoanal_ma_order_param,
            ]
        )
        num_of_rows = kutil.number_of_rows(column)
        if num_of_rows < max(set_val):
            raise knext.InvalidParametersError(
                f"Number of rows must be greater than maximum lag: '{max(set_val)}' to train the model. The maximum lag is the max of p, q, s*P, and s*Q."
            )

    def get_coeffs_and_stats(self, model):
        # estimates of the parameter coefficients
        coeff = model.params.to_frame()

        # calculate standard deviation of the parameters in the coefficients
        coeff_errors = model.bse.to_frame().reset_index()
        coeff_errors["index"] = coeff_errors["index"].apply(lambda x: x + " Std. Err")
        coeff_errors = coeff_errors.set_index("index")

        # extract log likelihood of the trained model
        log_likelihood = pd.DataFrame(
            data=model.llf, index=["Log Likelihood"], columns=[0]
        )

        # extract AIC (Akaike Information Criterion)
        aic = pd.DataFrame(data=model.aic, index=["AIC"], columns=[0])

        # extract BIC (Bayesian Information Criterion)
        bic = pd.DataFrame(data=model.bic, index=["BIC"], columns=[0])

        # extract Mean Squared Error
        mse = pd.DataFrame(data=model.mse, index=["MSE"], columns=[0])

        # extract Mean Absolute error
        mae = pd.DataFrame(data=model.mae, index=["MAE"], columns=[0])

        summary = pd.concat(
            [coeff, coeff_errors, log_likelihood, aic, bic, mse, mae]
        ).rename(columns={0: "Value"})

        return summary
