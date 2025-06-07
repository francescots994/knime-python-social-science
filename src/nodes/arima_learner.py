import logging
import knime.extension as knext
from util import utils as kutil
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from statsmodels.tsa.stattools import kpss
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning


LOGGER = logging.getLogger(__name__)


@knext.parameter_group("Non Seasonal Parameters")
class NonSeasonalParams:
    """
    Non-seasonal parameters constraints for the SARIMA model.
    """

    max_ar = knext.IntParameter(
        label="Max AR Order (p)",
        description="The maximum order of lagged observations to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )
    max_i = knext.IntParameter(
        label="Max I Order (d)",
        description="The maximum order of differencing to enforce in the optimization.",
        default_value=2,
        min_value=0,
    )
    max_ma = knext.IntParameter(
        label="Max MA Order (q)",
        description="The maximum order of lagged forecast errors to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )


@knext.parameter_group("Seasonal Parameters")
class SeasonalParams:
    """
    Seasonal parameters constraints for the SARIMA model.
    """

    max_s_ar = knext.IntParameter(
        label="Max Seasonal AR Order (P)",
        description="The maximum order of seasonally lagged observations to enforce in the optimization.",
        default_value=3,
        min_value=0,
    )
    max_s_i = knext.IntParameter(
        label="Max Seasonal I Order (D)",
        description="The maximum order of seasonal differencing to enforce in the optimization.",
        default_value=2,
        min_value=0,
    )
    max_s_ma = knext.IntParameter(
        label="Max Seasonal MA Order (Q)",
        description="The maximum order of seasonal lagged forecast errors to enforce in the optimization.",
        default_value=3,
        min_value=0,
    )


@knext.parameter_group("Optimization Loop Parameters")
class OptimizationLoopParams:
    """
    These parameters control the annealing process and the MCMC steps used to find the optimal SARIMA parameters.
    """

    anneal_steps = knext.IntParameter(
        label="Number of Annealing Steps",
        description="Number of different temperature levels (betas) in the annealing schedule. The higher the number, the more thorough the search for optimal parameters, but it will take longer to run.",
        default_value=5,
        min_value=2,
        is_advanced=True,
    )
    mcmc_steps = knext.IntParameter(
        label="MCMC Steps per Annealing Step",
        description="Number of MCMC steps (parameter proposals) to perform at each temperature level. The higher the number, the more thorough the search for optimal parameters, but it will take longer to run.",
        default_value=10,
        min_value=1,
        is_advanced=True,
    )
    beta0 = knext.DoubleParameter(
        label="Initial Annealing Temperature (beta0)",
        description="Initial (lowest) inverse temperature. Corresponds to high tolerance for accepting worse solutions. Set to 0.1 by default.",
        default_value=0.1,
        min_value=0.001,
        is_advanced=True,
    )
    beta1 = knext.DoubleParameter(
        label="Final Annealing Temperature (beta1)",
        description="Final (highest) finite inverse temperature before switching to infinity. Corresponds to low tolerance for accepting worse solutions. Set to 10.0 by default.",
        default_value=3.0,
        min_value=0.001,
        is_advanced=True,
    )
    step_size = knext.IntParameter(
        label="Step Size for Parameter Proposals",
        description="A step size of 1 (default) will propose new parameters higher or lower by 1. Be aware that a higher step size will be effective with looser parameter constraints.",
        default_value=1,
        min_value=1,
        max_value=3,
        is_advanced=True,
    )


@knext.node(
    name="Auto-SARIMA Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/SARIMA_Forecaster.png",
    category=kutil.category_timeseries,
    id="auto_sarima_learner",
)
@knext.input_table(
    name="Input Data",
    description="Table containing training data for the Auto-SARIMA model, must contain a numeric target column with no missing values.",
)
@knext.output_table(
    name="In-sample Predictions and Residuals",
    description="In-sample model prediction values for the configured column and the residuals (the difference between observed value and the predicted output).",
)
@knext.output_table(
    name="Coefficients and Statistics",
    description="Table containing optimal parameters, fitted model coefficients, variance of residuals, and several model metrics along with their standard errors.",
)
@knext.output_binary(
    name="Model",
    description="Pickled model object that can be used by the Auto-SARIMA Predictor node to generate different forecast lengths without refitting the model.",
    id="auto_sarima.model",
)
class AutoSarimaLearner:
    """
    Automatically finds the optimal parameters for and trains a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model on a given time series. This model is the SARIMAX class from the statsmodels library.

    **Model Overview:**

    SARIMA models capture temporal dependencies in time series data through several components:

    - **AR (AutoRegressive - p):** Models the relationship between an observation and a number (`p`) of lagged observations.

    - **I (Integrated - d):** Represents the degree (`d`) of non-seasonal differencing required to make the time series stationary.
    
    - **MA (Moving Average - q):** Models the relationship between an observation and a residual error from a moving average model applied to (`q`) lagged observations.
    
    - **Seasonal Components (P, D, Q, S):** Analogous to their non-seasonal counterparts (p, d, q), but applied at lags corresponding to the seasonal period (`S`). `P` is the seasonal AR order, `D` is the seasonal differencing order, and `Q` is the seasonal MA order.

    **Parameter Optimization:**
    
    This node employs a two-step search strategy to identify the best SARIMA parameters (p, d, q, P, D, Q) for the input data:
    
    1.  **Differencing Orders (d, D):** The orders of non-seasonal (`d`) and seasonal (`D`) differencing are determined first using repeated Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests. Differencing is applied until the series is deemed stationary (p-value >= 0.05) or the maximum orders (`max_i`, `max_s_i`) specified by the user are reached.
    
    2.  **AR and MA Orders (p, q, P, Q):** The remaining parameters are optimized using a Simulated Annealing algorithm. This algorithm intelligently explores different combinations of `p`, `q`, `P`, and `Q` (within the user-defined maximums `max_ar`, `max_ma`, `max_s_ar`, `max_s_ma`) to minimize the Akaike Information Criterion (AIC) of the fitted model. The annealing process (controlled by `anneal_steps`, `mcmc_steps`, `beta0`, `beta1`, `step_size`) allows the search to escape local optima and find a globally better parameter set.

    **Key Parameters & Behavior:**
    
    -   `input_column`: The target time series column. Must be numeric and contain no missing values.
    
    -   `seasonal_period_param (S)`: Defines the length of the seasonal cycle (e.g., 12 for monthly data with yearly seasonality). Setting S=0 disables seasonal components, effectively making the node search for the best **ARIMA** model (P, D, Q will be ignored).
    
    -   `natural_log`: If checked, applies a natural logarithm transformation to the data before modeling. This can help stabilize variance but requires all values in the target column to be positive. Forecasts are automatically exponentiated back to the original scale.
    
    -   Parameter Constraints (`max_ar`, `max_i`, etc.): These define the upper bounds for the parameter search.
    
    -   Optimization Parameters (`anneal_steps`, `mcmc_steps`, etc.): Control the thoroughness and behavior of the simulated annealing search. `beta0` must be less than `beta1`.

    **Outputs:**

    1.  In-sample predictions and residuals.

    2.  Optimal parameters, model coefficients, standard errors, and goodness-of-fit statistics (AIC, BIC, MSE, MAE).
    
    3.  A pickled `SARIMAXResults` object representing the trained model, usable by the Auto-SARIMA Predictor node.
    """

    # General settings for the SARIMA model
    input_column = knext.ColumnParameter(
        label="Target Column",
        description="Table containing training data for fitting the SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
        port_index=0,
        column_filter=kutil.is_numeric,
    )
    seasonal_period_param = knext.IntParameter(
        label="Seasonal Period (s)",
        description="Specify the length of the Seasonal Period. Set = 0 for a non-seasonal ARIMA model.",
        default_value=2,
        min_value=0,
    )
    natural_log = knext.BoolParameter(
        label="Log-transform data for modelling",
        description="Optionally log your target column before model fitting and exponentiate the forecast before output. This may help reduce variance in the training data.",
        default_value=False,
    )

    # The parameters constraints for the automatic ARIMA model
    non_seasonal_params = NonSeasonalParams()
    seasonal_params = SeasonalParams()

    # Optimization loop parameters (Advanced)
    optimization_loop_params = OptimizationLoopParams()

    def configure(
        self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema
    ) -> knext.Schema:
        
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the first column that is compatible with the provided function. If none is compatible it throws an exception.
        self.input_column = kutil.column_exists_or_preset(
            configure_context,
            self.input_column,
            input_schema,
            kutil.is_numeric,
        )

        if self.optimization_loop_params.beta0 >= self.optimization_loop_params.beta1:
            raise knext.InvalidParametersError(
                "The initial annealing temperature (beta0) must be less than the final annealing temperature (beta1)."
            )

        insamp_res_schema = knext.Schema(
            [knext.double(), knext.double()], ["In-Sample Predictions", "Residuals"]
        )
        model_summary_schema = knext.Column(knext.double(), "Model Summary")
        binary_model_schema = knext.BinaryPortObjectSpec("auto_sarima.model")

        return (
            insamp_res_schema,
            model_summary_schema,
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input: knext.Table):

        df = input.to_pandas()
        target_col = df[self.input_column]

        # if enabled, apply natural logarithm transformation. If negative values are present, raise an error
        if self.natural_log:
            num_negative_vals = kutil.count_negative_values(target_col)

            if num_negative_vals > 0:
                raise knext.InvalidParametersError(
                    f" There are '{num_negative_vals}' non-positive values in the target column."
                )
            target_col = np.log(target_col)

        exec_context.set_progress(0.1)

        # check if the number of obsevations is not lower than seasonality
        if len(target_col) < self.seasonal_period_param:
            raise knext.InvalidParametersError(
                f"The number of observations in the target column ({len(target_col)}) is lower than the seasonal period ({self.seasonal_period_param})."
            )

        # check for missing values
        self.__validate_col(target_col)

        exec_context.set_progress(0.2)

        best_params = self.__params_optimization_loop(
            target_col,
            exec_context=exec_context,
        )

        exec_context.set_progress(0.8)

        trained_model = self.__evaluate_arima_model(
            target_col,
            best_params,
            exec_context=exec_context,
        )[1]
        exec_context.set_progress(0.9)

        # produce in-sample predictions for the whole series and put it in a Pandas Series
        in_samples_series = trained_model.predict()
        in_samples = pd.DataFrame(in_samples_series)
        # reverse log transformation for in-sample values
        if self.natural_log:
            in_samples = np.exp(in_samples)

        # produce residuals for the whole series based on the predictions just made and put it in a Pandas Series
        residuals_series = trained_model.resid
        residuals = pd.DataFrame(residuals_series)

        # combine the two dfs
        in_samps_and_residuals = pd.concat([in_samples, residuals], axis=1)
        in_samps_and_residuals.columns = ["In-Sample Predictions", "Residuals"]

        # populate model coefficients
        coeffs_and_stats = self.__get_coeffs_and_stats(trained_model, best_params)

        model_binary = pickle.dumps(trained_model)

        exec_context.set_progress(0.99)

        return (
            knext.Table.from_pandas(in_samps_and_residuals, row_ids="keep"),
            knext.Table.from_pandas(coeffs_and_stats, row_ids="keep"),
            model_binary,
        )

    def __validate_col(self, column):
        """
        Validates the input time series column for missing values.

        Checks if the provided Pandas Series contains any missing (NaN) values.
        If missing values are found, it raises an InvalidParametersError.

        Parameters:
        - column: pd.Series
            The time series data to validate.

        Raises:
        - knext.InvalidParametersError
            If the input column contains missing values.
        """
        if kutil.check_missing_values(column):
            missing_count = kutil.count_missing_values(column)
            raise knext.InvalidParametersError(
                f"""There are "{missing_count}" number of missing values in the target column."""
            )

    def __validate_params(self, column, p, q, P, Q):
        """
        Validates the proposed (S)ARIMA parameters against the time series data length and potential overlaps.

        This function performs two checks:
        1. Ensures the time series (`column`) has enough data points to estimate the model given the highest order parameter (p, q, S*P, S*Q).
        2. Checks for invalid parameter combinations where seasonal and non-seasonal AR or MA terms might overlap (e.g., p >= S when P > 0, or q >= S when Q > 0).

        Parameters:
        - column: pd.Series
            The time series data.
        - p: int
            The non-seasonal AR order.
        - q: int
            The non-seasonal MA order.
        - P: int
            The seasonal AR order.
        - Q: int
            The seasonal MA order.

        Returns:
        - bool
            True if the parameters are valid for the given series, False otherwise.
        """
        S = self.seasonal_period_param
        set_val = set([p, q, S * P, S * Q])
        num_of_rows = kutil.number_of_rows(column)

        if num_of_rows < max(set_val):
            return False

        # handle P, Q > 0 and p, q >= S
        if (
            (P > 0)
            and (p >= S)
            or
            # "Autoregressive terms overlap with seasonal autogressive terms, p should be less than S when using seasonal auto regressive terms"
            (Q > 0)
            and (q >= S)
        ):
            # "Moving average terms overlap with seasonal moving average terms, q should be less than S when using seasonal moving average terms."
            return False

        return True

    def __get_coeffs_and_stats(self, model, best_params):
        """
        Extracts coefficients, standard errors, and various statistical metrics from a fitted SARIMAX model.

        This function takes a fitted `statsmodels.tsa.statespace.sarimax.SARIMAXResults` object
        and the dictionary of best parameters found during optimization. It compiles these into a
        single Pandas DataFrame suitable for output.

        Parameters:
        - model: statsmodels.tsa.statespace.sarimax.SARIMAXResults
            The fitted SARIMAX model object.
        - best_params: dict
            A dictionary containing the optimal parameters found: {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.

        Returns:
        - pd.DataFrame
            A DataFrame containing the model summary, including:
            - Parameter coefficients
            - Standard errors of coefficients
            - Log Likelihood
            - AIC (Akaike Information Criterion)
            - BIC (Bayesian Information Criterion)
            - MSE (Mean Squared Error)
            - MAE (Mean Absolute Error)
            - The best p, d, q, P, D, Q parameters used.
        """
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

        p = pd.DataFrame(data=best_params["p"], index=["Best p parameter"], columns=[0])
        d = pd.DataFrame(data=best_params["d"], index=["Best d parameter"], columns=[0])
        q = pd.DataFrame(data=best_params["q"], index=["Best q parameter"], columns=[0])
        P = pd.DataFrame(data=best_params["P"], index=["Best P parameter"], columns=[0])
        D = pd.DataFrame(data=best_params["D"], index=["Best D parameter"], columns=[0])
        Q = pd.DataFrame(data=best_params["Q"], index=["Best Q parameter"], columns=[0])

        # concatenate all metrics and parameters
        summary = pd.concat(
            [coeff, coeff_errors, log_likelihood, aic, bic, mse, mae, p, d, q, P, D, Q]
        ).rename(columns={0: "Model Summary"})

        return summary

    def __find_optimal_integration_params(
        self, series,
    ):
        """
        Determines the optimal orders of non-seasonal (d) and seasonal (D) differencing required to make the time series stationary using the KPSS test.

        The function iteratively applies seasonal differencing (up to `max_i_s` times) and then non-seasonal differencing (up to `max_i` times).
        In each step, it performs the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test. The null hypothesis of the KPSS test is that the series is stationary around a deterministic trend.
        Differencing continues as long as the p-value of the KPSS test is below the significance level `alpha`, indicating non-stationarity. The number of differencing steps taken determines the values of D and d.
        Note: This function modifies the input `series` by applying differencing directly.

        Parameters:
        - series: pd.Series
            The input time series data. This series will be modified in place.
        - seasonality: int
            The seasonal period of the time series. Used for seasonal differencing.
        - max_i: int, optional (default=2)
            The maximum order of non-seasonal differencing (d) to test.
        - max_i_s: int, optional (default=2)
            The maximum order of seasonal differencing (D) to test.
        - alpha: float, optional (default=0.05)
            The significance level for the KPSS test. If the p-value is greater than or equal to alpha, the series is considered stationary.

        Returns:
        - tuple (int, int)
            A tuple containing the determined optimal non-seasonal differencing order (d) and seasonal differencing order (D).
        """
        # Initialize d and D parameters
        d = 0
        D = 0

        # significance level for KPSS test
        alpha = 0.05 

        seasonality = self.seasonal_period_param

        # Check for seasonal stationarity (D parameter)
        for _ in range(self.seasonal_params.max_s_i):
            p_value_d_s = kpss(series)[1]
            if p_value_d_s >= alpha:
                break
            # Apply seasonal differencing
            series = series.diff(seasonality).dropna()
            D += 1

        # Check for trend stationarity (d parameter)
        for _ in range(self.non_seasonal_params.max_i):
            p_value_d = kpss(series)[1]
            if p_value_d >= alpha:
                break
            # Apply non seasonal differencing
            series = series.diff().dropna()
            d += 1

        return d, D

    def __propose_initial_params(self, d, D):
        """
        Proposes initial simple parameters for the SARIMA model optimization process.

        Given the pre-determined differencing orders (d, D), this function sets the initial
        AR and MA orders (p, q, P, Q) to 1, unless the corresponding maximum allowed value
        (max_p, max_q, max_p_s, max_q_s) is 0. This provides a basic starting point for the
        simulated annealing optimization.

        Parameters:
        - d: int
            The non-seasonal differencing order.
        - D: int
            The seasonal differencing order.
        - max_p: int, optional (default=3)
            The maximum allowed non-seasonal AR order.
        - max_q: int, optional (default=3)
            The maximum allowed non-seasonal MA order.
        - max_p_s: int, optional (default=5)
            The maximum allowed seasonal AR order.
        - max_q_s: int, optional (default=5)
            The maximum allowed seasonal MA order.

        Returns:
        - dict
            A dictionary containing the initial proposed parameters:
            {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.
        """
        p = min([self.non_seasonal_params.max_ar, 1])
        q = min([self.non_seasonal_params.max_ma, 1])
        P = min([self.seasonal_params.max_s_ar, 1])
        Q = min([self.seasonal_params.max_s_ma, 1])

        return {"p": p, "d": d, "q": q, "P": P, "D": D, "Q": Q}

    def __propose_new_params(
        self,
        series,
        current_params,
        exec_context: knext.ExecutionContext,
    ):
        """
        Proposes a new set of SARIMA parameters by randomly adjusting the current parameters.

        This function is used within the simulated annealing loop. It takes the current set of
        parameters and randomly decides whether to adjust the non-seasonal (p, q) or seasonal (P, Q)
        orders based on a random threshold. The chosen orders are incremented or decremented by `step_size`
        (randomly chosen direction) or kept the same. The new parameters are constrained within the allowed maximums
        (max_p, max_q, max_p_s, max_q_s) and non-negativity. Finally, it validates the proposed
        parameters using `__validate_params`. If the proposed parameters are invalid, it logs a warning
        and returns the original `current_params`. If the random threshold logic fails to select
        parameters to update (an edge case), it raises an error.

        Parameters:
        - series: pd.Series
            The time series data, used for validation via `__validate_params`.
        - current_params: dict
            A dictionary containing the current parameters: {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.
        - exec_context: knext.ExecutionContext
            The execution context for logging warnings if validation fails.
        - max_p: int, optional (default=5)
            The maximum allowed non-seasonal AR order.
        - max_q: int, optional (default=5)
            The maximum allowed non-seasonal MA order.
        - max_p_s: int, optional (default=3)
            The maximum allowed seasonal AR order.
        - max_q_s: int, optional (default=3)
            The maximum allowed seasonal MA order.

        Returns:
        - dict
            A dictionary containing the newly proposed (and validated) parameters, or the `current_params`
            if the proposed ones were invalid.

        Raises:
        - knext.InvalidParametersError
            If the internal logic fails to update any parameters (should not typically happen with the current logic).
        """
        updated_params = current_params.copy()
        threshold = np.random.rand()

        step_size = self.optimization_loop_params.step_size
        max_p = self.non_seasonal_params.max_ar
        max_q = self.non_seasonal_params.max_ma
        max_p_s = self.seasonal_params.max_s_ar
        max_q_s = self.seasonal_params.max_s_ma

        steps = np.arange(1, step_size + 1)

        if threshold <= (1 / 2) or (
            max_p_s == 0 and max_q_s == 0
        ):  # update trend parameters (p, q)
            current_p, current_q = updated_params["p"], updated_params["q"]

            new_p, new_q = (
                current_p + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0]),
                current_q + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0]),
            )

            new_p = max([min([max_p, new_p]), 0])
            new_q = max([min([max_q, new_q]), 0])

            updated_params["p"], updated_params["q"] = new_p, new_q

        elif (
            threshold > (1 / 2) and (max_p_s > 0 and max_q_s > 0)
        ):  # update seasonal parameters (P, Q) only if they are not zero (seasonal model)
            current_ps, current_qs = updated_params["P"], updated_params["Q"]

            new_ps, new_qs = (
                current_ps + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0]),
                current_qs + (np.random.choice([-1, 0, 1]) * np.random.choice(steps, size=1)[0]),
            )

            new_ps = max(min(max_p_s, new_ps), 0)
            new_qs = max(min(max_q_s, new_qs), 0)

            updated_params["P"], updated_params["Q"] = new_ps, new_qs

        else:
            raise knext.InvalidParametersError(
                f"No parameters were updated because the conditions for updating parameters were not met. Threshold: {threshold}, constraints: {max_p_s}, {max_q_s}."
            )

        if self.__validate_params(
            series,
            updated_params["p"],
            updated_params["q"],
            updated_params["P"],
            updated_params["Q"],
        ):
            return updated_params
        else:
            LOGGER.info(
                f"Proposed parameters {updated_params} are invalid for the series. Retaining current parameters."
            )
            return current_params

    def __evaluate_arima_model(
        self,
        series, 
        params, 
        exec_context: knext.ExecutionContext
    ):
        """
        Fits a SARIMAX model with the given parameters and evaluates its AIC score.

        This function attempts to fit a `statsmodels.tsa.statespace.sarimax.SARIMAX` model
        using the provided time series, parameter dictionary, and seasonality. It captures
        the Akaike Information Criterion (AIC) as the primary evaluation metric (lower is better).
        It specifically checks for `ConvergenceWarning` during fitting using `warnings.catch_warnings`.
        If a `ConvergenceWarning` occurs or any other exception is raised during fitting, it logs a
        warning via the `exec_context` and returns an infinite AIC score and the potentially partially
        fitted model object (or None if fitting failed early). This heavily penalizes problematic
        parameters in the optimization process. Successful fits are also logged with their AIC.

        Parameters:
        - series: pd.Series
            The time series data to fit the model on.
        - params: dict
            A dictionary containing the SARIMA parameters: {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.
        - seasonality: int
            The seasonal period for the SARIMA model.
        - exec_context: knext.ExecutionContext
            The execution context for logging warnings about fitting issues or convergence.

        Returns:
        - tuple (float, statsmodels.tsa.statespace.sarimax.SARIMAXResults or None)
            A tuple containing:
            - float: The AIC score of the fitted model. Returns `np.inf` if fitting fails or results in a ConvergenceWarning.
            - SARIMAXResults or None: The fitted model object if successful, otherwise None or the object from a failed/non-converged fit.
        """
        aic_score = np.inf
        convergence_warning_occurred = False
        model_fit = None  # Initialize model_fit to handle potential early exceptions

        seasonality = self.seasonal_period_param

        try:
            # Use catch_warnings to capture any warnings during fit()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")  # Ensure all warnings are caught

                model = SARIMAX(
                    endog=series,
                    order=(params["p"], params["d"], params["q"]),
                    seasonal_order=(params["P"], params["D"], params["Q"], seasonality),
                )
                model_fit = model.fit(disp=False)
                aic_score = model_fit.aic

                # Check if any ConvergenceWarning was caught
                for warning in caught_warnings:
                    if issubclass(warning.category, ConvergenceWarning):
                        convergence_warning_occurred = True
                        break

        except Exception as e:
            # Handle potential errors during model fitting (e.g., invalid parameters)
            exec_context.set_warning(
                f"WARNING Error fitting SARIMAX with params {params}: {e}"
            )
            return (
                np.inf,
                model_fit,  # Return potentially partially fitted model or None
            )  # Treat errors as convergence issues or high cost

        if convergence_warning_occurred:
            # return infinity to make impossible to accept these parameters
            LOGGER.info(
                f"ConvergenceWarning occurred for params {params}. Model fitting failed, returning infinity."
            )
            return (
                np.inf,
                model_fit,
            )  # Return the model_fit object even if convergence failed

        LOGGER.info(
            f"Model fitted successfully for params {params}, AIC: {aic_score}"
        )

        return (aic_score, model_fit)

    def __accept_new_params(self, delta_cost, beta):
        """
        Decides whether to accept a new set of parameters based on the change in cost (AIC) and the current annealing temperature (beta).

        This function implements the Metropolis acceptance criterion used in simulated annealing:
        1. If the new parameters result in a lower or equal cost (`delta_cost` <= 0), they are always accepted.
        2. If the new parameters result in a higher cost (`delta_cost` > 0):
           - If `beta` is infinite (representing the final stage of annealing where only improvements are accepted), the move is rejected as in a greedy algorithm.
           - Otherwise, the move is accepted probabilistically based on the Boltzmann factor `exp(-beta * delta_cost)`. A random number is drawn from [0, 1); if it's less than the Boltzmann factor, the move is accepted. This allows the algorithm to occasionally escape local optima early in the process when beta is low (high temperature).

        Parameters:
        - delta_cost: float
            The change in the cost function (AIC) between the new parameters and the current parameters (new_cost - current_cost).
        - beta: float
            The inverse temperature parameter in the simulated annealing process. Higher beta means lower tolerance for accepting worse solutions. Can be `np.inf`.

        Returns:
        - bool
            True if the new parameters should be accepted, False otherwise.
        """
        # If the cost doesn't increase, we always accept
        if delta_cost <= 0:
            return True
        
        # If the cost increases and beta is infinite (last iteration), we always reject. Explicitly check delta_cost > 0 for clarity
        elif (beta == np.inf) and (delta_cost > 0):
            return False
        
        # In all other cases (delta_cost > 0 and beta < inf), accept based on probability p compared to a random draw from [0, 1)
        else: 
            p = np.exp(-beta * delta_cost)
            return np.random.rand() < p

    def __params_optimization_loop(
        self,
        series,
        exec_context: knext.ExecutionContext,
    ):
        """
        Performs hyperparameter optimization for the SARIMA model using a simulated annealing algorithm.

        This function searches for the optimal combination of non-seasonal (p, q) and seasonal (P, Q)
        parameters within the constraints provided by the user in the configuration dialogue. The differencing orders (d, D)
        are determined beforehand using `find_optimal_integration_params` and remain fixed.

        The algorithm works as follows:
        1. Determine initial d and D using KPSS tests (`find_optimal_integration_params`).
        2. Propose simple initial parameters (p=1, q=1, P=1, Q=1, respecting constraints) using `propose_initial_params`.
        3. Evaluate the initial model's AIC score (`evaluate_arima_model`).
        4. Initialize the best parameters and cost found so far.
        5. Start the simulated annealing loop:
           - Iterate through a schedule of inverse temperatures (`beta`), starting low (`beta0`) and increasing to high (`beta1`), ending with infinity.
           - For each `beta`, run a Markov Chain Monte Carlo (MCMC) simulation for `mcmc_steps`:
             - Propose new parameters (p, q, P, Q) by slightly modifying the current ones (`propose_new_params`).
             - Evaluate the AIC score of the model with the proposed parameters (`evaluate_arima_model`).
             - Calculate the change in cost (`delta_cost`).
             - Decide whether to accept the proposed parameters using the Metropolis criterion (`accept_new_params`).
             - If accepted, update the current parameters and cost.
             - If the accepted parameters yield the best cost seen so far, update the best parameters and cost.
           - Report progress and results for the current `beta`
        6. Return the best set of parameters found throughout the process.

        Parameters:
        - series: pd.Series
            The input time series data.
        - seasonality: int
            The seasonal period of the time series.
        - exec_context: knext.ExecutionContext
            The execution context for progress reporting and logging warnings.
        - anneal_steps: int, optional (default=5)
            The number of different temperature levels (betas) in the annealing schedule.
        - mcmc_steps: int, optional (default=10)
            The number of MCMC steps (parameter proposals) to perform at each temperature level.
        - beta0: float, optional (default=0.1)
            The initial (lowest) inverse temperature. Corresponds to high tolerance for accepting worse solutions.
        - beta1: float, optional (default=10.0)
            The final (highest) finite inverse temperature before switching to infinity. Corresponds to low tolerance.

        Returns:
        - dict
            A dictionary containing the best set of SARIMA parameters found:
            {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}.

        Raises:
        - RuntimeError: If the initial model evaluation fails even after trying a fallback simple model.
        """
        anneal_steps = self.optimization_loop_params.anneal_steps
        mcmc_steps = self.optimization_loop_params.mcmc_steps
        beta0 = self.optimization_loop_params.beta0
        beta1 = self.optimization_loop_params.beta1

        # Set up the list of betas.
        beta_list = np.zeros(anneal_steps)
        # All but the last one are evenly spaced between beta0 and beta1 (included)
        beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
        # The last one is set to infinty
        beta_list[-1] = np.inf
        # Set up the progress bar
        progress = np.linspace(0.2, 0.8, anneal_steps)

        if len(progress) != len(beta_list):
            raise knext.InvalidParametersError(
                "The number of progress steps must match the number of beta values."
            )

        LOGGER.info(
            "Preparing to find optimal integration parameters..."
        )
        # Create a copy to avoid modifying the original series outside this function scope
        series_copy_for_diff = series.copy()
        d, D = self.__find_optimal_integration_params(
            series_copy_for_diff
        )
        LOGGER.info(
            f"Optimal integration parameters found: d={d}, D={D}. Generating the remaining initial parameters..."
        )
        current_params = self.__propose_initial_params(
            d, D
        )

        LOGGER.info(
            f"Initial parameters: {current_params}. Evaluating initial model based on the AIC..."
        )
        current_cost = self.__evaluate_arima_model(
            series,
            current_params,
            exec_context=exec_context
        )[0]

        # Handle case where initial evaluation fails
        if current_cost == np.inf:
            LOGGER.info(
                f"Initial model evaluation failed (AIC=inf) for params {current_params}. Trying a simpler model (0,{d},0)(0,{D},0)..."
            )
            # Try a very simple model as fallback
            fallback_params = {"p": 0, "d": d, "q": 0, "P": 0, "D": D, "Q": 0}
            # Validate fallback params before trying to fit
            if self.__validate_params(
                series,
                fallback_params["p"],
                fallback_params["q"],
                fallback_params["P"],
                fallback_params["Q"],
            ):
                current_params = fallback_params
                current_cost = self.__evaluate_arima_model(
                    series,
                    current_params,
                    exec_context=exec_context
                )[0]
                if current_cost == np.inf:
                    raise RuntimeError(
                        f"Even the simplest fallback model (0,{d},0)(0,{D},0) failed to fit. Cannot proceed with optimization."
                    )
                else:
                    LOGGER.info(
                        f"Using fallback initial parameters {current_params} with AIC: {current_cost}"
                    )
            else:
                # If even the fallback is invalid (e.g., series too short), raise error
                raise RuntimeError(
                    f"Initial parameters {current_params} failed to fit and fallback parameters (0,{d},0)(0,{D},0) are invalid for the series length. Cannot proceed."
                )

        LOGGER.info(
            f"Initial model evaluated with AIC: {current_cost}. Starting optimization loop..."
        )

        # Keep the best criterion seen so far, and its associated configuration.
        best_params = current_params.copy()
        best_cost = current_cost

        # create a set of parameters already proposed by the algorithm
        proposed_params_cache = set()

        # Main loop of the simulated annealing process: Loop over the betas (the list of tolerances for moves that increase the cost)
        for beta in beta_list:
            # At each beta record the acceptance rate using a counter for the number of accepted moves
            accepted_moves = 0
            # For each beta, perform a number of MCMC steps
            for t in range(mcmc_steps):
                # Propose new parameters based on the current ones and the step size
                proposed_params = self.__propose_new_params(
                    series,
                    current_params,
                    exec_context,
                )
                # If the proposed parameters are already in the cache, skip this iteration. Else, add them to the cache.
                if tuple(proposed_params.items()) in proposed_params_cache:
                    LOGGER.info(
                        f"Proposed parameters {proposed_params} already evaluated. Moving to next parameters proposal."
                    )
                    continue
                else:
                    proposed_params_cache.add(tuple(proposed_params.items()))
                # Only evaluate if parameters actually changed
                if proposed_params != current_params:
                    new_cost = self.__evaluate_arima_model(
                            series,
                            proposed_params,
                            exec_context=exec_context,
                        )[0]
                    delta_cost = new_cost - current_cost
                    
                else:
                    continue
                LOGGER.info(
                    f"MCMC step: {t + 1}/{mcmc_steps} for beta: {beta} Proposed params: {proposed_params}, Delta cost: {delta_cost}"
                )

                # Metropolis rule
                if self.__accept_new_params(delta_cost, beta):
                    current_params = proposed_params.copy()
                    current_cost = new_cost
                    accepted_moves += 1

                    if current_cost <= best_cost:
                        best_cost = current_cost
                        best_params = current_params.copy()

            # Dynamic progress update based on the current beta
            exec_context.set_progress(progress[beta_list.tolist().index(beta)])

            # Print in the console the current beta, the acceptance rate, and the best parameters found so far
            LOGGER.info(
                f"Iteration: {beta_list.tolist().index(beta) + 1}, beta: {beta}, accept_freq: {accepted_moves / mcmc_steps}, best params: {best_params}, best cost: {best_cost}"
            )

        # Return the best instance
        LOGGER.debug(
            f"Optimization finished. Final best parameters: {best_params} with AIC: {best_cost:.2f}"
        )
        return best_params
