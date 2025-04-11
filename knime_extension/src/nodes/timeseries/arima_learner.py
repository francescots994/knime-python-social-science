import logging
import knime.extension as knext
from util import utils as kutil
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
from copy import deepcopy
from statsmodels.tsa.stattools import kpss
import random

LOGGER = logging.getLogger(__name__)



@knext.parameter_group("Non Seasonal Parameters")
class non_seasonal_params:
    """Non-seasonal parameters constraints for the SARIMA model."""

    max_ar = knext.IntParameter(
        label="Max AR Order (p)",
        description="The maximum number of lagged observations to enforce in the optimization.",
        default_value=8,
        min_value=0,
    )
    max_i = knext.IntParameter(
        label="Max I Order (d)",
        description="The maximum number of times to apply differencing to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )
    max_ma = knext.IntParameter(
        label="Max MA Order (q)",
        description="The maximum number of lagged forecast errors to enforce in the optimization.",
        default_value=8,
        min_value=0,
    )


@knext.parameter_group("Seasonal Parameters")
class seasonal_params:
    """Seasonal parameters constraints for the SARIMA model."""

    max_s_ar = knext.IntParameter(
        label="Max Seasonal AR Order (P)",
        description="The maximum number of seasonally lagged observations to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )
    max_s_i = knext.IntParameter(
        label="Max Seasonal I Order (D)",
        description="The maximum number of times to apply seasonal differencing to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )
    max_s_ma = knext.IntParameter(
        label="Max Seasonal MA Order (Q)",
        description="The maximum number of seasonal lagged forecast errors to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )




@knext.node(
    name="Auto SARIMA Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/SARIMA_Forecaster.png",
    category=kutil.category_timeseries,
    id="auto_sarima_learner",
)
@knext.input_table(
    name="Input Data",
    description="Table containing training data for fitting the SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
)
@knext.output_table(
    name="In-sample Predictions and Residuals",
    description="In sample model prediction values for the configured column and the residuals (the difference between observed value and the predicted output).",
)
@knext.output_table(
    name="Coefficients and Statistics",
    description="Table containing fitted model coefficients, variance of residuals (sigma2), and several model metrics along with their standard errors.",
)
@knext.output_binary(
    name="Model",
    description="Pickled model object that can be used by the SARIMA (Apply) node to generate different forecast lengths without refitting the model",
    id="auto_sarima.model",
)
class SarimaLearner:
    """
    Trains and generates a forecast with a (S)ARIMA Model

    Trains and generates a forecast using a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model. The SARIMA models captures temporal structures in time series data in the following components:

    - **AR (AutoRegressive):** Relationship between the current observation and a number (p) of lagged observations.
    - **I (Integrated):** Degree (d) of differencing required to make the time series stationary.
    - **MA (Moving Average):** Time series mean and the relationship between the current forecast error and a number (q) of lagged forecast errors.

    *Seasonal versions of these components operate similarly, with lag intervals equal to the seasonal period (S).*
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
        description="Specify the length of the Seasonal Period",
        default_value=2,
        min_value=2,
    )
    dynamic_check = knext.BoolParameter(
        label="Generate in-samples dynamically",
        description="Check this box to use in-sample prediction as lagged values. Otherwise use true values.",
        default_value=False,
    )
    natural_log = knext.BoolParameter(
        label="Log-transform data for modeling",
        description="Optionally log your target column before model fitting and exponentiate the forecast before output. This may help reduce variance in the training data.",
        default_value=False,
    )

    # The parameters constraints for the automatic ARIMA model
    non_seasonal_params = non_seasonal_params()
    seasonal_params = seasonal_params()
    max_ar = non_seasonal_params.max_ar
    max_i = non_seasonal_params.max_i
    max_ma = non_seasonal_params.max_ma
    max_s_ar = seasonal_params.max_s_ar
    max_s_i = seasonal_params.max_s_i
    max_s_ma = seasonal_params.max_s_ma


    def configure(self, configure_context, input_schema):
        
        # Checks that the given column is not None and exists in the given schema. If none is selected it returns the first column that is compatible with the provided function. If none is compatible it throws an exception.
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

        insamp_res_schema = knext.Schema([knext.double(), knext.double()], ["In-Samples", "Residuals"])
        model_summary_schema = knext.Column(knext.double(), "Value")
        binary_model_schema = knext.BinaryPortObjectSpec("auto_sarima.model")

        return (
            insamp_res_schema,
            model_summary_schema,
            binary_model_schema,
        )

    def execute(
            self,
            exec_context: knext.ExecutionContext,
            input: knext.Table
        ):
        
        df: pd.DataFrame
        df = input.to_pandas()
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
        self.__validate_col(target_col) # check for missing values
        exec_context.set_progress(0.2)

        params_constraints = (self.max_ar, self.max_i, self.max_ma, self.max_s_ar, self.max_s_i, self.max_s_ma)
        best_params = self.params_optimization_loop(target_col, 
                                self.seasonal_period_param, 
                                params_constraints, 
                                exec_context = exec_context,
                                step_size=1, 
                                anneal_steps=5, 
                                mcmc_steps=2, 
                                beta0=0.1,
                                beta1=10.0, 
                                seed=None)
        exec_context.set_progress(0.8)
        
        model = SARIMAX(
            endog = target_col,
            order = (best_params['p'], best_params['d'], best_params['q']),
            seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], self.seasonal_period_param)
        )
        trained_model = model.fit()
        exec_context.set_progress(0.9)

        # produce in-sample predictions for the whole series and put it in a Pandas Series
        in_samples_series = trained_model.predict(start=1, dynamic=self.dynamic_check)
        in_samples = pd.DataFrame(in_samples_series)
        # reverse log transformation for in-sample values
        if self.natural_log:
            in_samples = np.exp(in_samples)

        # produce residuals for the whole series based on the predictions just made and put it in a Pandas Series
        residuals_series = trained_model.resid[1:]
        residuals = pd.DataFrame(residuals_series)

        # combine the two dfs
        in_samps_and_residuals = pd.concat([in_samples, residuals], axis=1)
        in_samps_and_residuals.columns = ["In Sample Predictions", "Residuals"]

        # populate model coefficients
        coeffs_and_stats = self.get_coeffs_and_stats(trained_model, best_params)

        model_binary = pickle.dumps(trained_model)

        exec_context.set_progress(0.9)
        return (
            knext.Table.from_pandas(in_samps_and_residuals, row_ids = "keep"),
            knext.Table.from_pandas(coeffs_and_stats, row_ids = "keep"),
            model_binary,
        )

    def __validate_col(self, column):
        if kutil.check_missing_values(column):
            missing_count = kutil.count_missing_values(column)
            raise knext.InvalidParametersError(
                f"""There are "{missing_count}" number of missing values in the target column."""
            )

    def __validate_params(self, column, p, q, P, Q):
        # validate that enough values are being provided to train the SARIMA model
        S = self.seasonal_period_param
        set_val = set(
            [p, q, 
             S * P, 
             S * Q]
        )
        num_of_rows = kutil.number_of_rows(column)
        if num_of_rows < max(set_val):
            # raise knext.InvalidParametersError(
            #     f"Number of rows must be greater than maximum lag: '{max(set_val)}' to train the model. The maximum lag is the max of p, q, s*P, and s*Q."
            return False

        # handle Q > 0 and q >= s
        if ((P > 0) and (p >= S) or
            #"Autoregressive terms overlap with seasonal autogressive terms, p should be less than S when using seasonal auto regressive terms"
            (Q > 0) and (q >= S)):
            #"Moving average terms overlap with seasonal moving average terms, q should be less than S when using seasonal moving average terms."
            return False

        return True


    def get_coeffs_and_stats(self, model, best_params):
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

        p = pd.DataFrame(data=best_params['p'], index=["Best p parameter"], columns=[0])
        d = pd.DataFrame(data=best_params['d'], index=["Best d parameter"], columns=[0])
        q = pd.DataFrame(data=best_params['q'], index=["Best q parameter"], columns=[0])
        P = pd.DataFrame(data=best_params['P'], index=["Best P parameter"], columns=[0])
        D = pd.DataFrame(data=best_params['D'], index=["Best D parameter"], columns=[0])
        Q = pd.DataFrame(data=best_params['Q'], index=["Best Q parameter"], columns=[0])

        # concatenate all metrics and parameters
        summary = pd.concat(
            [coeff, coeff_errors, log_likelihood, aic, bic, mse, mae, p, d, q, P, D, Q]
        ).rename(columns={0: "Value"})

        return summary
        
        
    def find_optimal_integration_params(self, series, seasonality, max_i=5, max_is=5, alpha=0.05):
        """
        Perform the KPSS test to check for stationarity in a time series.

        Parameters:
        - series: Pandas Series (Time Series Data)
        - seasonality: Seasonal period (integer)
        - max_i: Maximum iterations for d parameter
        - max_is: Maximum iterations for D parameter
        - alpha: Significance level (default=0.05)

        Returns:
        - Tuple (d parameter, D parameter)
        """
        # Initialize d and D parameters
        d = 0
        D = 0

        # Check for seasonal stationarity (D parameter)
        for _ in range(max_is):
            p_value_ds = kpss(series, regression="c", nlags="auto")[1]
            if p_value_ds >= alpha:
                break
            # Apply seasonal differencing
            series = deepcopy(series.diff(seasonality).dropna())
            D += 1

        # Check for trend stationarity (d parameter)
        for _ in range(max_i):
            p_value_d = kpss(series, regression="c", nlags="auto")[1]
            if p_value_d >= alpha:
                break
            # Apply trend differencing
            series = deepcopy(series.diff().dropna())
            d += 1

        return d, D

    def propose_initial_params(self, d, D, max_p=8, max_q=8, max_ps=5, max_qs=5):
        """
        Proposes initial SARIMA hyperparameters.
        """
        p = round(max_p / 2)
        q = round(max_q / 2)
        P = round(max_ps / 2)
        Q = round(max_qs / 2)

        return {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q}

    def propose_new_params(self, series, current_params, max_p=8, max_q=8, max_ps=5, max_qs=5, step_size=1):
        """
        Proposes new random SARIMA hyperparameter(s) based on the current one with a small random step.
        """

        updated_params = deepcopy(current_params)
        threshold = np.random.rand()

        if threshold <= (1/2): # update trend parameters (p, q)

            current_p, current_q = updated_params['p'], updated_params['q']

            new_p, new_q = current_p + (np.random.choice([-1, 1]) * step_size), current_q + (np.random.choice([-1, 1]) * step_size)

            new_p = min(max_p, new_p)
            new_q = min(max_q, new_q)

            updated_params['p'], updated_params['q'] = new_p, new_q

        elif threshold > (1/2): # update seasonal parameters (P, Q)

            current_ps, current_qs = updated_params['P'], updated_params['Q']

            new_ps, new_qs = current_ps + (np.random.choice([-1, 1]) * step_size), current_qs + (np.random.choice([-1, 1]) * step_size)

            new_ps = min(max_ps, new_ps)
            new_qs = min(max_qs, new_qs)

            updated_params['P'], updated_params['Q'] = new_ps, new_qs

        if self.__validate_params(series, updated_params['p'], updated_params['q'], updated_params['P'], updated_params['Q']):
            return updated_params
        else:
            return current_params



    def evaluate_arima_model(self, series, params, seasonality):
        """
        Fits an ARIMA model and returns the AIC value as a score.
        """
        try:
            # AIC as scoring metric
            model_fit = SARIMAX(
                endog = series,
                order = (params['p'], params['d'], params['q']), # p, d, q order parameters for arima
                seasonal_order = (params['P'], params['D'], params['Q'], seasonality) # P, D, Q seasonal order parameters
            ).fit()

            return model_fit.aic
        except:
            return np.inf

    def accept_new_params(self, delta_cost, beta):
        # If the cost doesn't increase, we always accept
        if delta_cost <= 0:
            return True
        # If the cost increases and beta is infinite (last iteration), we always reject
        if beta == np.inf:
            return False

        p = np.exp(-beta * delta_cost)
        return np.random.rand() < p

    # The simulated annealing generic solver. NOTE: The default beta0 and beta1 are arbitrary.
    def params_optimization_loop(self, series, seasonality, 
            params_constraints, exec_context: knext.ExecutionContext, 
            step_size=1, anneal_steps = 10, mcmc_steps = 100,
            beta0 = 0.1, beta1 = 10.0, seed = None):
        # Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)

        # Set up the list of betas.
        beta_list = np.zeros(anneal_steps)
        # All but the last one are evenly spaced between beta0 and beta1 (included)
        beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
        # The last one is set to infinty
        beta_list[-1] = np.inf

        max_p, max_i, max_q, max_ps, max_is, max_qs = params_constraints

        d, D = self.find_optimal_integration_params(series, seasonality, max_i, max_is, alpha=0.05)
        current_params = self.propose_initial_params(d, D, max_p, max_q, max_ps, max_qs)

        current_cost = self.evaluate_arima_model(series, current_params, seasonality)

        # Keep the best cost seen so far, and its associated configuration.
        best_params = deepcopy(current_params)
        best_cost = current_cost

        # Main loop of the annaling: Loop over the betas
        for beta in beta_list:
            # At each beta, we want to record the acceptance rate, so we need a counter for the number of accepted moves
            accepted_moves = 0
            # For each beta, perform a number of MCMC steps
            for t in range(mcmc_steps):
                proposed_params = self.propose_new_params(series, current_params, step_size)
                delta_cost = self.evaluate_arima_model(series, proposed_params, seasonality) - current_cost
                
                # Metropolis rule
                if self.accept_new_params(delta_cost, beta):
                    current_params = deepcopy(proposed_params)
                    current_cost += delta_cost
                    accepted_moves += 1

                    if current_cost <= best_cost:
                        best_cost = current_cost
                        best_params = deepcopy(current_params)
            
            exec_context.set_warning(
                f'Iteration: {beta_list.tolist().index(beta)}, beta: {beta}, accept_freq: {accepted_moves/mcmc_steps}, best params: {best_params}, best cost: {best_cost}'
                )

        # Return the best instance
        return best_params