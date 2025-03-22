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



@knext.node(
    name="ARIMA Learner",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/SARIMA_Forecaster.png",
    category=kutil.category_timeseries,
    id="arima_learner",
)
@knext.input_table(
    name="Input Data",
    description="Table containing training data for fitting the SARIMA model, must contain a numeric target column with no missing values to be used for forecasting.",
)
@knext.output_table(
    name="In-sample Predictions",
    description="In sample model prediction values for the configured column.",
)
@knext.output_table(
    name="Residuals",
    description="In sample model prediction residuals (the difference between observed value and the predicted output).",
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


    # The parameters for the automatic ARIMA model
    max_ar = knext.IntParameter(
        label="Max AR Order (p)",
        description="The maximum number of lagged observations to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )
    min_ar = knext.IntParameter(
        label="Min AR Order (p)",
        description="The minimum number of lagged observations to enforce in the optimization. ",
        default_value=0,
        min_value=0,
    )
    max_i = knext.IntParameter(
        label="Max I Order (d)",
        description="The maximum number of times to apply differencing to enforce in the optimization.",
        default_value=2,
        min_value=0,
    )
    min_i = knext.IntParameter(
        label="Min I Order (d)",
        description="The minimum number of times to apply differencing to enforce in the optimization.",
        default_value=0,
        min_value=0,
    )
    max_ma = knext.IntParameter(
        label="Max MA Order (q)",
        description="The maximum number of lagged forecast errors to enforce in the optimization.",
        default_value=5,
        min_value=0,
    )
    min_ma = knext.IntParameter(
        label="Min MA Order (q)",
        description="The minimum number of lagged forecast errors to enforce in the optimization.",
        default_value=0,
        min_value=0,
    )
    max_s_ar = knext.IntParameter(
        label="Max Seasonal AR Order (P)",
        description="The maximum number of seasonally lagged observations to enforce in the optimization.",
        default_value=2,
        min_value=0,
    )
    min_s_ar = knext.IntParameter(
        label="Min Seasonal AR Order (P)",
        description="The minimum number of seasonally lagged observations to enforce in the optimization.",
        default_value=0,
        min_value=0,
    )
    max_s_i = knext.IntParameter(
        label="Max Seasonal I Order (D)",
        description="The maximum number of times to apply seasonal differencing to enforce in the optimization.",
        default_value=1,
        min_value=0,
    )
    min_s_i = knext.IntParameter(
        label="Min Seasonal I Order (D)",
        description="The minimum number of times to apply seasonal differencing to enforce in the optimization.",
        default_value=0,
        min_value=0,
    )
    max_s_ma = knext.IntParameter(
        label="Max Seasonal MA Order (Q)",
        description="The maximum number of seasonal lagged forecast errors to enforce in the optimization.",
        default_value=2,
        min_value=0,
    )
    min_s_ma = knext.IntParameter(
        label="Min Seasonal MA Order (Q)",
        description="The minimum number of seasonal lagged forecast errors to enforce in the optimization.",
        default_value=0,
        min_value=0,
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
        ):
        
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

        # handle P > 0 and p >= s
        if (self.min_s_ar > 0) and (
            self.min_ar >= self.seasonal_period_param
        ):
            configure_context.set_warning(
                "Autoregressive terms overlap with seasonal autogressive terms, p should be less than S when using seasonal auto regressive terms"
            )

        # handle Q > 0 and q >= s
        if (self.min_s_ma > 0) and (
            self.min_ma >= self.seasonal_period_param
        ):
            configure_context.set_warning(
                "Moving average terms overlap with seasonal moving average terms, q should be less than S when using seasonal moving average terms."
            )
        
        if (self.max_ar < self.min_ar or self.max_i < self.min_i or self.max_ma < self.min_ma or 
            self.max_s_ar < self.min_s_ar or self.max_s_i < self.min_s_i or self.max_s_ma < self.min_s_ma):
            configure_context.set_warning(
                "Max order parameters should be greater than or equal to min order parameters."
            )


        insamp_schema = knext.Column(knext.double(), "In-Samples")
        res_schema = knext.Column(knext.double(), "Residuals")
        model_summary_schema = knext.Column(knext.double(), "Value")
        binary_model_schema = knext.BinaryPortObjectSpec("auto_sarima.model") # change ID for model

        return (
            insamp_schema,
            res_schema,
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
        # self.__validate(target_col)
        exec_context.set_progress(0.3)

        if (self.min_ar == self.max_ar) and (self.min_i == self.max_i) and (
            self.min_ma == self.max_ma) and (self.min_s_ar == self.max_s_ar) and (
                self.min_s_i == self.max_s_i) and (self.min_s_ma == self.max_s_ma):
        # model initialization and training
            model = SARIMAX(
                target_col,
                order=(
                    self.min_ar,
                    self.min_i,
                    self.min_ma,
                ),
                seasonal_order=(
                    self.min_s_ar,
                    self.min_s_i,
                    self.min_s_ma,
                    self.seasonal_period_param,
                ),
            )
            exec_context.set_progress(0.5)
            trained_model = model.fit()
            exec_context.set_progress(0.8)

        else:
            
            params_constraints = (self.min_ar, self.max_ar, self.min_i, self.max_i, self.min_ma, self.max_ma, self.min_s_ar, self.max_s_ar, self.min_s_i, self.max_s_i, self.min_s_ma, self.max_s_ma)
            best_params = self.simann(target_col, self.seasonal_period_param, params_constraints, step_size=1, anneal_steps=5, mcmc_steps=10, beta0=0.1, beta1=10.0, seed=None, debug_delta_cost=False)
            exec_context.set_progress(0.8)
            model = SARIMAX(
                endog = target_col,
                order = (best_params['p'], best_params['d'], best_params['q']),
                seasonal_order = (best_params['P'], best_params['D'], best_params['Q'], self.seasonal_period_param)
            )
            trained_model = model.fit()
            exec_context.set_progress(0.9)

        # produce in-sample predictions for the whole series and put it in a Pandas Series
        in_samples = pd.Series(dtype=np.float64)
        preds_col = trained_model.predict(
            start=1, dynamic=self.dynamic_check
        )
        in_samples = pd.concat([in_samples, preds_col])
        in_samples = pd.DataFrame(in_samples, columns=["In-Samples"])

        # reverse log transformation for in-sample values
        if self.natural_log:
            in_samples = np.exp(in_samples)

        residuals = trained_model.resid[1:]
        residuals = pd.DataFrame(residuals, columns=["Residuals"])

        # populate model coefficients
        coeffs_and_stats = self.get_coeffs_and_stats(trained_model)

        model_binary = pickle.dumps(trained_model)

        exec_context.set_progress(0.9)
        return (
            knext.Table.from_pandas(in_samples),
            knext.Table.from_pandas(residuals),
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
                self.ar_order_param,
                # q
                self.ma_order_param,
                # S * P
                self.seasonal_period_param
                * self.seasonal_ar_order_param,
                # S*Q
                self.seasonal_period_param
                * self.seasonal_ma_order_param,
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
        
        
    def diff_parameters(self, series, seasonality, max_i, min_i, max_is, min_is, alpha=0.05):
        """
        Perform the KPSS test to check for stationarity in a time series.

        Parameters:
        - series: Pandas Series (Time Series Data)
        - alpha: Significance level (default=0.05)

        Returns:
        - Tuple (d paranmeter, D parameter)
        """
        p_value_ds = kpss(series, regression="c", nlags="auto")[1]

        if p_value_ds > alpha:
            ds_parameter = min(max(0, min_is), max_is) # Series is likely stationary at seasonal level (Fail to reject H₀)."

            # No seasonal difference applied. Now check for trend d
            p_value_d = kpss(series, regression="c", nlags="auto")[1]
            
            if p_value_d > alpha:
                print("Series is stationary at seasonal and trend levels (d=0, D=0).")
                return (min(max(0, min_i), max_i), ds_parameter)
            else:
                first_diff_series = series.diff().dropna()
                p_value_d = kpss(first_diff_series, regression="c", nlags="auto")[1]
            
                if p_value_d > alpha:
                    print("Series is stationary at seasonal level, but needs difference at trend level (d=1, D=0).")
                    return (min(max(1, min_i), max_i), ds_parameter)
                else:
                    print("Series is stationary at seasonal level, but needs 2 differences at trend level (d=2, D=0).")
                    return (min(max(2, min_i), max_i), ds_parameter)
        
        else:
            ds_parameter = min(max(1, min_is), max_is) # Series is likely non stationary at seasonal level (Reject H₀, differencing needed)

            # Apply seasonal difference
            first_seasonal_diff_series = series.diff(seasonality).dropna()

            # Now check for trend d after a seasonal difference has been applied
            p_value_d = kpss(first_seasonal_diff_series, regression="c", nlags="auto")[1]
            
            if p_value_d > alpha:
                print("Series is non stationary at seasonal level and stationary at trend level (d=0, D=1).")
                return (min(max(0, min_i), max_i), ds_parameter)
            else:
                first_diff_series = first_seasonal_diff_series.diff().dropna()
                p_value_d = kpss(first_diff_series, regression="c", nlags="auto")[1]
            
                if p_value_d > alpha:
                    print("Series is non stationary at seasonal level and needs difference at trend level (d=1, D=1).")
                    return (min(max(1, min_i), max_i), ds_parameter)
                else:
                    print("Series is non stationary at seasonal level and needs 2 differences at trend level (d=2, D=1).")
                    return (min(max(2, min_i), max_i), ds_parameter)
                

    def propose_initial_params(self, d, D, min_p=0, max_p=5, min_q=0, max_q=5, min_ps=0, max_ps=2, min_qs=0, max_qs=2):
        """
        Proposes initial random SARIMA hyperparameters.
        """
        p = round((min_p + max_p) / 2)
        q = round((min_q + max_q) / 2)
        P = round((min_ps + max_ps) / 2)
        Q = round((min_qs + max_qs) / 2)

        return {'p': p, 'd': d, 'q': q, 'P': P, 'D': D, 'Q': Q}

    def propose_new_params(self, current_params, min_p=0, max_p=5, min_q=0, max_q=5, min_ps=0, max_ps=2, min_qs=0, max_qs=2, step_size=1):
        """
        Proposes new random SARIMA hyperparameter(s) based on the current one with a small random step.
        """

        updated_params = deepcopy(current_params)
        threshold = np.random.rand()

        if threshold <= (1/2): # update trend parameters (p, q)

            current_p, current_q = updated_params['p'], updated_params['q']

            new_p, new_q = current_p + (np.random.choice([-1, 1]) * step_size), current_q + (np.random.choice([-1, 1]) * step_size)

            new_p = min(max_q, max(min_p, new_p))
            new_q = min(max_q, max(min_q, new_q))

            updated_params['p'], updated_params['q'] = new_p, new_q
        
        elif threshold > (1/2): # update seasonal parameters (P, Q)

            current_ps, current_qs = updated_params['P'], updated_params['Q']

            new_ps, new_qs = current_ps + (np.random.choice([-1, 1]) * step_size), current_qs + (np.random.choice([-1, 1]) * step_size)

            new_ps = min(max_qs, max(min_ps, new_ps))
            new_qs = min(max_qs, max(min_qs, new_qs))

            updated_params['P'], updated_params['Q'] = new_ps, new_qs

        return updated_params

    def evaluate_arima_model(self, series, params, seasonality):
        """
        Fits an ARIMA model and returns the AIC value as a score.
        """
        try:
            # AIC as scoring metric
            model_fit = SARIMAX(
                endog = series,
                order = (params['p'], params['d'], params['q']), # p, d, q order parameters for arima
                # trend = [params['a0'], params['a1']], # A(t) = a0 + a1*t + a2*t^2 ... 
                seasonal_order = (params['P'], params['D'], params['Q'], seasonality) # P, D, Q seasonal order parameters
            ).fit()

            return model_fit.aic
        except:
            return np.inf

    def accept(self, delta_c, beta):
        ## If the cost doesn't increase, we always accept
        if delta_c <= 0:
            return True
        ## If the cost increases and beta is infinite, we always reject
        if beta == np.inf:
            return False
        ## Otherwise the probability is going to be somwhere between 0 and 1
        p = np.exp(-beta * delta_c)
        ## Returns True with probability p
        return np.random.rand() < p

    ## The simulated annealing generic solver.
    ## Assumes that the proposals are symmetric.
    ## NOTE: The default beta0 and beta1 are arbitrary.
    def simann(self, series, seasonality, params_constraints, step_size=1,
            anneal_steps = 10, mcmc_steps = 100,
            beta0 = 0.1, beta1 = 10.0,
            seed = None, debug_delta_cost = False):
        ## Optionally set up the random number generator state
        if seed is not None:
            np.random.seed(seed)

        # Set up the list of betas.
        # First allocate an array with the required number of steps
        beta_list = np.zeros(anneal_steps)
        # All but the last one are evenly spaced between beta0 and beta1 (included)
        beta_list[:-1] = np.linspace(beta0, beta1, anneal_steps - 1)
        # The last one is set to infinty
        beta_list[-1] = np.inf

        # Set up the initial configuration, compute and print the initial cost
        # unpack the parameters constraints
        min_p, max_p, min_i, max_i, min_q, max_q, min_ps, max_ps, min_is, max_is, min_qs, max_qs = params_constraints

        d, D = self.diff_parameters(series, seasonality, max_i, min_i, max_is, min_is, alpha=0.05)
        current_params = self.propose_initial_params(d, D, min_p, max_p, min_q, max_q, min_ps, max_ps, min_qs, max_qs)

        current_c = self.evaluate_arima_model(series, current_params, seasonality)
        print(f"initial cost = {current_c}")

        ## Keep the best cost seen so far, and its associated configuration.
        best_params = deepcopy(current_params)
        best_c = current_c

        # Main loop of the annaling: Loop over the betas
        for beta in beta_list:
            ## At each beta, we want to record the acceptance rate, so we need a
            ## counter for the number of accepted moves
            accepted = 0
            # For each beta, perform a number of MCMC steps
            for t in range(mcmc_steps):
                move = self.propose_new_params(current_params, step_size)
                print(f"new params = {move}")
                delta_c = self.evaluate_arima_model(series, move, seasonality) - current_c
                print(f"delta_c = {delta_c}")
                ## Optinal (expensive) check that `compute_delta_cost` works
                # if debug_delta_cost:
                #     probl_copy = probl.copy()
                #     probl_copy.accept_move(move)
                #     assert abs(c + delta_c - probl_copy.cost()) < 1e-10
                ## Metropolis rule
                if self.accept(delta_c, beta):
                    current_params = deepcopy(move)
                    current_c += delta_c
                    accepted += 1
                    if current_c <= best_c:
                        best_c = current_c
                        best_params = deepcopy(current_params)
            # print(f"acc.rate={accepted/mcmc_steps} beta={beta} current cost={current_c} [best cost={best_c}]")

        ## Return the best instance
        # print(f"final cost = {best_c}, final params = {best_params}")
        return best_params