StatModel class at model.py implements Weighted Piecewise Linear Regression Model for arbitrary non-empty regressor and dependent
variable by specified grid of regression function inflexion nodes and (additionally) weights of regressors.


Example:

logger = logging.getLogger()

stat_model = StatModel(logger)

stat_model.fit(X, Y, nodes=nodes, weights=weights)
