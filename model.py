import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

from scipy import optimize
from copy import deepcopy


class StatModel(object):

    def __init__(self, logger):

        self.logger = logger

    def __validate_args(self, x_arg, y_arg, nodes_arg, w_arg):

        if (x_arg is None) or len(x_arg) == 0:
            self.logger.error('Empty Input Arrays')
        elif len(x_arg) != len(y_arg):
            self.logger.error('Input Arrays Inconsistent Size')

        if nodes_arg is None:
            nodes = np.array([np.min(x_arg), np.max(x_arg)])
        elif type(nodes_arg) is not np.ndarray:
            nodes = np.array(nodes_arg)
        else:
            nodes = deepcopy(nodes_arg)

        ndim = len(x_arg)

        if type(x_arg) is np.ndarray:
            X = x_arg.reshape((ndim, 1))
        else:
            X = np.array(x_arg).reshape((ndim, 1))

        if type(y_arg) is np.ndarray:
            Y = y_arg.reshape((ndim, 1))
        else:
            Y = np.array(y_arg).reshape((ndim, 1))

        if w_arg is None:
            W = np.identity(ndim)
        elif type(w_arg) is not np.ndarray and len(w_arg) > 0:
            W = np.diag(w_arg)
        else:
            W = w_arg.reshape((ndim, ndim))

        return X, Y, nodes, W

    @staticmethod
    def __collect_args(X, Y, nodes):

        ndim = X.size
        mdim = nodes.size - 1

        func_matrix = np.zeros((ndim, 2 * mdim))

        eq_matrix = np.zeros((mdim - 1, 2 * mdim))

        y_gen_arr = np.zeros((ndim, 1))

        j = []

        for i in xrange(mdim):

            if i == 0:
                j.append(np.where((X <= nodes[i + 1]))[0])
            else:
                j.append(np.where((X > nodes[i]) & (X <= nodes[i + 1]))[0])

            if j[i].size == 0:
                continue

        vpos_st = 0

        for _dim in xrange(mdim):

            grid_dim = j[_dim].size

            if grid_dim == 0:
                continue

            vpos_en = vpos_st + grid_dim

            hpos = 2 * _dim

            y_gen_arr[vpos_st:vpos_en] = Y[j[_dim]]

            func_matrix[vpos_st:vpos_en, hpos] = X[j[_dim]].ravel()
            func_matrix[vpos_st:vpos_en, hpos + 1] = 1

            vpos_st = vpos_en

        for _dim in xrange(mdim - 1):

            zi = nodes[_dim + 1]

            hpos_st = 2 * _dim
            hpos_en = hpos_st + 4

            eq_vector = np.array([zi, 1, -zi, -1])

            eq_matrix[_dim, hpos_st:hpos_en] = eq_vector

        return func_matrix, eq_matrix, y_gen_arr

    @staticmethod
    def optimizer(H, c, c0, **kwargs):

        '''
        Solve quadratic optimization problem
        0.5 <x, Hx> - <x, c> + 0.5 * c0 --> min

        In condition:
        Ax = b
        If b is not mentioned:
        Ax = 0

        In case A and b are not mentioned -- solve general
        LS minimization problem
        '''

        loss = lambda x: 0.5 * np.dot(x.T, np.dot(H, x)) - np.dot(c.T, x) + 0.5 * c0

        jac = lambda x: np.dot(x.T, H) - c.T

        A = kwargs.get('A')
        b = kwargs.get('b')

        if (type(A) is np.ndarray) and (type(b) is np.ndarray) and (A.size > 0):

            eq_cons = {'type': 'eq',
                       'fun': lambda x: b - np.dot(A, x),
                       'jac': lambda x: -A}

        elif (type(A) is np.ndarray) and (A.size > 0):

            eq_cons = {'type': 'eq',
                       'fun': lambda x: np.dot(A, x),
                       'jac': lambda x: A}
        else:
            eq_cons = None

        init_val = np.random.randn(H.shape[0], 1)

        if eq_cons:
            opt_cons = optimize.minimize(loss, init_val, jac=jac, constraints=eq_cons,
                                         method='SLSQP', options={'disp': False})
        else:
            opt_cons = optimize.minimize(loss, init_val, jac=jac, method='SLSQP',
                                         options={'disp': False})

        return opt_cons.x, opt_cons.status

    @staticmethod
    def target_regress_values(nodes, linear_args):

        node_dim = nodes.size

        Y_target = np.zeros(node_dim)

        for _ord in xrange(node_dim - 1):

            slope = linear_args[2 * _ord]

            intercept = linear_args[2 * _ord + 1]

            Y_target[_ord] = slope * nodes[_ord] + intercept

            Y_target[_ord + 1] = slope * nodes[_ord + 1] + intercept

        return Y_target

    def fit(self, x_arg, y_arg, **kwargs):

        '''
        Fit Piece-Wise Linear Regression Trend
        by the current Data Sample at the Fixed Nodes

        :param x_arg: n-dimension numpy.ndarray or list
        :param y_arg: n-dimension numpy.ndarray or list
        :param kwargs:

        nodes: m-dimension numpy.ndarray or list
        of the regression function inflexion points

        weights: n-dimension numpy.ndarray or list
        of the sample points weights

        :return: m-dimension numpy.ndarray of the regression
        function values at the data nodes
        '''

        nodes_arg = kwargs.get('nodes')
        w_arg = kwargs.get('weights')

        X, Y, nodes, W = self.__validate_args(x_arg, y_arg, nodes_arg, w_arg)

        func_matrix, eq_matrix, y_gen_arr = self.__collect_args(X, Y, nodes)

        H = np.dot(func_matrix.T, np.dot(W, func_matrix))

        c = np.dot(func_matrix.T, np.dot(W, y_gen_arr))

        c0 = np.dot(y_gen_arr.T, np.dot(W, y_gen_arr))

        opt_linear_arg, opt_code = self.optimizer(H, c, c0, A=eq_matrix)

        if opt_code:
            self.logger.warning('Optimization Terminated with Status -- {0}'.format(opt_code))

        target_values = self.target_regress_values(nodes, opt_linear_arg)

        return target_values


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    lmodel = StatModel(logger)

    segm_func = lambda x: 2 * x - 1 if x <= 2 else 2 - x

    arr_func = np.vectorize(segm_func)

    X = np.linspace(-1, 5, 100)

    Y = arr_func(X) + np.random.normal(size=(X.size))

    nodes = np.array([-1, -0.5, 0, 1, 2, 4, 5])

    opt_regress = lmodel.fit(X, Y, nodes=nodes)

    plt.figure(figsize=(8, 10))

    plt.title('Piece-Wise Linear Regression Trend')

    plt.plot(X, Y, 'bo', nodes, opt_regress, '--r')

    plt.xlabel('X data')

    plt.ylabel('Y data')

    plt.savefig('PWLR_test.png')
