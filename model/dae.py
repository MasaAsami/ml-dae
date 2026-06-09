import numpy as np
from scipy.optimize import fmin_slsqp
from multiprocessing import Pool
from tqdm import tqdm

def optimize_one_unit(args):
    inputs, outputs, unit, m, r, n = args

    def target(x):
        in_w, out_w, lambdas = x[:m], x[m:m + r], x[m + r:]
        denominator = np.dot(inputs[unit], in_w)
        numerator = np.dot(outputs[unit], out_w)
        return numerator / denominator

    def constraints(x):
        in_w, out_w, lambdas = x[:m], x[m:m + r], x[m + r:]
        t = target(x)
        constr = []

        for i in range(m):
            lhs = np.dot(inputs[:, i], lambdas)
            constr.append(t * inputs[unit, i] - lhs)

        for o in range(r):
            lhs = np.dot(outputs[:, o], lambdas)
            constr.append(lhs - outputs[unit, o])

        for u in range(n):
            constr.append(lambdas[u])

        return np.array(constr)

    d0 = m + r + n
    x0 = np.random.rand(d0) - 0.5
    result = fmin_slsqp(target, x0, f_ieqcons=constraints, disp=False)
    in_w, out_w = result[:m], result[m:m + r]
    denominator = np.dot(inputs, in_w)
    numerator = np.dot(outputs, out_w)
    return (numerator / denominator)[unit]


class DEA(object):

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.n = inputs.shape[0]
        self.m = inputs.shape[1]
        self.r = outputs.shape[1]

        self.unit_ = range(self.n)
        self.input_ = range(self.m)
        self.output_ = range(self.r)

        self.output_w = np.zeros((self.r, 1), dtype=float)
        self.input_w = np.zeros((self.m, 1), dtype=float)
        self.lambdas = np.zeros((self.n, 1), dtype=float)
        self.efficiency = np.zeros_like(self.lambdas)

        self.names = []

    def __optimize(self):
        args = [(self.inputs, self.outputs, unit, self.m, self.r, self.n) for unit in self.unit_]
        with Pool() as pool:
            results = list(tqdm(pool.imap(optimize_one_unit, args), total=self.n, desc="Optimizing DEA"))
        self.efficiency = np.array(results).reshape(-1, 1)

    def name_units(self, names):
        assert self.n == len(names)
        self.names = names

    def fit(self):
        self.__optimize()
