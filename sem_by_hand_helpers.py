import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from itertools import combinations, chain
from scipy.stats import norm, mvn
import scipy.linalg.lapack as lapack


from collections import defaultdict, namedtuple
import re

Operation = namedtuple('Operation', 'name, params, onto',
                       defaults=(None, None))
__prt_lvalue = r'(\w[\w\.]*(?:\s*,\s*[\w.]*)*)'
__prt_op = r'\s*((?:\s\w+\s)|(?:[=~\\\*@\$<>\-]+\S*?))\s*'
__prt_rvalue = r'(-?\w[\w.-]*(?:\s*\*\s*\w[\w.]*)?(?:\s*\+\s*-?\w[\w.-]*(?:\s*\*\s*\w[\w.]*)?)*)'
PTRN_EFFECT = re.compile(__prt_lvalue + __prt_op + __prt_rvalue)
PTRN_OPERATION = re.compile(r'([A-Z][A-Z_]+(?:\(.*\))?)\s*([\w\s]+)*')
PTRN_OPERATION_FULL = re.compile(r'([a-z][a-z_]*)\s*(.*?)\s*:\s*(.*)\s*')
PTRN_OPERATION_PARAM = re.compile(r'([a-z][a-z_]*)\s*[\"\'\`]\s*(.+)\s*[\"\'\`]')
PTRN_RVALUE = re.compile(r'((-?\w[\w.-]*\*)?\w[\w.]*)')
PTRN_OP = re.compile(r'(\w+)(\(.*\))?')


def mvn_negloglik(dat, Sigma):
    """
    Multivariate normal negative log-likelihood loss function for tensorsem nn module.
    :param dat: The centered dataset as a tensor
    :param Sigma: The model() implied covariance matrix
    :return: Tensor scalar negative log likelihood
    """
    mu = torch.zeros(Sigma.shape[0], dtype=Sigma.dtype)
    mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=Sigma)
    # except ValueError:
    #     Sigma = Sigma + torch.diag(0.001 + torch.ones(Sigma.shape[0]))
    #     mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=Sigma)

    return - mvn.log_prob(dat).sum()


def separate_token(token: str):
    """
    Test if token satisfies basic command semopy syntax and separates token.

    Parameters
    ----------
    token : str
        A text line with either effect command or operation command.

    Raises
    ------
    SyntaxError
        Token happens to be incorrect, i.e. it does not follows basic
        semopy command pattern.

    Returns
    -------
    int
        0 if effect, 1-2 if operation (depending on the format, there are 2
        formats: the frist one is new small-case, the other is the old
        capital-case).
    tuple
        A tuple of (lvalue, operation, rvalue) if command is effect or
        (operation, operands) if command is operation.

    """
    effect = PTRN_EFFECT.fullmatch(token)
    if effect:
        return 0, effect.groups()
    operation = PTRN_OPERATION_FULL.fullmatch(token)
    if operation:
        return 1, operation.groups()
    operation = PTRN_OPERATION_PARAM.fullmatch(token)
    if operation:
        return 2, operation.groups()
    operation = PTRN_OPERATION.fullmatch(token)
    if not operation:
        raise SyntaxError(f'Invalind syntax for line:\n{token}')
    return 3, operation.groups()


def parse_rvalues(token: str):
    """
    Separate token by  '+' sign and parses expression "val*x" into tuples.

    Parameters
    ----------
    token : str
        Right values from operand.

    Raises
    ------
    Exception
        Rises when a certain rvalue can't be processed.

    Returns
    -------
    rvalues : dict
        A mapping Variable->Multiplicator.

    """
    token = token.replace(' ', '')
    rvalues = dict()
    for tok in token.split('+'):
        rval = PTRN_RVALUE.match(tok)
        if not rval:
            raise Exception(f'{rval} does not seem like a correct semopy \
                            expression')
        groups = rval.groups()
        name = groups[0].split('*')[-1]
        rvalues[name] = groups[1][:-1] if groups[1] else None
    return rvalues


def parse_operation(operation: str, operands: str):
    """
    Parse an operation according to semopy syntax.

    Parameters
    ----------
    operation : str
        Operation string with possible arguments.
    operands : str
        Variables/values that operation acts upon.

    Raises
    ------
    SyntaxError
        Rises when there is an error during parsing.

    Returns
    -------
    operation : Operation
        Named tuple containing information on operation.

    """
    oper = PTRN_OP.match(operation)
    if not oper:
        raise SyntaxError(f'Incorrect operation pattern: {operation}')
    operands = [op.strip() for op in operands.split()] if operands else list()
    groups = oper.groups()
    name = groups[0]
    params = groups[1]
    if params is not None:
        params = [t.strip() for t in params[1:-1].split(',')]
    operation = Operation(name, params, operands)
    return operation


def parse_new_operation(groups: tuple):
    """
    Parse an operation according to semopy syntax.

    Version for a new operation syntax.
    Parameters
    ----------
    operation : tuple
        Groups as returned by regex parser.

    Returns
    -------
    operation : Operation
        Named tuple containing information on operation.

    """
    name = groups[0]
    params = groups[1]
    try:
        try:
            operands = groups[2].split()
        except IndexError:
            operands = None
    except AttributeError:
        operands = None
        if not params:
            raise SyntaxError("Unknown syntax error.")
    operation = Operation(name, params, operands)
    return operation


def parse_desc(desc: str):
    """
    Parse a model description provided in semopy's format.

    Parameters
    ----------
    desc : str
        Model description in semopy format.

    Returns
    -------
    effects : defaultdict
        Mapping operation->lvalue->rvalue->multiplicator.
    operations : dict
        Mapping operationName->list[Operation type].

    """
    desc = desc.replace(chr(8764), chr(126))
    effects = defaultdict(lambda: defaultdict(dict))
    operations = defaultdict(list)
    for line in desc.splitlines():
        try:
            i = line.index('#')
            line = line[:i]
        except ValueError:
            pass
        line = line.strip()
        if line:
            try:
                kind, items = separate_token(line)
                if kind == 0:
                    lefts, op_symb, rights = items
                    for left in lefts.split(','):
                        rvalues = parse_rvalues(rights)
                        effects[op_symb][left.strip()].update(rvalues)
                elif kind < 3:
                    t = parse_new_operation(items)
                    operations[t.name].append(t)
                else:
                    operation, operands = items
                    t = parse_operation(operation, operands)
                    operations[t.name].append(t)
            except SyntaxError:
                raise SyntaxError(f"Syntax error for line:\n{line}")
    return effects, operations


def cov(x: np.ndarray):
    """
    Compute covariance matrix takin in account missing values.

    Parameters
    ----------
    x : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Covariance matrix.

    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    cov = np.ma.cov(masked_x, bias=True, rowvar=False).data
    if cov.size == 1:
        cov.resize((1,1))
    return cov


def cor(x: np.ndarray):
    """
    Compute correlation matrix takin in account missing values.

    Parameters
    ----------
    x : np.ndarray
        Data.

    Returns
    -------
    np.ndarray
        Correlation matrix.

    """
    masked_x = np.ma.array(x, mask=np.isnan(x))
    cor = np.ma.corrcoef(masked_x, bias=True, rowvar=False).data
    if cor.size == 1:
        cor.resize((1,1))
    return cor


def chol_inv(x: np.array):
    """
    Calculate invserse of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.array
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    lapack.dpotri(c, overwrite_c=1)
    return c + c.T - np.diag(c.diagonal())


def chol_inv2(x: np.ndarray):
    """
    Calculate invserse and logdet of matrix using Cholesky decomposition.

    Parameters
    ----------
    x : np.ndarray
        Data with columns as variables and rows as observations.

    Raises
    ------
    np.linalg.LinAlgError
        Rises when matrix is either ill-posed or not PD.

    Returns
    -------
    c : np.ndarray
        x^(-1).
    logdet : float
        ln|x|

    """
    c, info = lapack.dpotrf(x)
    if info:
        raise np.linalg.LinAlgError
    d = c.diagonal()
    logdet = 2 * np.sum(np.log(d))
    lapack.dpotri(c, overwrite_c=1)
    return c + c.T - np.diag(d), logdet


def bivariate_cdf(lower, upper, corr, means=[0, 0], var=[1, 1]):
    """
    Estimates an integral of bivariate pdf.

    Estimates an integral of bivariate pdf given integration lower and
    upper limits. Consider using relatively big (i.e. 20 if using default mean
    and variance) lower and/or upper bounds when integrating to/from infinity.
    Parameters
    ----------
    lower : float
        Lower integration bounds.
    upper : float
        Upper integration bounds.
    corr : float
        Correlation coefficient between variables.
    means : list, optional
        Mean values of variables. The default is [0, 0].
    var : list, optional
        Variances of variables. The default is [1, 1].

    Returns
    -------
    float
        P(lower[0] < x < upper[0], lower[1] < y < upper[1]).

    """
    s = np.array([[var[0], corr], [corr, var[1]]])
    return mvn.mvnun(lower, upper, means, s)[0]


def estimate_intervals(x, inf=10):
    """
    Estimate intervals of the polytomized underlying latent variable.

    Parameters
    ----------
    x : np.ndarray
        An array of values the ordinal variable..
    inf : float, optional
        A numerical infinity substitute. The default is 10.

    Returns
    -------
    np.ndarray
        An array containing polytomy intervals.
    np.ndarray
        An array containing indices of intervals corresponding to each entry.

    """
    x_f = x[~np.isnan(x)]
    u, counts = np.unique(x_f, return_counts=True)
    sz = len(x_f)
    cumcounts = np.cumsum(counts[:-1])
    u = [np.where(u == sample)[0][0] + 1 for sample in x]
    return list(chain([-inf], (norm.ppf(n / sz) for n in cumcounts), [inf])), u


def polyserial_corr(x, y, x_mean=None, x_var=None, x_z=None, x_pdfs=None,
                    y_ints=None, scalar=True):
    """
    Estimate polyserial correlation.

    Estimate polyserial correlation between continious variable x and
    ordinal variable y.
    Parameters
    ----------
    x : np.ndarray
        Data sample corresponding to x.
    y : np.ndarray
        Data sample corresponding to ordinal variable y.
    x_mean : float, optional
        Mean of x (calculate if not provided). The default is None.
    x_var : float, optional
        Variance of x (calculate if not provided). The default is None.
    x_z : np.ndarray, optional
        Stabdardized x (calculated if not provided). The default is None.
    x_pdfs : np.ndarray, optional
        x's logpdf sampled at each point (calculated if not provided). The
        default is None.
    y_ints : list, optional
        Polytomic intervals of an underlying latent variable
        correspoding to y (calculated if not provided) as returned by
        estimate_intervals.. The default is None.
    scalar : bool, optional
        If true minimize_scalar is used instead of SLSQP.. The default is True.

    Returns
    -------
    float
        A polyserial correlation coefficient for x and y..

    """
    if x_mean is None:
        x_mean = np.nanmean(x)
    if x_var is None:
        x_var = np.nanvar(x)
    if y_ints is None:
        y_ints = estimate_intervals(y)
    if x_z is None:
        x_z = (x - x_mean) / x_var
    if x_pdfs is None:
        x_pdfs = norm.logpdf(x, x_mean, x_var)
    ints, inds = y_ints

    def transform_tau(tau, rho, z):
        return (tau - rho * z) / np.sqrt(1 - rho ** 2)

    def sub_pr(k, rho, z):
        i = transform_tau(ints[k], rho, z)
        j = transform_tau(ints[k - 1], rho, z)
        return univariate_cdf(j, i)

    def calc_likelihood(rho):
        return -sum(pdf + np.log(sub_pr(ind, rho, z))
                    for z, ind, pdf in zip(x_z, inds, x_pdfs))

    def calc_likelihood_derivative(rho):
        def sub(k, z):
            i = transform_tau(ints[k], rho, z)
            j = transform_tau(ints[k - 1], rho, z)
            a = norm.pdf(i) * (ints[k] * rho - z)
            b = norm.pdf(j) * (ints[k - 1] * rho - z)
            return a - b

        t = (1 - rho ** 2) ** 1.5
        return -sum(sub(ind, z) / sub_pr(ind, rho, z)
                    for x, z, ind in zip(x, x_z, inds) if not np.isnan(x)) / t

    if not scalar:
        res = minimize(calc_likelihood, [0.0], jac=calc_likelihood_derivative,
                       method='SLSQP', bounds=[(-1.0, 1.0)]).x[0]
    else:
        res = minimize_scalar(calc_likelihood, bounds=(-1, 1),
                              method='bounded').x
    return res


def polychoric_corr(x, y, x_ints=None, y_ints=None):
    """
    Estimate polyserial correlation between ordinal variables x and y.

    Parameters
    ----------
    x : np.ndarray
        Data sample corresponding to x.
    y : np.ndarray
        Data sample corresponding to y.
    x_ints : list, optional
        Polytomic intervals of an underlying latent variable correspoding to y
        (calculated if not provided) as returned by estimate_intervals. The
        default is None.
    y_ints : list, optional
        Polytomic intervals of an underlying latent variable correspoding to y
        (calculated if not provided) as returned by estimate_intervals. The
        default is None.

    Returns
    -------
    float
        A polychoric correlation coefficient for x and y.

    """
    if x_ints is None:
        x_ints = estimate_intervals(x)
    if y_ints is None:
        y_ints = estimate_intervals(y)
    x_ints, x_inds = x_ints
    y_ints, y_inds = y_ints
    p, m = len(x_ints) - 1, len(y_ints) - 1
    n = np.zeros((p, m))
    for a, b in zip(x_inds, y_inds):
        if not (np.isnan(a) or np.isnan(b)):
            n[a - 1, b - 1] += 1

    def calc_likelihood(r):
        return -sum(np.log(bivariate_cdf([x_ints[i], y_ints[j]],
                                         [x_ints[i + 1], y_ints[j + 1]], r)) * n[i, j]
                    for i in range(p) for j in range(m))

    return minimize_scalar(calc_likelihood, bounds=(-1, 1), method='bounded').x

def hetcor(data, ords=None, nearest=False):
    """
    Compute a heterogenous correlation matrix.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        DESCRIPTION.
    ords : list, optional
        Names of ordinal variables if data is DataFrame or indices of
        ordinal numbers if data is np.array. If ords are None then ordinal
        variables will be determined automatically. The default is None.
    nearest : bool, optional
        If True, then nearest PD correlation matrix is returned instead. The
        default is False.

    Returns
    -------
    cor : pd.DataFrame
        A heterogenous correlation matrix.

    """
    if type(data) is np.ndarray:
        cov = cor(data)
        if ords is None:
            ords = set()
            for i in range(data.shape[1]):
                if len(np.unique(data[:, i])) / data.shape[0] < 0.3:
                    ords.add(i)
        conts = set(range(data.shape[1])) - set(ords)
    else:
        cov = data.corr()
        if ords is None:
            ords = set()
            for var in data:
                if len(data[var].unique()) / len(data[var]) < 0.3:
                    ords.add(var)
        conts = set(data.columns) - set(ords)
    data = data.T
    c_means = {v: np.nanmean(data[v]) for v in conts}
    c_vars = {v: np.nanvar(data[v]) for v in conts}
    c_z = {v: (data[v] - c_means[v]) / c_vars[v] for v in conts}
    c_pdfs = {v: norm.logpdf(data[v], c_means[v], c_vars[v]) for v in conts}
    o_ints = {v: estimate_intervals(data[v]) for v in ords}

    for c, o in product(conts, ords):
        cov[c][o] = polyserial_corr(data[c], data[o], x_mean=c_means[c],
                                    x_var=c_vars[c], x_z=c_z[c],
                                    x_pdfs=c_pdfs[c], y_ints=o_ints[o])
        cov[o][c] = cov[c][o]
    for a, b in combinations(ords, 2):
        cov[a][b] = polychoric_corr(data[a], data[b], o_ints[a], o_ints[b])
        cov[b][a] = cov[a][b]
    if nearest:
        if type(cov) is pd.DataFrame:
            names = cov.columns
            cov = corr_nearest(cov,threshold=0.05)
            cov = pd.DataFrame(cov, columns=names, index=names)
        else:
            cov = corr_nearest(cov, threshold=0.05)
    return cov


def cov_nearest(cov, method='clipped', threshold=1e-15, n_fact=100,
                return_all=False):
    """
    Find the nearest covariance matrix that is positive (semi-) definite

    This leaves the diagonal, i.e. the variance, unchanged

    Parameters
    ----------
    cov : ndarray, (k,k)
        initial covariance matrix
    method : str
        if "clipped", then the faster but less accurate ``corr_clipped`` is
        used.if "nearest", then ``corr_nearest`` is used
    threshold : float
        clipping threshold for smallest eigen value, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations in
        ``corr_nearest``. See its doc string
    return_all : bool
        if False (default), then only the covariance matrix is returned.
        If True, then correlation matrix and standard deviation are
        additionally returned.

    Returns
    -------
    cov_ : ndarray
        corrected covariance matrix
    corr_ : ndarray, (optional)
        corrected correlation matrix
    std_ : ndarray, (optional)
        standard deviation


    Notes
    -----
    This converts the covariance matrix to a correlation matrix. Then, finds
    the nearest correlation matrix that is positive semidefinite and converts
    it back to a covariance matrix using the initial standard deviation.

    The smallest eigenvalue of the intermediate correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input covariance matrix is symmetric.

    See Also
    --------
    corr_nearest
    corr_clipped
    """

    from statsmodels.stats.moment_helpers import cov2corr, corr2cov
    cov_, std_ = cov2corr(cov, return_std=True)
    if method == 'clipped':
        corr_ = corr_clipped(cov_, threshold=threshold)
    else:  # method == 'nearest'
        corr_ = corr_nearest(cov_, threshold=threshold, n_fact=n_fact)

    cov_ = corr2cov(corr_, std_)

    if return_all:
        return cov_, corr_, std_
    else:
        return cov_


def corr_clipped(corr, threshold=1e-15):
    '''
    Find a near correlation matrix that is positive semi-definite

    This function clips the eigenvalues, replacing eigenvalues smaller than
    the threshold by the threshold. The new matrix is normalized, so that the
    diagonal elements are one.
    Compared to corr_nearest, the distance between the original correlation
    matrix and the positive definite correlation matrix is larger, however,
    it is much faster since it only computes eigenvalues once.

    Parameters
    ----------
    corr : ndarray, (k, k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigenvalue, see Notes

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix


    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold``. In examples, the
    smallest eigenvalue can be by a factor of 10 smaller than the threshold,
    e.g. threshold 1e-8 can result in smallest eigenvalue in the range
    between 1e-9 and 1e-8.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric. The diagonal elements of
    returned correlation matrix is set to ones.

    If the correlation matrix is already positive semi-definite given the
    threshold, then the original correlation matrix is returned.

    ``cov_clipped`` is 40 or more times faster than ``cov_nearest`` in simple
    example, but has a slightly larger approximation error.

    See Also
    --------
    corr_nearest
    cov_nearest

    '''
    x_new, clipped = clip_evals(corr, value=threshold)
    if not clipped:
        return corr

    # cov2corr
    x_std = np.sqrt(np.diag(x_new))
    x_new = x_new / x_std / x_std[:, None]
    return x_new


def corr_nearest(corr, threshold=1e-15, n_fact=100):
    '''
    Find the nearest correlation matrix that is positive semi-definite.

    The function iteratively adjust the correlation matrix by clipping the
    eigenvalues of a difference matrix. The diagonal elements are set to one.

    Parameters
    ----------
    corr : ndarray, (k, k)
        initial correlation matrix
    threshold : float
        clipping threshold for smallest eigenvalue, see Notes
    n_fact : int or float
        factor to determine the maximum number of iterations. The maximum
        number of iterations is the integer part of the number of columns in
        the correlation matrix times n_fact.

    Returns
    -------
    corr_new : ndarray, (optional)
        corrected correlation matrix

    Notes
    -----
    The smallest eigenvalue of the corrected correlation matrix is
    approximately equal to the ``threshold``.
    If the threshold=0, then the smallest eigenvalue of the correlation matrix
    might be negative, but zero within a numerical error, for example in the
    range of -1e-16.

    Assumes input correlation matrix is symmetric.

    Stops after the first step if correlation matrix is already positive
    semi-definite or positive definite, so that smallest eigenvalue is above
    threshold. In this case, the returned array is not the original, but
    is equal to it within numerical precision.

    See Also
    --------
    corr_clipped
    cov_nearest

    '''
    k_vars = corr.shape[0]
    if k_vars != corr.shape[1]:
        raise ValueError("matrix is not square")

    diff = np.zeros(corr.shape)
    x_new = corr.copy()
    diag_idx = np.arange(k_vars)

    for ii in range(int(len(corr) * n_fact)):
        x_adj = x_new - diff
        x_psd, clipped = clip_evals(x_adj, value=threshold)
        if not clipped:
            x_new = x_psd
            break
        diff = x_psd - x_adj
        x_new = x_psd.copy()
        x_new[diag_idx, diag_idx] = 1
    else:
        print("Warning!")

    return x_new

def clip_evals(x, value=0):  # threshold=0, value=0):
    evals, evecs = np.linalg.eigh(x)
    clipped = np.any(evals < value)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return x_new, clipped

