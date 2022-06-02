"""
Adapted from Semopy and Torchsem
"""


import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass
from itertools import combinations, chain
from scipy.stats import norm, mvn
import scipy.linalg.lapack as lapack

from tqdm import tqdm

from sem_by_hand_helpers import *
from sem_by_hands_inspect import inspect_list


@dataclass
class ParameterLoc:
    """Structure for keeping track of parameter's location in matrices."""

    __slots__ = ['matrix', 'indices', 'symmetric']

    matrix: np.ndarray
    indices: tuple
    symmetric: bool


@dataclass
class Parameter:
    """Structure for basic parameter info used internally in Model."""

    __slots__ = ['start', 'active', 'bound', 'locations']

    start: float
    active: bool
    bound: tuple
    locations: list


class Model(torch.nn.Module):

    matrices_names = 'beta', 'lambda', 'psi', 'theta'

    symb_regression = '~'
    symb_covariance = '~~'
    symb_measurement = '=~'

    symb_starting_values = 'START'

    dict_effects = dict()
    dict_operations = dict()

    def __init__(self, desc, mimic_lavaan=False, cov_diag=False):
        super().__init__()

        self.mimic_lavaan = mimic_lavaan
        self.cov_diag = cov_diag

        self._par = dict()
        self.n_param_reg = 0
        self.n_param_cov = 0

        effects, operations = parse_desc(desc)

        self.classify_variables(effects, operations)
        self.finalize_variable_classification(effects)
        self.preprocess_effects(effects)
        self.setup_matrices()
        self.create_parameters(effects)
        self.apply_operations(operations)

    def classify_variables(self, effects: dict, operations: dict):
        """
        Classify and instantiate vars dict.

        Parameters
        ----------
        effects : dict
            Dict returned from parse_desc.

        operations: dict
            Dict of operations as returned from parse_desc.

        Returns
        -------
        None.

        """
        self.vars = dict()
        latents = set()
        in_arrows, out_arrows = set(), set()
        senders = defaultdict(set)
        indicators = set()
        for rv, inds in effects[self.symb_measurement].items():
            latents.add(rv)
            indicators.update(inds)
            out_arrows.add(rv)
            senders[rv].update(inds)
        in_arrows.update(indicators)
        for rv, lvs in effects[self.symb_regression].items():
            in_arrows.add(rv)
            out_arrows.update(lvs)
            for lv in lvs:
                senders[lv].add(rv)

        allvars = out_arrows | in_arrows
        exogenous = out_arrows - in_arrows
        outputs = in_arrows - out_arrows
        endogenous = allvars - exogenous
        observed = allvars - latents
        self.vars['all'] = allvars
        self.vars['endogenous'] = endogenous
        self.vars['exogenous'] = exogenous
        self.vars['observed'] = observed
        self.vars['latent'] = latents
        self.vars['indicator'] = indicators
        self.vars['output'] = outputs
        self.vars_senders = senders

    def finalize_variable_classification(self, effects: dict):
        """
        Finalize variable classification.

        Reorders variables for better visual fancyness and does extra
        model-specific variable respecification.

        Parameters
        -------
        effects : dict
            Maping opcode->values->rvalues->mutiplicator.

        Returns
        -------
        None.

        """
        if not self.mimic_lavaan:
            outputs = self.vars['output']
            self.vars['_output'] = outputs
            self.vars['output'] = {}
        else:
            inds = self.vars['indicator']
            # It's possible that user violates that classical requirement that
            # measurement variables must be outputs, here we take such case
            # in account by ANDing inds with outputs.
            self.vars['_output'] = inds & self.vars['output']
            self.vars['output'] -= inds
        inners = self.vars['all'] - self.vars['_output']
        self.vars['inner'] = inners
        # This is entirely for visual reasons only, one might replace sorted
        # with list as well.
        obs = self.vars['observed']
        inners = sorted(self.vars['latent'] & inners) + sorted(obs & inners)
        self.vars['inner'] = inners
        t = obs & self.vars['_output']
        self.vars['observed'] = sorted(t) + sorted(obs - t)

    def setup_matrices(self):
        """
        Initialize base matrix structures of the model.

        Returns
        -------
        None.

        """
        # I don't use dicts here as matrices structures and further ranges
        # structures will be used very often in optimsation procedure, and
        # key-retrieival process takes a toll on performance.
        # Please, don't rearrange matrices below, let the order stay the same.
        # It's necessary because I assumed fixed matrix positions in those
        # structures when coding linear algebra parts of the code.
        self.matrices = list()
        self.names = list()
        self.start_rules = list()
        for v in self.matrices_names:
            mx, names = getattr(self, f'build_{v}')()
            setattr(self, f'mx_{v}', mx)
            setattr(self, f'names_{v}', names)
            self.matrices.append(mx)
            self.names.append(names)
            self.start_rules.append(getattr(self, f'start_{v}'))

    def build_beta(self):
        """
        Beta matrix contains relationships between all non-_output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['inner']
        n = len(names)
        mx = torch.zeros((n, n), dtype=torch.float64)
        return mx, (names, names)

    def build_lambda(self):
        """
        Lambda matrix loads non-_output variables onto _output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        obs = self.vars['observed']
        inner = self.vars['inner']
        row, col = obs, inner
        n, m = len(row), len(col)
        mx = torch.zeros((n, m), dtype=torch.float64)
        for v in obs:
            if v in inner:
                i, j = obs.index(v), inner.index(v)
                mx[i, j] = 1.0
        return mx, (row, col)

    def build_psi(self):
        """
        Psi matrix is a covariance matrix for non-_output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['inner']
        n = len(names)
        mx = torch.zeros((n, n), dtype=torch.float64)
        return mx, (names, names)

    def build_theta(self):
        """
        Theta matrix is a covariance matrix for _output variables.

        Returns
        -------
        np.ndarray
            Matrix.
        tuple
            Tuple of rownames and colnames.

        """
        names = self.vars['observed']
        n = len(names)
        mx = torch.zeros((n, n), dtype=torch.float64)
        return mx, (names, names)

    def preprocess_effects(self, effects: dict):
        """
        Run a routine just before effects are applied.

        Used to apply covariances to model.
        Parameters
        -------
        effects : dict
            Mapping opcode->lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        cov = effects[self.symb_covariance]
        exo = self.vars['exogenous']
        obs_exo = set(self.vars['observed']) & exo
        for v in obs_exo:
            if v not in cov[v]:
                cov[v][v] = self.symb_starting_values
        for v in chain(self.vars['endogenous'], self.vars['latent']):
            if v not in cov[v]:
                cov[v][v] = None
        for a, b in combinations(obs_exo, 2):
            if a not in cov[b] and b not in cov[a]:
                cov[a][b] = self.symb_starting_values
        if not self.cov_diag:
            exo_lat = self.vars['exogenous'] & self.vars['latent']
            for a, b in chain(combinations(self.vars['output'], 2),
                              combinations(exo_lat, 2)):
                if b not in cov[a] and a not in cov[b]:
                    cov[a][b] = None

    def create_parameters(self, effects: dict):
        """
        Instantiate parameters in a model.

        Parameters
        ----------
        effects : dict
            Mapping of effects as returned by parse_desc.

        Raises
        ------
        NotImplementedError
            Raises in case of unknown effect symbol.

        Returns
        -------
        None.

        """

        self.dict_effects[self.symb_regression] = self.effect_regression
        self.dict_effects[self.symb_covariance] = self.effect_covariance
        self.dict_effects[self.symb_measurement] = self.effect_measurement

        for operation, items in effects.items():
            try:
                self.dict_effects[operation](items)
            except KeyError:
                raise NotImplementedError(f'{operation} is an unknown op.')

    def effect_regression(self, items: dict):
        """
        Work through regression operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        outputs = self.vars['_output']
        for lv, rvs in items.items():
            lv_is_out = lv in outputs
            if lv_is_out:
                mx = self.mx_lambda
                rows, cols = self.names_lambda
            else:
                mx = self.mx_beta
                rows, cols = self.names_beta
            i = rows.index(lv)
            for rv, mult in rvs.items():
                name = None
                active = True
                try:
                    val = float(mult)
                    active = False
                except (TypeError, ValueError):
                    if mult is not None:
                        if mult == self.symb_starting_values:
                            active = False
                        else:
                            name = mult
                    val = None
                if name is None:
                    self.n_param_reg += 1
                    name = '_b%s' % self.n_param_reg
                j = cols.index(rv)
                ind = (i, j)
                self.add_param(name=name, matrix=mx, indices=ind, start=val,
                               active=active, symmetric=False,
                               bound=(None, None))

    def effect_covariance(self, items: dict):
        """
        Work through covariance operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Returns
        -------
        None.

        """
        inners = self.vars['inner']
        lats = self.vars['latent']
        for lv, rvs in items.items():
            lv_is_inner = lv in inners
            for rv, mult in rvs.items():
                name = None
                try:
                    val = float(mult)
                    active = False
                except (TypeError, ValueError):
                    active = True
                    if mult is not None:
                        if mult != self.symb_starting_values:
                            name = mult
                        else:
                            active = False
                    val = None
                rv_is_inner = rv in inners
                if name is None:
                    self.n_param_cov += 1
                    name = '_c%s' % self.n_param_cov
                if lv_is_inner and rv_is_inner:
                    mx = self.mx_psi
                    rows, cols = self.names_psi
                else:
                    mx = self.mx_theta
                    rows, cols = self.names_theta
                    if lv_is_inner != rv_is_inner:
                        print('Covariances between _outputs and \
                                     inner variables are not recommended.')
                i, j = rows.index(lv), cols.index(rv)
                ind = (i, j)
                if i == j:
                    # if self.baseline and lv in lats:
                    #     continue
                    bound = (0, None)
                    symm = False
                else:
                    # if self.baseline:
                    #     continue
                    bound = (None, None)
                    symm = True
                self.add_param(name, matrix=mx, indices=ind, start=val,
                               active=active, symmetric=symm, bound=bound)

    def effect_measurement(self, items: dict):
        """
        Work through measurement operation.

        Parameters
        ----------
        items : dict
            Mapping lvalues->rvalues->multiplicator.

        Raises
        -------
        Exception
            Rises when indicator is misspecified and not observable.

        Returns
        -------
        None.

        """
        reverse_dict = defaultdict(dict)
        self.first_manifs = defaultdict(lambda: None)
        obs = self.vars['observed']
        for lat, inds in items.items():
            first = None
            lt = list()
            for ind, mult in inds.items():
                if ind not in obs:
                    print(f'Manifest variables should be observed,\
                                   but {ind} appears to be latent.')
                try:
                    float(mult)
                    if first is None:
                        first = len(lt)
                except (TypeError, ValueError):
                    if mult == self.symb_starting_values and first is None:
                        first = len(lt)
                lt.append((ind, mult))
            if first is None:
                for i, (ind, mult) in enumerate(lt):
                    if mult is None:
                        first = i
                        break
                if first is None:
                    print('No fixed loadings for %s.', lat)
                else:
                    lt[first] = (ind, 1.0)
            for ind, mult in lt:
                reverse_dict[ind][lat] = mult
            if first is not None:
                self.first_manifs[lat] = lt[first][0]
        self.effect_regression(reverse_dict)

    def calc_sigma(self):
        """
        Calculate model-implied covariance matrix.

        Returns
        -------
        sigma : np.ndarray
            Sigma model-implied covariance matrix.
        tuple
            Tuple of auxiliary matrics Lambda @ C and C, where C = (I - B)^-1.

        """

        self.update_matrices()
        beta, lamb, psi, theta = self.matrices
        c = torch.linalg.inv(self.identity_c - beta)
        m = lamb @ c
        return m @ psi @ m.T + theta, (m, c)

    def update_matrices(self):
        """
        Update all matrices from a parameter vector.
        """
        for mx, (r1, r2) in zip(self.matrices, self.param_ranges):
            if r1:
                mx[r1] = self.torch_param_val[r2]

    def load_starting_values(self):
        """
        Load starting values for parameters from empirical data.

        Returns
        -------
        None.

        """
        params_to_start = set()
        for name, param in self._par.items():
            if param.start is None:
                params_to_start.add(name)
                loc = param.locations[0]
                n = next(n for n in range(len(self.matrices))
                         if self.matrices[n] is loc.matrix)
                row, col = self.names[n]
                lval, rval = row[loc.indices[0]], col[loc.indices[1]]
                param.start = self.start_rules[n](lval, rval)
            for loc in param.locations:
                loc.matrix[loc.indices] = param.start
                if loc.symmetric:
                    loc.matrix[loc.indices[::-1]] = param.start
        if params_to_start:
            self.params_to_start = params_to_start

    def operation_define(self, operation):
        """
        Works through DEFINE command.

        Here, its main purpose is to load ordinal variables into the model.
        Parameters
        ----------
        operation : Operation
            Operation namedtuple.

        Returns
        -------
        None.

        """
        if operation.params and operation.params[0] == 'ordinal':
            if 'ordinal' not in self.vars:
                ords = set()
                self.vars['ordinal'] = ords
            else:
                ords = self.vars['ordinal']
            ords.update(operation.onto)

    def prepare_params(self):
        """
        Prepare structures for effective optimization routines.

        Returns
        -------
        None.

        """
        active_params = {name: param
                         for name, param in self._par.items()
                         if param.active}
        param_vals = [None] * len(active_params)
        diff_matrices = [None] * len(active_params)
        ranges = [[list(), list()] for _ in self.matrices]
        for i, (_, param) in enumerate(active_params.items()):
            param_vals[i] = param.start
            dm = [0] * len(self.matrices)
            for loc in param.locations:
                n = next(n for n in range(len(self.matrices))
                         if self.matrices[n] is loc.matrix)
                ranges[n][0].append(loc.indices)
                ranges[n][1].append(i)
                t = np.zeros_like(loc.matrix)
                t[loc.indices] = 1.0
                if loc.symmetric:
                    rind = loc.indices[::-1]
                    ranges[n][0].append(rind)
                    ranges[n][1].append(i)
                    t[rind] = 1.0
                dm[n] += t
            diff_matrices[i] = [m if type(m) is not int else None for m in dm]
        for rng in ranges:
            rng[0] = tuple(zip(*rng[0]))
        self.param_vals = np.array(param_vals)
        self.param_ranges = ranges
        self.mx_diffs = diff_matrices
        self.identity_c = torch.eye(self.mx_beta.shape[0])

    def add_param(self, name: str, active: bool, start: float, bound: tuple,
                  matrix: np.ndarray, indices: tuple, symmetric: bool):
        """
        Add parameter/update parameter locations in semopy matrices.

        If name is not present in self.parameters, then just locations will be
        updated. Otherwise, a new Parameter instance is added to
        self.parameters.
        Parameters
        ----------
        name : str
            Name of parameter.
        active : bool
            Is the parameter "active", i.e. is it subject to further
            optimization.
        start : float
            Starting value of parameter. If parameter is not active, then it is
            a fixing value.
        bound : tuple
            Bound constraints on parameter (a, b). None is treated as infinity.
        matrix : np.ndarray
            Reference to matrix.
        indices : tuple
            Indices of parameter in the matrix.
        symmetric : bool
            Should be True if matrix is symmetric.

        Returns
        -------
        None.

        """
        loc = ParameterLoc(matrix=matrix, indices=indices,
                           symmetric=symmetric)
        if name in self._par:
            p = self._par[name]
            p.locations.append(loc)
        else:
            p = Parameter(active=active, start=start, bound=bound,
                          locations=[loc])
            self._par[name] = p

    def apply_operations(self, operations: dict):
        """
        Apply operations to model.

        Parameters
        ----------
        operations : dict
            Mapping of operations as returned by parse_desc.

        Raises
        ------
        NotImplementedError
            Raises in case of unknown command name.

        Returns
        -------
        None.

        """
        for command, items in operations.items():
            try:
                list(map(self.dict_operations[command], items))
            except KeyError:
                raise NotImplementedError(f'{command} is an unknown command.')

    def load_data(self, data: pd.DataFrame, covariance=None, groups=None):
        """
        Load dataset from data matrix.

        Parameters
        ----------
        data : pd.DataFrame
            Dataset with columns as variables and rows as observations.
        covariance : pd.DataFrame, optional
            Custom covariance matrix. The default is None.
        groups : list, optional
            List of group names to center across. The default is None.

        Returns
        -------
        None.

        """
        if groups is None:
            groups = list()
        obs = self.vars['observed']
        for group in groups:
            for g in data[group].unique():
                inds = data[group] == g
                if sum(inds) == 1:
                    continue
                data.loc[inds, obs] -= data.loc[inds, obs].mean()
                data.loc[inds, group] = g
        self.mx_data = data[obs].values
        if len(self.mx_data.shape) != 2:
            self.mx_data = self.mx_data[:, np.newaxis]
        if 'ordinal' not in self.vars:
            self.load_cov(covariance.loc[obs, obs].values
                          if covariance is not None else cov(self.mx_data))
        else:
            inds = [obs.index(v) for v in self.vars['ordinal']]
            self.load_cov(hetcor(self.mx_data, inds))
        self.n_samples, self.n_obs = self.mx_data.shape

    def load_cov(self, covariance: np.ndarray):
        """
        Load covariance matrix.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix.

        Returns
        -------
        None.

        """
        if type(covariance) is pd.DataFrame:
            obs = self.vars['observed']
            covariance = covariance.loc[obs, obs].values
        if covariance.size == 1:
            covariance.resize((1, 1))
        self.mx_cov = covariance
        try:
            self.mx_cov_inv, self.cov_logdet = chol_inv2(self.mx_cov)
        except np.linalg.LinAlgError:
            print('Sample covariance matrix is not PD. It may '
                            'indicate that data is bad. Also, it arises often '
                            'when polychoric/polyserial correlations are used.'
                            ' semopy now will run nearPD subroutines.')
            self.mx_cov = cov_nearest(covariance, threshold=1e-2)
            self.mx_cov_inv, self.cov_logdet = chol_inv2(self.mx_cov)
        self.mx_covlike_identity = np.identity(self.mx_cov.shape[0])

    def start_beta(model, lval: str, rval: str):
        """
        Calculate starting value for parameter in data given data in model.

        Parameters in beta are traditionally set to 0 at start.
        Parameters
        ----------
        model : Model
            Model instance.
        lval : str
            L-value name.
        rval : str
            R-value name.

        Returns
        -------
        float
            Starting value.

        """
        return 0.0

    def start_lambda(model, lval: str, rval: str):
        """
        Calculate starting value for parameter in data given data in model.

        Manifest variables are regressed onto their counterpart with fixed
        regression coefficient.
        Parameters
        ----------
        model : Model
            Model instance.
        lval : str
            L-value name.
        rval : str
            R-value name.

        Returns
        -------
        float
            Starting value.

        """
        if rval not in model.vars['latent']:
            return 0.0
        obs = model.vars['observed']
        first = rval
        while first not in obs:
            try:
                first = model.first_manifs[first]
                if first == rval:
                    return 0.0
            except KeyError:
                return 0.0
        if first is None or not hasattr(model, 'mx_data'):
            return 0.0
        i, j = obs.index(first), obs.index(lval)
        data = model.mx_data
        x, y = data[:, i], data[:, j]
        mask = np.isfinite(x) & np.isfinite(y)
        return linregress(x[mask], y[mask]).slope

    def start_psi(model, lval: str, rval: str):
        """
        Calculate starting value for parameter in data given data in model.

        Exogenous covariances are fixed to their empirical values.
        All other variances are halved. Latent variances are set to 0.05,
        everything else is set to zero.
        Parameters
        ----------
        model : Model
            Model instance.
        lval : str
            L-value name.
        rval : str
            R-value name.

        Returns
        -------
        float
            Starting value.

        """
        lat = model.vars['latent']
        if rval in lat or lval in lat:
            if rval == lval:
                return 0.05
            return 0.0
        exo = model.vars['exogenous']
        obs = model.vars['observed']
        i, j = obs.index(lval), obs.index(rval)
        if lval in exo:
            return model.mx_cov[i, j]
        elif i == j:
            return model.mx_cov[i, j] / 2
        return 0.0

    def start_theta(model, lval: str, rval: str):
        """
        Calculate starting value for parameter in data given data in model.

        Variances are set to half of observed variances.
        Parameters
        ----------
        model : Model
            Model instance.
        lval : str
            L-value name.
        rval : str
            R-value name.

        Returns
        -------
        float
            Starting value.

        """
        if lval != rval:
            return 0.0
        obs = model.vars['observed']
        i, j = obs.index(lval), obs.index(rval)
        return model.mx_cov[i, j] / 2

    def prepare(self, data):

        self.load_data(data)
        self.load_starting_values()
        self.prepare_params()

        self.torch_param_val = torch.nn.Parameter(torch.from_numpy(self.param_vals))

    def calc_fim(self, inverse=False):
        """
        Calculate Fisher Information Matrix.

        Exponential-family distributions are assumed.
        Parameters
        ----------
        inverse : bool, optional
            If True, function also returns inverse of FIM. The default is
            False.

        Returns
        -------
        np.ndarray
            FIM.
        np.ndarray, optional
            FIM^{-1}.

        """
        sigma, (m, c) = self.calc_sigma()
        sigma_grad = self.calc_sigma_grad(m, c)
        inv_sigma = chol_inv(sigma.detach().numpy())
        sz = len(sigma_grad)
        info = np.zeros((sz, sz))
        sgs = [sg @ inv_sigma for sg in sigma_grad]
        n = self.n_samples
        if n is None:
            raise AttributeError('For FIM estimation in a covariance-matr'
                                 'ix-only setting, you must provide the'
                                 ' n_samples argument to the fit or load'
                                 ' methods.')
        n /= 2

        for i in range(sz):
            for k in range(i, sz):
                info[i, k] = n * np.einsum('ij,ji->', sgs[i], sgs[k])
        fim = info + np.triu(info, 1).T
        if inverse:
            try:
                fim_inv = chol_inv(fim)
                self._fim_warn = False
            except np.linalg.LinAlgError:
                print("Fisher Information Matrix is not PD."
                             "Moore-Penrose inverse will be used instead of "
                             "Cholesky decomposition. See "
                             "10.1109/TSP.2012.2208105.")
                self._fim_warn = True
                fim_inv = np.linalg.pinv(fim)
            return (fim, fim_inv)
        return fim

    def calc_sigma_grad(self, m: np.ndarray, c: np.ndarray):
        """
        Calculate gradient wrt to parameters vector of Sigma matrix.

        Parameters
        ----------
        m : np.ndarray
            Auxilary matrix returned from calc_sigma Lambda @ C.
        c : np.ndarray
            Auxilary matrix returned from calc_sigma (I - B)^-1

        Returns
        -------
        grad : list
            List of derivatives of Sigma wrt to parameters vector.

        """
        psi = self.matrices[2].detach().numpy()
        m = m.detach().numpy()
        c = c.detach().numpy()
        m_t = m.T
        p = c @ psi
        d = p @ m_t
        grad = list()
        for dmxs in self.mx_diffs:
            g = np.float32(0.0)
            if dmxs[0] is not None:  # Beta
                t = dmxs[0] @ p
                g += m @ (t + t.T) @ m_t
            if dmxs[1] is not None:  # Lambda
                t = dmxs[1] @ d
                g += t + t.T
            if dmxs[2] is not None:  # Psi
                g += m @ dmxs[2] @ m_t
            if dmxs[3] is not None:  # Theta
                g += dmxs[3]
            grad.append(g)
        return grad


def main():

    # data = pd.read_csv("worland5.csv")
    #
    # desc = \
    #     "read ~ ppsych + motiv" + "\n" + \
    #     "arith ~ motiv"

    data = pd.read_csv("fake.csv")

    # theta1 = np.array([[1., 2., 4.], [-3., -2., 4.]])
    # theta2 = np.array([5., -3., 4.])
    # cst = -3.

    print(data)

    desc = \
      "h ~ x\n" \
      "m ~ x\n" \
      "e ~ x\n" \
      "y ~ h + m + e"

    model = Model(desc)
    # optim = torch.optim.Adam(model.parameters())  # init the optimizer
    # for epoch in range(1000):
    #     optim.zero_grad()  # reset the gradients of the parameters
    #     Sigma = model()  # compute the model-implied covariance matrix
    #     loss = mvn_negloglik(data, Sigma)  # compute the negative log-likelihood, dat tensor should exist
    #     loss.backward()  # compute the gradients and store them in the parameter tensors
    #     optim.step()  # take a step in the negative gradient direction using adam

    # {'all': {'read', 'ppsych', 'motiv', 'arith'}, 'endogenous': {'read', 'arith'}, 'exogenous': {'ppsych', 'motiv'}, 'observed': ['arith', 'read', 'motiv', 'ppsych'], 'latent': set(), 'indicator': set(), 'output': {}, '_output': {'read', 'arith'}, 'inner': ['motiv', 'ppsych']}

    model.prepare(data)

    torch_data = torch.from_numpy(data[model.vars['observed']].to_numpy())

    optim = torch.optim.Adam(model.parameters(), lr=0.01)  # init the optimizer

    epochs = 2000
    with tqdm(total=epochs) as pb:
        for epoch in range(epochs):
            optim.zero_grad()  # reset the gradients of the parameters
            Sigma, _ = model.calc_sigma()  # compute the model-implied covariance matrix
            loss = mvn_negloglik(torch_data, Sigma)  # compute the negative log-likelihood, dat tensor should exist
            loss.backward(retain_graph=True)  # compute the gradients and store them in the parameter tensors
            optim.step()  # take a step in the negative gradient direction using adam

            pb.update()
    # defaultdict(<class 'dict'>, {'motiv': {'motiv': 'START', 'ppsych': 'START'}, 'ppsych': {'ppsych': 'START'}, 'read': {'read': None}, 'arith': {'arith': None}})
    model.update_matrices()
    print(inspect_list(model))


if __name__ == "__main__":
    main()
