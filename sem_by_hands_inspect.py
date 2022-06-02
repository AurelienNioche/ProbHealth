import pandas as pd
import numpy as np
from semopy import stats


def inspect_list(model, information='expected', std_est=False,
                 se_robust=False, index_names=False):
    """
    Get a pandas DataFrame containin a view of parameters estimates.

    Parameters
    ----------
    model : Model
        Model.
    information : str
        If 'expected', expected Fisher information is used. Otherwise,
        observed information is employed. If None, the no p-values are
        calculated. The default is 'expected'.
    std_est : bool
        If True, standardized coefficients are also returned as Std. Ests col.
        The default is False.
    se_robust : bool, optional
        If True, then robust SE are computed instead. Robustness here means
        that MLR-esque sandwich correction is applied. The default is False.
    index_names : bool, optional
        If True, then the returned DataFrame has parameter names as indices.

    Returns
    -------
    pd.DataFrame
        DataFrame with parameters information.

    """
    res = list()
    try:
        vals = model.param_vals
    except AttributeError:
        vals = ['Not estimated'] * len(model.parameters)
        information = None
        std_est = False
        se_robust = False
    if std_est:
        sigma, (_, c) = model.calc_sigma()
        w_cov = c @ model.mx_psi @ c.T
        stds = np.sqrt(w_cov.diagonal())
        std_full = np.sqrt(sigma.diagonal())
        if std_est == 'lv':
            std_full[:] = 1.0
    if information is not None:
        se = stats.calc_se(model, information=information, robust=se_robust)
        zscores = stats.calc_zvals(model, std_errors=se)
        pvals = stats.calc_pvals(model, z_scores=zscores)
    else:
        se = ['-'] * len(vals)
        zscores = pvals = se
    keys = list(model._par.keys())
    keys_active = [k for k in keys if model._par[k].active]
    param_names = list()
    # Beta
    if hasattr(model, 'mx_beta'):
        mx = model.mx_beta
        names = model.names_beta
        op = '~'
        for name, param in model._par.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if std_est:
                        val_std = val / stds[ind[0]] * stds[ind[1]]
                        res.append((a, op, b, val, val_std, std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))

    means = list()
    # Gamma1
    if hasattr(model, 'mx_gamma1'):
        mx = model.mx_gamma1
        names = model.names_gamma1
        op = '~'
        for name, param in model.parameters.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if b == '1':
                        if std_est:
                            val_std = val * stds[ind[1]] / std_full[ind[0]]
                            means.append((a, op, b, val, val_std, std, zs, pval))
                        else:
                            means.append((a, op, b, val, std, zs, pval))
                    else:
                        if std_est:
                            val_std = val * stds[ind[1]] / std_full[ind[0]]
                            res.append((a, op, b, val, val_std, std, zs, pval))
                        else:
                            res.append((a, op, b, val, std, zs, pval))

    # Gamma2
    if hasattr(model, 'mx_gamma2'):
        mx = model.mx_gamma2
        names = model.names_gamma2
        op = '~'
        for name, param in model.parameters.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if b == '1':
                        if std_est:
                            val_std = val / std_full[ind[0]]
                            means.append((a, op, b, val, val_std, std, zs, pval))
                        else:
                            means.append((a, op, b, val, std, zs, pval))
                    else:
                        if std_est:
                            val_std = val / std_full[ind[0]]
                            res.append((a, op, b, val, val_std, std, zs, pval))
                        else:
                            res.append((a, op, b, val, std, zs, pval))

    # Lambda
    if hasattr(model, 'mx_lambda'):
        mx = model.mx_lambda
        names = model.names_lambda
        op = '~'
        for name, param in model._par.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if std_est:
                        val_std = val * stds[ind[1]]
                        if std_est != 'lv':
                            val_std /= std_full[ind[0]]
                        res.append((a, op, b, val, val_std, std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))
    res.extend(means)
    means.clear()
    # Psi
    if hasattr(model, 'mx_psi'):
        mx = model.mx_psi
        names = model.names_psi
        op = '~~'
        obs_exo = set(model.vars['observed']) & model.vars['exogenous']
        for name, param in model._par.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        if a in obs_exo and b in obs_exo:
                            continue
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    param_names.append(name)
                    if std_est:
                        val_std = val / stds[ind[0]] / stds[ind[1]]
                        res.append((a, op, b, val, val_std, std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))
    # Theta
    if hasattr(model, 'mx_theta'):
        mx = model.mx_theta
        names = model.names_theta
        op = '~~'
        for name, param in model._par.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if std_est:
                        val_std = val / std_full[ind[0]] / std_full[ind[1]]
                        res.append((a, op, b, val, val_std, std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))

    # D -- Variance of random effects matrix
    rgr = hasattr(model, 'effects_names')
    if rgr and model.effects_names:
        rgr = True
    if hasattr(model, 'mx_d'):
        mx = model.mx_d
        names = model.names_d
        op = 'RF'
        if rgr:
            try:
                rgr = next(iter(model.effects_names))
            except StopIteration:
                rgr = False
        for name, param in model.parameters.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if rgr and a == b:
                        op = '~'
                        val = val ** 0.5
                        b = rgr
                    else:
                        op = 'RF'
                    if std_est:
                        val_est = val / std_full[ind[0]] / std_full[ind[1]]
                        res.append((a, op, b, val, val_est, std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))
    # D_(i) -- Variance of random effects matrix in ModelGeneralizedEffects
    i = 1
    if rgr:
        rgr = iter(model.effects_names)
    while hasattr(model, f'mx_d{i}'):
        mx = getattr(model, f'mx_d{i}')
        names = getattr(model, f'names_d{i}')
        if rgr:
            eff = next(rgr)
        for name, param in model.parameters.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if rgr and a == b:
                        op = '~'
                        val = val ** 0.5
                        b = eff
                    else:
                        op = f'RF{i}'
                    if std_est:
                        val_est = val / std_full[ind[0]] / std_full[ind[1]]
                        res.append((a, op, b, val, val_est, std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))
        i += 1
    # v -- Variance of random effects variable
    if hasattr(model, 'mx_v'):
        mx = model.mx_v
        names = model.names_v
        op = 'RF(v)'
        for name, param in model.parameters.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    if std_est:
                        res.append((a, op, b, val, '-', std, zs, pval))
                    else:
                        res.append((a, op, b, val, std, zs, pval))
    # Data_imp -- Matrix of imputed data
    if hasattr(model, 'mx_data_imp'):
        mx = model.mx_data_imp
        names = model.names_data_imp
        op = '@'
        for name, param in model.parameters.items():
            for loc in param.locations:
                if loc.matrix is mx:
                    param_names.append(name)
                    ind = loc.indices
                    a, b = names[0][ind[0]], names[1][ind[1]]
                    if param.active:
                        i = keys_active.index(name)
                        val = vals[i]
                        std = se[i]
                        zs = zscores[i]
                        pval = pvals[i]
                    else:
                        val = param.start
                        std = '-'
                        zs = '-'
                        pval = '-'
                    res.append((a, op, b, val, std, zs, pval))
    if std_est:
        cols = ['lval', 'op', 'rval', 'Estimate', 'Est. Std', 'Std. Err',
                'z-value', 'p-value']
    else:
        cols = ['lval', 'op', 'rval', 'Estimate', 'Std. Err', 'z-value',
                'p-value']
    res = pd.DataFrame(res, columns=cols)
    if index_names:
        res.index = param_names
    return res