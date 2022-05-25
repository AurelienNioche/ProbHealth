import os
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch import distributions as dist

from . flows import NormalizingFlows


def likelihood(
        z_flow, n_sample,
        n_u, n_w,
        u, w, x, r, y,
        **kwargs
):

    # Z: Sample base distribution and apply transformation
    z0_Z = z_flow.sample_base_dist(n_sample)
    zk_Z, ln_q0_Z, sum_ld_Z = z_flow(z0_Z)

    # Get Z-values used for first parameter
    Zu1 = zk_Z[:, :n_u].T
    Zw1 = zk_Z[:, n_u:n_w + n_u].T

    # Get Z-values used for second first parameter
    Zu2 = zk_Z[:, n_w + n_u:n_w + n_u * 2].T
    Zw2 = zk_Z[:, n_w + n_u * 2:].T

    # Compute Z-values for both parameters
    Z1 = Zu1[u] + Zw1[w]
    Z2 = Zu2[u] + Zw2[w]

    # Go to constrained space
    param1 = torch.exp(Z1)
    param2 = torch.sigmoid(Z2)

    # Compute log probability of recall
    log_p = -param1 * x * (1 - param2) ** r

    # Comp. log-likelihood of observations
    ll = torch.distributions.Bernoulli(
        probs=torch.exp(log_p)
    ).log_prob(y)
    ll_sum = ll.sum(axis=0)  # Sum over observations (final shape: n_sample)

    return ll_sum, {
        'll': ll.mean(axis=1),  # Mean over samples (final shape: n_obs)
        'Zu1': Zu1,
        'Zu2': Zu2,
        'Zw1': Zw1,
        'Zw2': Zw2,
        'log_p': log_p,
        'ln_q0_Z': ln_q0_Z,
        'sum_ld_Z': sum_ld_Z}


def free_energy(
        theta_flow,
        n_sample,
        ll_sum,
        Zu1,
        Zu2,
        Zw1,
        Zw2,
        ln_q0_Z,
        sum_ld_Z,
        total_n_obs,
        batch_size,
        **kwargs):

    # # Get unique users for this (mini)batch
    # uniq_u = np.unique(u)
    # uniq_w = np.unique(w)

    # θ: Sample base distribution and apply transformation
    z0_θ = theta_flow.sample_base_dist(n_sample)
    zk_θ, ln_q0_θ, sum_ld_θ = theta_flow(z0_θ)

    # Get θ-values for both parameters
    half_mu1, log_var_u1, log_var_w1 = zk_θ[:, :3].T
    half_mu2, log_var_u2, log_var_w2 = zk_θ[:, 3:].T

    sg_u1 = torch.exp(0.5 * log_var_u1)
    sg_u2 = torch.exp(0.5 * log_var_u2)
    sg_w1 = torch.exp(0.5 * log_var_w1)
    sg_w2 = torch.exp(0.5 * log_var_w2)

    # Comp. likelihood Z-values given population parameterization for first parameter
    ll_Zu1 = dist.Normal(half_mu1, sg_u1).log_prob(Zu1).sum(axis=0)  # .mean(axis=1)  # mean over sample // shape: N user
    ll_Zw1 = dist.Normal(half_mu1, sg_w1).log_prob(Zw1).sum(axis=0)  # .mean(axis=1)  # mean over sample // shape: N word

    # Comp. likelihood Z-values given population parameterization for second parameter
    ll_Zu2 = dist.Normal(half_mu2, sg_u2).log_prob(Zu2).sum(axis=0)  # .mean(axis=1)
    ll_Zw2 = dist.Normal(half_mu2, sg_w2).log_prob(Zw2).sum(axis=0)  # .mean(axis=1)

    # Add all the loss terms and compute average (= expectation estimate)
    ln_q0 = ln_q0_θ + ln_q0_Z  # log q0
    sum_ln_det = sum_ld_θ + sum_ld_Z  # sum log determinant
    lls = ll_sum + ll_Zu1 + ll_Zu2 + ll_Zw1 + ll_Zw2

    batch_ratio = total_n_obs / batch_size
    # beta = min(batch_ratio, 0.01 + batch_ratio*epoch/total_n_epochs)
    loss = (ln_q0 - sum_ln_det - batch_ratio*lls).sum() / (n_sample * total_n_obs)

    # Return - ELBO
    return loss, {
        'mu1': half_mu1*2.,
        'mu2': half_mu2*2.,
        'sg_u1': sg_u1,
        'sg_u2': sg_u2,
        'sg_w1': sg_w1,
        'sg_w2': sg_w2,
    }


def train(
        dataset,
        bkp_folder,
        bkp_name,
        load_if_exists=False,
        batch_size=None,
        training_split=1.0,
        flow_length=16,
        epochs=5000,
        optimizer_name="Adam",
        optimizer_kwargs=None,
        initial_lr=0.01,
        constant_lr=False,
        scheduler_name=None,
        n_sample=40,
        seed=123,
        truth=None,  # just for putting all backup at same place
        ):

    z_bkp_file = f"{bkp_folder}/{bkp_name}_z.p"
    theta_bkp_file = f"{bkp_folder}/{bkp_name}_theta.p"
    hist_bkp_file = f"{bkp_folder}/{bkp_name}_hist.p"
    truth_bkp_file = f"{bkp_folder}/{bkp_name}_truth.p"
    config_bkp_file = f"{bkp_folder}/{bkp_name}_config.p"

    if load_if_exists:
        try:
            z_flow = NormalizingFlows.load(z_bkp_file)
            theta_flow = NormalizingFlows.load(theta_bkp_file)
            hist = torch.load(hist_bkp_file)
            config = torch.load(config_bkp_file)
            if truth is not None:
                truth = torch.load(truth_bkp_file)

            print("Load successfully from backup")
            return z_flow, theta_flow, hist, config

        except FileNotFoundError:
            print("Didn't find backup. Run the inference process instead...")

    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(dataset)

    if batch_size is None:
        batch_size = n

    if training_split < 1.0:
        n_training = int(training_split * n)
        n_validation = n - n_training

        train_set, val_set = random_split(
            dataset,
            [n_training, n_validation])

        print("N training", n_training)
        print("N validation", n_validation)

        training_data = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        validation_data = DataLoader(val_set, batch_size=n_validation, shuffle=True)
    else:
        training_data = DataLoader(dataset, batch_size=n, shuffle=False)
        n_training = n
        n_validation = 0
        validation_data = ()

    n_u, n_w = dataset.n_u, dataset.n_w

    z_flow = NormalizingFlows(dim=(n_u + n_w) * 2, flow_length=flow_length)
    theta_flow = NormalizingFlows(6, flow_length=flow_length)

    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    optimizer = getattr(optim, optimizer_name)(
            list(z_flow.parameters()) + list(theta_flow.parameters()),
            lr=initial_lr, **optimizer_kwargs)

    if scheduler_name is not None:
        scheduler = getattr(optim.lr_scheduler, scheduler_name)(optimizer)
        assert constant_lr is False
    else:
        scheduler = None

    bce_loss = torch.nn.BCELoss()

    metrics = [
        "bce",
        "free_energy",
        "accuracy"
    ]

    key_vars_free_energy = [
        'mu1',
        'mu2',
        'sg_u1',
        'sg_u2',
        'sg_w1',
        'sg_w2'
    ]

    key_vars_ll = [
        'Zu1',
        'Zu2',
        'Zw1',
        'Zw2',
    ]

    # Additional key is 'p'

    parameters = key_vars_ll + key_vars_free_energy + ['p', ]

    hist = {k: {m: [] for m in metrics} for k in ['train', 'val']}
    hist.update({k: {p: [] for p in parameters} for k in ['comp_truth_train', 'comp_truth_val']})

    with tqdm(total=epochs) as pbar:

        for i in range(epochs):

            theta_flow.train()
            z_flow.train()

            metric_values = {k: [] for k in metrics}
            param_values = {k: [] for k in parameters}

            for d in training_data:

                optimizer.zero_grad()

                # kwargs_likelihood = {k: v for k, v in d.items() if k != 'i'}

                ll_sum, ll_var = likelihood(
                    z_flow=z_flow,
                    n_sample=n_sample,
                    n_u=n_u,
                    n_w=n_w,
                    **d)

                loss, loss_var = free_energy(
                    theta_flow=theta_flow,
                    n_sample=n_sample,
                    ll_sum=ll_sum,
                    total_n_obs=n_training,
                    batch_size=batch_size,
                    **ll_var)

                loss.backward()
                optimizer.step()

                if i > 0:
                    pbar.set_postfix({'loss': loss.item()})

                if constant_lr:
                    for g in optimizer.param_groups:
                        g['lr'] = initial_lr

                if scheduler is not None:
                    scheduler.step()

                z_flow.eval()
                theta_flow.eval()

                ll = ll_var["ll"]

                p_y = torch.exp(ll)
                y = d['y'].squeeze()

                with torch.no_grad():
                    for k in metrics:
                        if k == 'bce':
                            r = bce_loss(p_y, y).item()
                        elif k == 'accuracy':
                            y_pred = (p_y > 0.5).float()
                            r = (y == y_pred).sum() / y.size(0)
                        elif k == 'free_energy':
                            r = loss.item()
                        else:
                            raise ValueError

                        metric_values[k].append(r)

                    if truth is not None:

                        for k in key_vars_ll:
                            try:
                                obs = ll_var[k].mean(axis=1)
                                delta = (obs - truth[k]).abs().mean().item()
                                param_values[k].append(delta)
                            except Exception as e:
                                raise e

                        for k in key_vars_free_energy:

                            obs = loss_var[k].mean()
                            delta = (obs - truth[k]).abs().item()
                            param_values[k].append(delta)

                        obs = torch.exp(ll_var['log_p']).mean(axis=1)
                        index = d['i']
                        tr = truth['p'][index]
                        delta = (obs - tr).abs().mean().item()
                        param_values['p'].append(delta)

            for k in metrics:
                hist['train'][k].append(np.mean(metric_values[k]))

            if truth is not None:
                for k in parameters:
                    hist['comp_truth_train'][k].append(np.mean(param_values[k]))

            pbar.set_postfix({'loss': loss.item()})
            pbar.update()

            z_flow.eval()
            theta_flow.eval()

            metric_values = {k: [] for k in metrics}
            param_values = {k: [] for k in parameters}

            with torch.no_grad():

                for d in validation_data:

                    ll_sum, ll_var = likelihood(
                        z_flow=z_flow,
                        n_sample=n_sample,
                        n_u=n_u,
                        n_w=n_w,
                        **d)

                    ll = ll_var["ll"]
                    p_y = torch.exp(ll)
                    y = d['y'].squeeze()

                    for k in metrics:
                        if k == 'bce':
                            r = bce_loss(p_y, y).item()
                        elif k == 'accuracy':
                            y_pred = (p_y > 0.5).float()
                            r = (y == y_pred).sum() / y.size(0)
                        elif k == 'free_energy':
                            fe, fe_var = free_energy(
                                theta_flow=theta_flow,
                                n_sample=n_sample,
                                ll_sum=ll_sum,
                                **ll_var,
                                total_n_obs=n_validation,
                                batch_size=batch_size)
                            r = fe.item()
                        else:
                            raise ValueError(f'Metric not recognized: {k}')

                        metric_values[k].append(r)

                    if truth is not None:

                        for k in key_vars_ll:
                            try:
                                obs = ll_var[k].mean(axis=1)
                                delta = (obs - truth[k]).abs().mean().item()
                                param_values[k].append(delta)
                            except Exception as e:
                                raise e

                        for k in key_vars_free_energy:

                            obs = loss_var[k].mean()
                            delta = (obs - truth[k]).abs().item()
                            param_values[k].append(delta)

                        obs = torch.exp(ll_var['log_p']).mean(axis=1)
                        index = d['i']
                        tr = truth['p'][index]
                        delta = (obs - tr).abs().mean().item()
                        param_values['p'].append(delta)

            for k in metrics:
                v = metric_values[k]
                if len(v):
                    hist['val'][k].append(np.mean(v))

            if truth is not None:
                for k in parameters:
                    v = param_values[k]
                    if len(v):
                        hist['comp_truth_val'][k].append(np.mean(v))

    config = dict(
        batch_size=batch_size,
        training_split=training_split,
        flow_length=flow_length,
        epochs=epochs,
        optimizer_name=optimizer_name,
        optimizer_kwargs=optimizer_kwargs,
        initial_lr=initial_lr,
        constant_lr=constant_lr,
        scheduler_name=scheduler_name,
        n_sample=n_sample,
        seed=seed
    )

    os.makedirs(bkp_folder, exist_ok=True)

    z_flow.save(z_bkp_file)
    theta_flow.save(theta_bkp_file)
    torch.save(f=hist_bkp_file, obj=hist)
    torch.save(f=config_bkp_file, obj=config)

    if truth is not None:
        torch.save(obj=truth, f=truth_bkp_file)

    return z_flow, theta_flow, hist, config


