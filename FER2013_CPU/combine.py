"""Grouped development evaluation and frozen inference of FER2013 ensembles.

Models are supplied as ID-aligned cached logits. Use --freeze to fit on clean
PublicTest, then --evaluate-frozen to score a separate split without refitting.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

NUM_CLASSES = 7


def align(cache, ids, view_count):
    with np.load(cache, allow_pickle=False) as data:
        target_ids = data['ids']
        logits = data['logits']
        if 'finished' in data and int(data['finished']) != len(target_ids):
            raise ValueError(f'Prediction cache is incomplete: {cache}')
        if logits.ndim == 3:
            logits = logits[:, :view_count].mean(1)
        elif view_count is not None:
            logits = data['views'][:, :view_count].mean(1)
        if logits.shape != (len(ids), NUM_CLASSES) or not np.isfinite(logits).all():
            raise ValueError(f'Invalid logits in {cache}: {logits.shape}')
        if len(set(target_ids)) != len(ids):
            raise ValueError(f'Missing or duplicate sample IDs in {cache}')
        index = {sample_id: i for i, sample_id in enumerate(target_ids)}
        logits = logits[[index[sample_id] for sample_id in ids]].astype(np.float64)
        return logits - logits.mean(1, keepdims=True)


def nll(logits, y):
    return float((logsumexp(logits, axis=1) - logits[np.arange(len(y)), y]).mean())


def calibrate(logits, y):
    opt = minimize_scalar(lambda log_t: nll(logits / np.exp(log_t), y),
                          bounds=(-2.3, 3.7), method='bounded')
    return float(np.exp(opt.x))


def fit(method, x, y):
    if method == 'rms':
        return {'temperatures': np.sqrt((x*x).mean((1,2))).clip(1e-6).tolist()}
    temperatures = np.array([calibrate(z, y) for z in x])
    x = x / temperatures[:, None, None]
    params = {'temperatures': temperatures.tolist()}
    if method == 'equal':
        return params
    if method == 'accuracy':
        # A smooth top-one objective with a small cross-entropy term and shrinkage.
        models, samples, classes = x.shape
        initial = np.r_[np.full(models, 1 / models), np.zeros(classes)]
        indicators = np.eye(classes)[y]
        regularization = np.r_[np.full(models, 0.025), np.full(classes, 2.0)]
        rows = np.arange(samples)

        def objective(coefs):
            weights, bias = coefs[:models], coefs[models:]
            scores = np.einsum('m,mnk->nk', weights, x) + bias
            sharp = softmax(scores/0.30,axis=1)
            correct = sharp[rows,y]
            difference = coefs-initial
            gradient = ((sharp-indicators) * correct[:,None]/0.30
                        + 0.12*(softmax(scores,axis=1)-indicators))/samples
            loss = -correct.mean() + 0.12*nll(scores,y) + (regularization*difference**2).sum()/2
            derivatives = np.r_[np.einsum('nk,mnk->m',gradient,x),gradient.sum(0)]
            return float(loss), derivatives+regularization*difference

        selected = minimize(objective,initial,jac=True,method='SLSQP',
                            bounds=[(0,1)]*models+[(-0.25,0.25)]*classes,
                            constraints={'type':'eq','fun':lambda a:a[:models].sum()-1},
                            options={'maxiter':100,'ftol':1e-9})
        if not selected.success:
            raise ValueError(selected.message)
        params['weights'] = selected.x[:models].tolist()
        params['bias'] = selected.x[models:].tolist()
        return params
    if method in ('probability', 'logits'):
        inputs = softmax(x, axis=2) if method == 'probability' else x
        count = len(inputs)
        def loss(weights):
            z = np.einsum('m,mnk->nk', weights, inputs)
            penalty = 0.025 * np.square(weights).sum()
            if method == 'probability':
                return float(-np.log(z[np.arange(len(y)), y].clip(1e-12)).mean() + penalty)
            return nll(z, y) + penalty
        result = minimize(loss, np.ones(count)/count, method='SLSQP',
                          bounds=[(0., 1.)]*count,
                          constraints={'type':'eq', 'fun':lambda w: w.sum()-1},
                          options={'maxiter':100,'ftol':1e-9})
        if not result.success:
            raise ValueError(result.message)
        params['weights'] = result.x.tolist()
        return params
    flattened = x.transpose(1,0,2).reshape(len(y),-1)
    # A shrunk correction to a calibrated mean retains the ensemble as the prior.
    prior = x.mean(0)
    onehot = np.eye(NUM_CLASSES)[y]
    feature_count = flattened.shape[1]
    strength = 2.0 if method == 'residual' else 0.75
    def loss(coefs):
        coefs = coefs.reshape(feature_count+1, NUM_CLASSES)
        update = flattened @ coefs[:-1] + coefs[-1]
        logits = prior + update
        grad = (softmax(logits, axis=1)-onehot) / len(y)
        value = nll(logits,y) + strength * (coefs**2).sum()/2
        deriv = np.vstack([flattened.T@grad,grad.sum(0)]) + strength*coefs
        return value, deriv.ravel()
    result = minimize(loss,np.zeros((feature_count+1)*NUM_CLASSES),jac=True,method='L-BFGS-B',
                      options={'maxiter':100})
    if not result.success:
        raise ValueError(result.message)
    params['correction'] = result.x.reshape(feature_count+1,NUM_CLASSES).tolist()
    return params


def predict(method, params, x):
    x = x / np.asarray(params['temperatures'])[:, None, None]
    if method == 'equal':
        return softmax(x,axis=2).mean(0)
    if method == 'rms':
        return softmax(x.mean(0),axis=1)
    if method == 'probability':
        return np.einsum('m,mnk->nk', params['weights'], softmax(x,axis=2))
    if method in ('logits','accuracy'):
        return softmax(np.einsum('m,mnk->nk', params['weights'],x)+params.get('bias',0.),axis=1)
    coefs = np.asarray(params['correction'])
    flattened = x.transpose(1,0,2).reshape(x.shape[1],-1)
    return softmax(x.mean(0)+flattened@coefs[:-1]+coefs[-1],axis=1)


def measure(probabilities, y, clean):
    predictions = probabilities.argmax(1)
    correct = predictions == y
    return {'accuracy': float(correct.mean()), 'correct': int(correct.sum()),
            'samples': len(y), 'macro_f1':float(f1_score(y,predictions,average='macro')),
            'clean_accuracy':float(correct[clean].mean()), 'clean_samples':int(clean.sum()),
            'nll':float(-np.log(probabilities[np.arange(len(y)),y].clip(1e-12)).mean())}


def load(data_path, configs):
    with np.load(data_path, allow_pickle=False) as data:
        ids, y, groups, clean = [data[k] for k in ['ids','labels','hashes','clean_mask']]
    x = np.stack([align(config['file'], ids, config.get('views')) for config in configs])
    return ids,y,groups,clean,x


def crossvalidate(x, y, groups, clean, method, folds=5):
    results = np.zeros((len(y), NUM_CLASSES))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=2026)
    for development, heldout in splitter.split(y, y, groups):
        development = development[clean[development]]
        assert not np.intersect1d(groups[development],groups[heldout]).size
        params = fit(method,x[:,development],y[development])
        results[heldout] = predict(method,params,x[:,heldout])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--models',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--method', choices=['equal','probability','logits','residual','residual2','rms','accuracy'],default='probability')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--freeze',action='store_true')
    modes.add_argument('--evaluate-frozen',type=Path)
    args = parser.parse_args()
    configs = json.loads(args.models.read_text())
    ids, labels, groups, clean, x = load(args.data,configs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    method = args.method
    names = [model['name'] for model in configs]
    if args.freeze:
        if args.data.stem != 'PublicTest':
            raise ValueError('Freeze the ensemble on PublicTest')
        params = fit(method,x[:,clean],labels[clean])
        result = {'method': method,'names':names,'params':params,'fit_split':args.data.stem}
    else:
        if args.evaluate_frozen:
            policy = json.loads(args.evaluate_frozen.read_text())
            if names != policy['names']:
                raise ValueError('Weights and prediction caches have different model order')
            method = policy['method']
            probabilities = predict(method,policy['params'],x)
            scope = 'frozen policy'
        else:
            if args.data.stem != 'PublicTest':
                raise ValueError('Use PublicTest for model development')
            probabilities = crossvalidate(x,labels,groups,clean,method)
            scope = 'five-fold grouped PublicTest development, each fold evaluated out of fold'
        result = measure(probabilities,labels,clean)
        result.update({'method':method,'names':names,'scope':scope,'split':args.data.stem})
        np.savez_compressed(args.output.with_suffix('.npz'),ids=ids,probabilities=probabilities)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
