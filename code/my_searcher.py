import logging
import my_cross_validation as mcv
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from hpbandster_sklearn import HpBandSterSearchCV

"""
Get Searcher
"""

def get_searcher(param_grid, model, search_method='grid', search_param={}, cv_method='stratified', cv_param={}, x=None, y=None):
  
    cv = mcv.get_cross_validation(cv_method, cv_param, x, y)

    if search_method=='grid':
        from sklearn.model_selection import GridSearchCV
        return GridSearchCV(model, param_grid, **search_param)
    elif search_method=='random':
        from sklearn.model_selection import RandomizedSearchCV
        return RandomizedSearchCV(model, param_grid, **search_param)
    elif search_method=='sv_grid':
        return HalvingGridSearchCV(model, param_grid, cv=cv, **search_param)
    elif search_method=='my_sv_grid':
        return MyHalvingGridSearchCV(model, param_grid, cv=cv, **search_param)
    elif search_method=='sv_random':
        from sklearn.experimental import enable_halving_search_cv
        from sklearn.model_selection import HalvingRandomSearchCV
        return HalvingRandomSearchCV(model, param_grid, cv=cv, **search_param)
    elif search_method=='hyperband':
        return HpBandSterSearchCV(model, param_grid, optimizer='hyperband', cv=cv, **search_param)
    elif search_method=='my_hyperband':
        return MyHpBandSterSearchCV(model, param_grid, optimizer='hyperband', cv=cv, **search_param)
    elif search_method=='bohb':
        #from hpbandster_sklearn import HpBandSterSearchCV
        return HpBandSterSearchCV(model, param_grid, optimizer='bohb', cv=cv, **search_param)
    elif search_method=='my_bohb':
        return MyHpBandSterSearchCV(model, param_grid, optimizer='bohb', cv=cv, **search_param)
    
        """
        elif search_method=='ours':
            from sklearn.experimental import enable_halving_search_cv
            from sklearn.model_selection import HalvingGridSearchCV

            return HalvingGridSearchCV(model, param_grid, **search_param)
        """
    else:
        logging.warning('Do not support such searcher ({}), please check your input!'.format(search_method))



"""
My HalvingGridSearchCV
"""
import warnings
import numpy as np
from scipy.stats import rankdata
from collections import defaultdict
from functools import partial
from numpy.ma import MaskedArray
from sklearn.model_selection._validation import _aggregate_score_dicts, _normalize_score_results
from sklearn.model_selection._search_successive_halving import _top_k,  _SubsampleMetaSplitter
from math import ceil, floor, log, tanh, atanh


class MyHalvingGridSearchCV(HalvingGridSearchCV):
    def __init__(self,
        estimator,
        param_grid,
        *,
        alpha=0.1,
        factor=3,
        resource="n_samples",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        cv=5,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=True,
        random_state=None,
        n_jobs=None,
        verbose=0,
    ):
        
        self.alpha = alpha

        super().__init__(
            estimator,
            param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            verbose=verbose,
            cv=cv,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
            max_resources=max_resources,
            resource=resource,
            factor=factor,
            min_resources=min_resources,
            aggressive_elimination=aggressive_elimination,
        )
    
    def _get_score(self, array_means, array_stds):
        a = 50
        b = 1
        y_max = (b-tanh(-2.5))*a
        y_min = (b-tanh(2.5))*a
        array_scores = array_means + self.alpha*(atanh(b-max(y_min,min(y_max,self.beta*100))/a)*2+5)*array_stds
        return array_scores

    """in sklearn.model_selection._search_successive_halving.BaseSuccessiveHalving"""
    def _run_search(self, evaluate_candidates):
        candidate_params = self._generate_candidate_params()

        if self.resource != "n_samples" and any(
            self.resource in candidate for candidate in candidate_params
        ):
            # Can only check this now since we need the candidates list
            raise ValueError(
                f"Cannot use parameter {self.resource} as the resource since "
                "it is part of the searched parameters."
            )

        # n_required_iterations is the number of iterations needed so that the
        # last iterations evaluates less than `factor` candidates.
        n_required_iterations = 1 + floor(log(len(candidate_params), self.factor))

        if self.min_resources == "exhaust":
            # To exhaust the resources, we want to start with the biggest
            # min_resources possible so that the last (required) iteration
            # uses as many resources as possible
            last_iteration = n_required_iterations - 1
            self.min_resources_ = max(
                self.min_resources_,
                self.max_resources_ // self.factor**last_iteration,
            )

        # n_possible_iterations is the number of iterations that we can
        # actually do starting from min_resources and without exceeding
        # max_resources. Depending on max_resources and the number of
        # candidates, this may be higher or smaller than
        # n_required_iterations.
        n_possible_iterations = 1 + floor(
            log(self.max_resources_ // self.min_resources_, self.factor)
        )

        if self.aggressive_elimination:
            n_iterations = n_required_iterations
        else:
            n_iterations = min(n_possible_iterations, n_required_iterations)

        if self.verbose:
            print(f"n_iterations: {n_iterations}")
            print(f"n_required_iterations: {n_required_iterations}")
            print(f"n_possible_iterations: {n_possible_iterations}")
            print(f"min_resources_: {self.min_resources_}")
            print(f"max_resources_: {self.max_resources_}")
            print(f"aggressive_elimination: {self.aggressive_elimination}")
            print(f"factor: {self.factor}")

        self.n_resources_ = []
        self.n_candidates_ = []

        for itr in range(n_iterations):
            power = itr  # default
            if self.aggressive_elimination:
                # this will set n_resources to the initial value (i.e. the
                # value of n_resources at the first iteration) for as many
                # iterations as needed (while candidates are being
                # eliminated), and then go on as usual.
                power = max(0, itr - n_required_iterations + n_possible_iterations)

            n_resources = int(self.factor**power * self.min_resources_)
            # guard, probably not needed
            n_resources = min(n_resources, self.max_resources_)
            self.n_resources_.append(n_resources)

            n_candidates = len(candidate_params)
            self.n_candidates_.append(n_candidates)

            if self.verbose:
                print("-" * 10)
                print(f"iter: {itr}")
                print(f"n_candidates: {n_candidates}")
                print(f"n_resources: {n_resources}")

            if self.resource == "n_samples":
                # subsampling will be done in cv.split()
                """"""
                self.beta = n_resources / self._n_samples_orig
                """"""
                cv = _SubsampleMetaSplitter(
                    base_cv=self._checked_cv_orig,
                    fraction=self.beta,
                    subsample_test=True,
                    random_state=self.random_state,
                )

            else:
                # Need copy so that the n_resources of next iteration does
                # not overwrite
                candidate_params = [c.copy() for c in candidate_params]
                for candidate in candidate_params:
                    candidate[self.resource] = n_resources
                cv = self._checked_cv_orig

            more_results = {
                "iter": [itr] * n_candidates,
                "n_resources": [n_resources] * n_candidates,
            }

            results = evaluate_candidates(
                candidate_params, cv, more_results=more_results
            )

            n_candidates_to_keep = ceil(n_candidates / self.factor)
            candidate_params = _top_k(results, n_candidates_to_keep, itr)

        self.n_remaining_candidates_ = len(candidate_params)
        self.n_required_iterations_ = n_required_iterations
        self.n_possible_iterations_ = n_possible_iterations
        self.n_iterations_ = n_iterations

    
    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)

        results = dict(more_results or {})
        for key, val in results.items():
            # each value is a list (as per evaluate_candidate's convention)
            # we convert it to an array for consistency with the other keys
            results[key] = np.asarray(val)

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_idx, key_name)] = array[:, split_idx]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(
                ~np.isfinite(array_means)
            ):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores "
                    f"are non-finite: {array_means}",
                    category=UserWarning,
                )

            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
                )
            )
            results["std_%s" % key_name] = array_stds


            """rewrite the evaluation metric""" 
            array_scores = self._get_score(array_means, array_stds)
            results["mean_%s" % key_name] = array_scores
            array_means = array_scores
            """"""

            if rank:
                # When the fit/scoring fails `array_means` contains NaNs, we
                # will exclude them from the ranking process and consider them
                # as tied with the worst performers.
                if np.isnan(array_means).all():
                    # All fit/scoring routines failed.
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means, method="min").astype(
                        np.int32, copy=False
                    )
                results["rank_%s" % key_name] = rank_result

        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        test_scores_dict = _normalize_score_results(out["test_scores"])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out["train_scores"])

        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores_dict[scorer_name],
                splits=True,
                rank=True,
                weights=None,
            )
            if self.return_train_score:
                _store(
                    "train_%s" % scorer_name,
                    train_scores_dict[scorer_name],
                    splits=True,
                )

        return results


"""
My HpBandSterSearchCV
"""
class MyHpBandSterSearchCV(HpBandSterSearchCV):

    def __init__(self,
        estimator,
        param_distributions,
        *,
        alpha=0.1,
        n_iter=10,
        optimizer="bohb",
        nameserver_host="127.0.0.1",
        nameserver_port=9090,
        min_budget=None,
        max_budget=None,
        resource_name=None,
        resource_type=None,
        cv=None,
        scoring=None,
        warm_start=True,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
        random_state=None,
        n_jobs=None,
        verbose=0,
        **kwargs,
    ):
        
        self.alpha = alpha
        self.current_iter = 0

        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            optimizer=optimizer,
            nameserver_host=nameserver_host,
            nameserver_port=nameserver_port,
            min_budget=min_budget,
            max_budget=max_budget,
            resource_name=resource_name,
            resource_type=resource_type,
            cv=cv,
            scoring=scoring,
            warm_start=warm_start,
            refit=refit,
            error_score=error_score,
            return_train_score=return_train_score,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )
    

    def _get_score(self, array_means, array_stds):
        a = 50
        b = 1
        y_max = (b-tanh(-2.5))*a
        y_min = (b-tanh(2.5))*a
        self.beta = self.n_resources_[self.current_iter]/self.n_resources_[-1]
        self.current_iter+=1
        #print('beta:', self.beta)

        array_scores = array_means + self.alpha*(atanh(b-max(y_min,min(y_max,self.beta*100))/a)*2+5)*array_stds
        return array_scores

    def _format_results(self, candidate_params, n_splits, out, more_results=None):
        n_candidates = len(candidate_params)
        out = _aggregate_score_dicts(out)

        results = dict(more_results or {})
        for key, val in results.items():
            # each value is a list (as per evaluate_candidate's convention)
            # we convert it to an array for consistency with the other keys
            results[key] = np.asarray(val)

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for split_idx in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s" % (split_idx, key_name)] = array[:, split_idx]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means

            if key_name.startswith(("train_", "test_")) and np.any(
                ~np.isfinite(array_means)
            ):
                warnings.warn(
                    f"One or more of the {key_name.split('_')[0]} scores "
                    f"are non-finite: {array_means}",
                    category=UserWarning,
                )

            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(
                np.average(
                    (array - array_means[:, np.newaxis]) ** 2, axis=1, weights=weights
                )
            )
            results["std_%s" % key_name] = array_stds


            """rewrite the evaluation metric""" 
            array_scores = self._get_score(array_means, array_stds)
            results["mean_%s" % key_name] = array_scores
            array_means = array_scores       
            """"""

            if rank:
                # When the fit/scoring fails `array_means` contains NaNs, we
                # will exclude them from the ranking process and consider them
                # as tied with the worst performers.
                if np.isnan(array_means).all():
                    # All fit/scoring routines failed.
                    rank_result = np.ones_like(array_means, dtype=np.int32)
                else:
                    min_array_means = np.nanmin(array_means) - 1
                    array_means = np.nan_to_num(array_means, nan=min_array_means)
                    rank_result = rankdata(-array_means, method="min").astype(
                        np.int32, copy=False
                    )
                results["rank_%s" % key_name] = rank_result

        _store("fit_time", out["fit_time"])
        _store("score_time", out["score_time"])
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_idx, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_idx] = value

        results.update(param_results)
        # Store a list of param dicts at the key 'params'
        results["params"] = candidate_params

        test_scores_dict = _normalize_score_results(out["test_scores"])
        if self.return_train_score:
            train_scores_dict = _normalize_score_results(out["train_scores"])

        for scorer_name in test_scores_dict:
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores_dict[scorer_name],
                splits=True,
                rank=True,
                weights=None,
            )
            if self.return_train_score:
                _store(
                    "train_%s" % scorer_name,
                    train_scores_dict[scorer_name],
                    splits=True,
                )

        return results
