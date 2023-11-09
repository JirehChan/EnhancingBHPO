import logging
from sklearn.model_selection import StratifiedKFold, KFold

"""
Get CVs
"""

def get_cross_validation(cv_method='stratified', cv_param={}, x=None, y=None):
  
    if cv_method=='ours':      
        return GSKFold(**cv_param).initial(x,y)
    elif cv_method=='new':
        return NewFold(**cv_param).initial(x,y)
    elif cv_method=='stratified':
        return StratifiedKFold(**cv_param)
    elif cv_method=='random':
        return KFold(**cv_param)
    elif cv_method=='original':
        return OrignialKFold(**cv_param)
    else:
        logging.warning('Do not support such type of cross validation ({}), please check your input!'.format(search_method))
        return None



"""
New CV
"""
import numpy as np
import time
import warnings
from collections import Counter
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, _num_samples, column_or_1d
from sklearn.utils.multiclass import type_of_target

class NewFold(StratifiedKFold):
    def __init__(self, 
                 n_general=5, n_special=0, 
                 cluster_type='kmeans', cluster_param={}, 
                 block_ratio=0.8, main_ratio=0.8, *, 
                 shuffle=False, random_state=None):
        self.n_general = n_general
        self.n_special = n_special
        self.cluster_type  = cluster_type
        self.cluster_param = cluster_param
        self.n_splits = n_general + n_special
        self.block_ratio = block_ratio
        self.main_ratio = main_ratio
        super().__init__(n_splits=self.n_splits, shuffle= shuffle, random_state=random_state)
    
    """
    Initial
    """
    def initial(self, X, y):
        self._generate_blocks(X, y)
        return self
    
    """
    Generate Blocks
    """
    def _generate_blocks(self, X, y):
        # Get the Classes
        self.n_class = len(dict(Counter(y)))
        self.y = y
        if self.n_special < 1.5:
            _n_special = min(self.n_class, 5)
        else:
            _n_special = self.n_special

        # Calculate the Clusters
        if self.cluster_type == 'kmeans':
            from sklearn.cluster import KMeans
            cluster = KMeans(n_clusters=_n_special, **self.cluster_param)

        n_instance = len(self.y)

        is_outlier = True
        outlier_ind = []
        all_ind = list(range(n_instance))

        while is_outlier:
            cluster = cluster.fit(X[all_ind])
            self.x = np.array(cluster.predict(X))

            is_outlier = False

            for i in range(_n_special):
                current_ind = np.where(self.x==i)[0]

                if len(current_ind)<(n_instance/_n_special)*self.block_ratio:
                    is_outlier = True
                    outlier_ind.extend(current_ind)
                    all_ind = list(set(all_ind)-set(current_ind))
            
            if len(all_ind) < (n_instance*0.5): 
                #self.n_special:
                self.block_ratio = self.block_ratio - 0.01

                if self.block_ratio<0.5:
                    logging.warning('cannnot construct the blocks!')
                    return None
                
                logging.warning('block_ratio --> {:.2f}'.format(self.block_ratio))

                is_outlier = True
                outlier_ind = []
                all_ind = list(range(n_instance))
                continue

        # Mix Category
        allo_blocks = {i:[] for i in range(_n_special)}

        ## count number
        counts = np.zeros((self.n_class, _n_special))
        indexs = {}
        for i in range(self.n_class):
            for j in range(_n_special):
                indexs['{}:{}'.format(i,j)] = []
        index_ing = list(indexs.keys())

        for i, (c_x, c_y) in enumerate(zip(self.x, self.y)):
            counts[int(c_y-1), c_x] += 1
            indexs['{}:{}'.format(int(c_y-1), c_x)].append(i)

        self.blocks = np.zeros(len(self.y), dtype=int)
        ## s1. allo. instances in clusters
        for j in range(self.n_special):
            class_to_clsuter = np.argsort(counts[:,j])[::-1]
            for i in class_to_clsuter:
                if len(allo_blocks[j]) > len(self.y)/self.n_special:
                    break
                allo_blocks[j].extend(indexs['{}:{}'.format(i,j)])
                self.blocks[indexs['{}:{}'.format(i,j)]] = j+1
                index_ing.remove('{}:{}'.format(i,j))

        ## s2. allo. remaining instances
        for i in range(self.n_class):
            cluster_to_class = np.argmax(counts[i,:])
            for j in range(_n_special):
                if '{}:{}'.format(i,j) in index_ing:
                    allo_blocks[cluster_to_class].extend(indexs['{}:{}'.format(i,j)])
                    self.blocks[indexs['{}:{}'.format(i,j)]] = cluster_to_class+1
                    index_ing.remove('{}:{}'.format(i,j))
        
        return 
    
    """
    Split
    """
    def split(self, X, y, groups=None):
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)
    
    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i
    
    def _make_test_folds(self, X, y=None):
        
        rng = check_random_state(self.random_state)
        
        # use the block labels to construct folds 
        y = self.blocks
        
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )
        
        y = column_or_1d(y)

        y_org, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        
        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )
        
        # count blocks
        label_blocks = {i:[] for i in class_perm}
        for i, ins in enumerate(y_encoded):
            label_blocks[ins].append(i)

        if self.shuffle:
            for i in class_perm:
                rng.shuffle(label_blocks[i])
        
        # Generate General Folds
        test_folds = np.zeros(len(y), dtype=int)
        n_fold_ins = [round(len(label_blocks[i])/self.n_splits) for i in class_perm]

        for j in range(self.n_general):
            for i, n in zip(class_perm, n_fold_ins):
                test_folds[label_blocks[i][:n]] = j+self.n_special
                label_blocks[i] = label_blocks[i][n:]
        
        # Generate Special Folds
        main_ratio = self.main_ratio
        other_ratio = (1-main_ratio)/(self.n_special-1)
        n_fold_ins = [len(label_blocks[i]) for i in class_perm] 
        
        for j in sorted(class_perm)[:-1]: # for folds
            for i in class_perm:
                if i==j:
                    num = round(n_fold_ins[i]*main_ratio)
                else:
                    num = round(n_fold_ins[i]*other_ratio)
                
                test_folds[label_blocks[i][:num]] = j+1
                label_blocks[i] = label_blocks[i][num:]

        return test_folds




"""
Our Researcher
"""

class GSKFold(StratifiedKFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None, 
                 gs_mode='g', gs_base='y', gs_ratio=0.5, gs_special_alpha=0.5,
                 gs_cluster='kmeans', gs_cluster_param={}):
        self.gs_mode = gs_mode
        self.gs_base = gs_base
        self.gs_alpha = gs_special_alpha
        self.gs_cluster = gs_cluster
        self.gs_cluster_param = gs_cluster_param
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    
    """----------------"""
    def initial(self, X, y):
        start_time = time.time()
        
        if self.gs_base == 'y':
            self.clusters = y
        elif self.gs_base == 'x':
            if self.gs_cluster=='kmeans':
                from sklearn.cluster import KMeans
                cluster = KMeans(**self.gs_cluster_param)
                cluster = cluster.fit(X)
                self.clusters = cluster.predict(X)
            else:
                raise ValueError(
                    "Do not support {} cluster type, please check your input.".format(
                        self.clustertype))
        elif self.gs_base == 'xy':
            if self.gs_cluster=='kmeans':
                from sklearn.cluster import KMeans
                cluster = KMeans(**self.gs_cluster_param)
                cluster = cluster.fit(X)
                self.clusters = cluster.predict(X)*1000+y
            else:
                raise ValueError(
                    "Do not support {} cluster type, please check your input.".format(
                        self.clustertype))
        else:
            raise ValueError(
                    "Do not support {} GS Base, please check your input.".format(
                        self.gs_base))
    """----------------"""
    
    def split(self, X, y, groups=None):
        y = check_array(y, input_name="y", ensure_2d=False, dtype=None)
        return super().split(X, y, groups)
        
    
    
    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        for i in range(self.n_splits):
            yield test_folds == i
    
    def _make_test_folds(self, X, y=None):
        rng = check_random_state(self.random_state)
        
        y = self.clusters
        
        y = np.asarray(y)
        type_of_target_y = type_of_target(y)
        
        allowed_target_types = ("binary", "multiclass")
        if type_of_target_y not in allowed_target_types:
            raise ValueError(
                "Supported target types are: {}. Got {!r} instead.".format(
                    allowed_target_types, type_of_target_y
                )
            )
        
        y = column_or_1d(y)

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]
        
        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)
        min_groups = np.min(y_counts)
        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "n_splits=%d cannot be greater than the"
                " number of members in each class." % (self.n_splits)
            )
        if self.n_splits > min_groups:
            warnings.warn(
                "The least populated class in y has only %d"
                " members, which is less than n_splits=%d."
                % (min_groups, self.n_splits),
                UserWarning,
            )
        
        if self.gs_mode == 'g':
            # general
            # Determine the optimal number of samples from each class in each fold,
            # using round robin over the sorted y. (This can be done direct from
            # counts, but that code is unreadable.)
            y_order = np.sort(y_encoded)
            allocation = np.asarray(
                [
                    np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                    for i in range(self.n_splits)
                ]
            )


            # To maintain the data order dependencies as best as possible within
            # the stratification constraint, we assign samples from each class in
            # blocks (and then mess that up when shuffle=True).
            test_folds = np.empty(len(y), dtype="i")
            for k in range(n_classes):
                # since the kth column of allocation stores the number of samples
                # of class k in each test set, this generates blocks of fold
                # indices corresponding to the allocation for class k.
                folds_for_class = np.arange(self.n_splits).repeat(allocation[:, k])
                if self.shuffle:
                    rng.shuffle(folds_for_class)
                test_folds[y_encoded == k] = folds_for_class
            
            return test_folds
        
        elif self.gs_mode == 's':
            # special
            alpha = self.gs_alpha
            beta = (1-alpha)/(n_classes-1)

            y_order = np.sort(y_encoded)
            y_nums = np.bincount(y_order)
            allocation = []

            for i in range(n_classes-1):
                allocation.append(
                    [int(y_nums[j]*alpha) if j==i else int(y_nums[j]*beta)
                        for j in range(n_classes)])
            allocation.append(list(y_nums-np.asarray(allocation).sum(axis=0)))
            allocation = np.asarray(allocation)

            test_folds = np.empty(len(y), dtype="i")
            for k in range(n_classes):
                folds_for_class = np.arange(n_classes).repeat(allocation[:, k])

                if self.shuffle:
                    rng.shuffle(folds_for_class)

                test_folds[y_encoded == k] = folds_for_class
            
            return test_folds
        
        elif self.gs_mode == 'gs':
            for i in range(n_classes):
                a_where = np.where(y_encoded==i)[0]
                y_encoded[np.random.choice(a_where, int(len(a_where)/2))]=i+1000
            
            # general
            y_where = np.where(y_encoded<n_classes)[0]
            y_order = np.sort(y_encoded[y_where])
            
            allocation = np.asarray(
                [
                    np.bincount(y_order[i :: self.n_splits-n_classes], minlength=n_classes)
                    for i in range(self.n_splits-n_classes)
                ]
            )

            test_folds = np.empty(len(y), dtype="i")
            
            for k in range(n_classes):
                folds_for_class = np.arange(self.n_splits-n_classes).repeat(allocation[:, k])
                if self.shuffle:
                    rng.shuffle(folds_for_class)
                
                test_folds[y_encoded == k] = folds_for_class
                
            # special
            alpha = self.gs_alpha
            beta = (1-alpha)/(n_classes-1)
            
            y_where = np.where(y_encoded+0.1>n_classes)[0]
            y_order = np.sort(y_encoded[y_where])-1000
            y_nums = np.bincount(y_order)
            allocation = []

            for i in range(n_classes-1):
                allocation.append(
                    [int(y_nums[j]*alpha) if j==i else int(y_nums[j]*beta)
                        for j in range(n_classes)])
            allocation.append(list(y_nums-np.asarray(allocation).sum(axis=0)))
            allocation = np.asarray(allocation)

            #test_folds = np.empty(len(y), dtype="i")
            for k in range(n_classes):
                folds_for_class = np.arange(n_classes).repeat(allocation[:, k])+self.n_splits-n_classes

                if self.shuffle:
                    rng.shuffle(folds_for_class)

                test_folds[y_encoded == k+1000] = folds_for_class

            return test_folds


"""
Original Researcher
"""
class OrignialKFold(StratifiedKFold):
    
    def _iter_test_masks(self, X, y=None, groups=None):
        test_folds = self._make_test_folds(X, y)
        print(len(y))
        print(len(test_folds))
        print(test_folds)
        for i in range(self.n_splits):
            yield test_folds == i

        import sys
        sys.exit()
