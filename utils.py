import functools
import jax
import operator
import numpy as np

class bind(functools.partial):
    """
    An improved version of partial which accepts Ellipsis (...) as a placeholder
    """
    def __call__(self, *args, **keywords):
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)
    

def _sub(x, y):
    return jax.tree_util.tree_multimap(operator.sub, x, y)
    
def _add(x, y):
    return jax.tree_util.tree_multimap(operator.add, x, y)

def _multiply(x, y):
    return jax.tree_util.tree_multimap(operator.mul, x, y)


def get_class_indices(train_labels, samples_per_class, seed = 0, n_classes = 10):    
    np.random.seed(seed)
    combined_indices = []

    for c in range(n_classes):
        class_indices = np.where(train_labels.numpy() == c)[0]
        combined_indices.extend(class_indices[np.random.choice(len(class_indices), samples_per_class, replace = False)])

    return combined_indices