from numpy.matlib import Infinity
import numpy as np
import pandas as pd

'''
    Y_DTYPE_C value
    unsigned int count
    unsigned int feature_idx
    X_DTYPE_C threshold
    unsigned int left
    unsigned int right
    Y_DTYPE_C gain
    unsigned int depth
    unsigned char is_leaf
    X_BINNED_DTYPE_C bin_threshold
    (0., 45, 97, -1.20557172, 1, 4, 2226097.70552018, 0, 0, 12, 0.)
    '''


def _predict_from_numeric_data_multi(nodes, X, out):
    for tmp_iter in range(X.shape[0]):
        out[tmp_iter] = _predict_one_from_numeric_data_multi(nodes, X, tmp_iter)


def _predict_one_from_numeric_data_multi(nodes, numeric_data, row):
    node = nodes[0]
    while True:
        if node['is_leaf']:
            return node['residual']
        if numeric_data[row, node['feature_idx']] == Infinity:
            # if data is +inf we always go to the right child, even when the
            # threshold is +inf
            node = nodes[node['right']]
        else:
            if numeric_data[row, node['feature_idx']] <= node['threshold']:
                node = nodes[node['left']]
            else:
                node = nodes[node['right']]


def _predict_from_binned_data_multi(nodes, binned_data, out):
    for i in range(binned_data.shape[0]):
        out[i] = _predict_one_from_binned_data_multi(nodes, binned_data, i)


def _predict_one_from_binned_data_multi(nodes, binned_data, row):
    node = nodes[0]
    while True:
        if node['is_leaf']:
            return node['residual']
        if binned_data[row, node['feature_idx']] <= node['bin_threshold']:
            node = nodes[node['left']]
        else:
            node = nodes[node['right']]


def _compute_partial_dependence_multi(nodes, X, target_features, out):
    """Partial dependence of the response on the ``target_features`` set.

    For each sample in ``X`` a tree traversal is performed.
    Each traversal starts from the root with weight 1.0.

    At each non-leaf node that splits on a target feature, either
    the left child or the right child is visited based on the feature
    value of the current sample, and the weight is not modified.
    At each non-leaf node that splits on a complementary feature,
    both children are visited and the weight is multiplied by the fraction
    of training samples which went to each child.

    At each leaf, the value of the node is multiplied by the current
    weight (weights sum to 1 for all visited terminal nodes).

    Parameters
    ----------
    nodes : view on array of PREDICTOR_RECORD_DTYPE, shape (n_nodes)
        The array representing the predictor tree.
    X : view on 2d ndarray, shape (n_samples, n_target_features)
        The grid points on which the partial dependence should be
        evaluated.
    target_features : view on 1d ndarray, shape (n_target_features)
        The set of target features for which the partial dependence
        should be evaluated.
    out : view on 1d ndarray, shape (n_samples)
        The value of the partial dependence function on each grid
        point.
    """

    node_idx_stack = np.zeros(shape=nodes.shape[0],dtype=np.uint32)
    weight_stack = np.zeros(shape=nodes.shape[0],dtype=np.float64)

        # node_struct * current_node  # pointer to avoid copying attributes

    res_row = np.zeros(shape=nodes.residual.shape[0],dtype=np.float64)

    for sample_idx in range(X.shape[0]):
        # init stacks for current sample
        stack_size = 1
        node_idx_stack[0] = 0  # root node
        weight_stack[0] = 1  # all the samples are in the root node
        total_weight = 0

        while stack_size > 0:

            # pop the stack
            stack_size -= 1
            current_node_idx = node_idx_stack[stack_size]
            current_node = nodes[current_node_idx]

            if current_node.is_leaf:
                res_row += list(weight_stack[stack_size] *
                                      np.asarray(current_node.residual))
                out[sample_idx] = res_row
                total_weight += weight_stack[stack_size]
            else:
                # determine if the split feature is a target feature
                is_target_feature = False
                for feature_idx in range(target_features.shape[0]):
                    if target_features[feature_idx] == current_node.feature_idx:
                        is_target_feature = True
                        break

                if is_target_feature:
                    # In this case, we push left or right child on stack
                    if X[sample_idx, feature_idx] <= current_node.threshold:
                        node_idx_stack[stack_size] = current_node.left
                    else:
                        node_idx_stack[stack_size] = current_node.right
                    stack_size += 1
                else:
                    # In this case, we push both children onto the stack,
                    # and give a weight proportional to the number of
                    # samples going through each branch.

                    # push left child
                    node_idx_stack[stack_size] = current_node.left
                    left_sample_frac = (nodes[current_node.left].count / current_node.count)
                    current_weight = weight_stack[stack_size]
                    weight_stack[stack_size] = current_weight * left_sample_frac
                    stack_size += 1

                    # push right child
                    node_idx_stack[stack_size] = current_node.right
                    weight_stack[stack_size] = (
                        current_weight * (1 - left_sample_frac))
                    stack_size += 1

        # Sanity check. Should never happen.
        if not (0.999 < total_weight < 1.001):
            raise ValueError("Total weight should be 1.0 but was %.9f" % total_weight)
