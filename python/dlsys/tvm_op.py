from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi

# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    b = tvm.const(const_k, dtype)
    C = tvm.compute(shape, lambda *i: A(*i) + b)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    b = tvm.const(const_k, dtype)
    C = tvm.compute(shape, lambda *i: A(*i) * b)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    zero = tvm.const(0, dtype)
    R = tvm.compute(shape, lambda *i: tvm.max(zero, A(*i)))

    s = tvm.create_schedule(R.op)
    f = tvm.build(s, [A, R], tgt, tgt_host, func_name)
    return f


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    inp = tvm.placeholder(shape, dtype, name="input")
    out_grad = tvm.placeholder(shape, dtype, name="input")

    inp_grad = tvm.compute(shape, lambda *i: tvm.select(inp(*i) > 0), out_grad(*i), tvm.const(0, dtype))

    s = tvm.create_schedule(inp_grad.op)
    f = tvm.build(s, [inp, out_grad], tgt, tgt_host, func_name)
    return f


def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.placeholder(shapeA, dtype, "A")
    B = tvm.placeholder(shapeB, dtype, "B")
    m, n = shapeA
    a, b = shapeB
    if transposeA:
        k = tvm.reduce_axis((0, m), name="k")
        if transposeB:
            assert m == b, "can't do matmul with %sx%s and %sx%s matrix" % (n, m, b, a)
            C = tvm.compute((n, a), lambda i,j: tvm.sum(A[k, i] * B[j, k], axis=k))
        else:
            assert m == a, "can't do matmul with %sx%s and %sx%s matrix" % (n, m, a, b)
            C = tvm.compute((n, b), lambda i,j: tvm.sum(A[k, i] * B[k, j], axis=k))
    else:
        k = tvm.reduce_axis((0, n), name="k")
        if transposeB:
            assert n == b, "can't do matmul with %sx%s and %sx%s matrix" % (m, n, b, a)
            C = tvm.compute((m, a), lambda i,j: tvm.sum(A[i, k] * B[j, k], axis=k))
        else:
            assert n == a, "can't do matmul with %sx%s and %sx%s matrix" % (m, n, a, b)
            C = tvm.compute((m, b), lambda i,j: tvm.sum(A[i, k] * B[k, j], axis=k))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, tgt_host, func_name)
    return f


def make_conv2d(shapeX, shapeF, padding, stride, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    outshape = N, M, int((H + padding - R + 1) / stride), int((W + padding - S + 1) / stride)

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    data = tvm.placeholder(shapeX, dtype, "data")
    filters = tvm.placeholder(shapeF, dtype, "filters")

    r = tvm.reduce_axis((0, R), "r")
    s = tvm.reduce_axis((0, S), "s")
    c = tvm.reduce_axis((0, C), "c")

    def conv_compute(n, m, h_, w_):
        x = h_ * stride - padding + r
        y = w_ * stride - padding + s
        return tvm.sum(
            tvm.select(
                tvm.any(
                    x < 0,
                    y < 0,
                    x >= H,
                    y >= W,
                ),
                tvm.const(0, dtype),              # padding
                data[n, c, x, y] * filters[m, c, r, s]
            ), axis = [c, r, s]
        )

    conv = tvm.compute(outshape, conv_compute)
    s = tvm.create_schedule(conv.op)
    f = tvm.build(s, [data, filters, conv], tgt, tgt_host, func_name)
    return f

def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):
    assert len(shape) == 2, "only a 2d tensor is accepted for batched softmax"
    num_batch, num_class = shape
    logits = tvm.placeholder(shape, dtype, "logits")
    k = tvm.reduce_axis((0, num_class), name="k")
    max_logits = tvm.compute((num_batch,), lambda i: tvm.max(logits[i, k], axis=k))
    logits_shifted = tvm.compute(shape, lambda i, j: logits[i, j] - max_logits[i])
    exps = tvm.compute(shape, lambda *i: tvm.exp(logits_shifted(*i)))
    k = tvm.reduce_axis((0, num_class), name="k")
    exps_sum = tvm.compute((num_batch,), lambda i: tvm.sum(exps[i, k], axis=k))
    softmaxes = tvm.compute(shape, lambda i, j: exps[i, j] / exps_sum[i])

    s = tvm.create_schedule(softmaxes.op)
    f = tvm.build(s, [logits, softmaxes], tgt, tgt_host, name=func_name)
    return f

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    assert len(shape) == 2, "only 2d tensor is accepted for batched softmax xent loss"
    num_batch, num_class = shape
    logits = tvm.placeholder(shape, dtype, "logits")
    truth = tvm.placeholder(shape, dtype, "truth")
    k = tvm.reduce_axis((0, num_class), name="k")
    max_logits = tvm.compute((num_batch,), lambda i: tvm.max(logits[i, k], axis=k))
    logits_shifted = tvm.compute(shape, lambda i, j: logits[i, j] - max_logits[i])
    exps = tvm.compute(shape, lambda *i: tvm.exp(logits_shifted(*i)))
    k = tvm.reduce_axis((0, num_class), name="k")
    exps_sum = tvm.compute((num_batch,), lambda i: tvm.sum(exps[i, k], axis=k))
    neg_pred_log = tvm.compute(shape, lambda i,j: tvm.log(exps_sum[i]) - logits_shifted[i, j])
    ewise_prod = tvm.compute(shape, lambda *i: truth(*i) * neg_pred_log(*i))

    i = tvm.reduce_axis((0, num_batch), name="i")
    j = tvm.reduce_axis((0, num_class), name="j")
    ce_sum = tvm.compute((1,), lambda _: tvm.sum(ewise_prod[i, j], axis=[i, j]))
    ce_mean = tvm.compute((1,), lambda _: ce_sum[0] / tvm.const(num_batch, dtype))

    s = tvm.create_schedule(ce_mean.op)
    f = tvm.build(s, [logits, truth, ce_mean], tgt, tgt_host, func_name)
    return f



def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f