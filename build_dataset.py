"""
Step 1 - build the dataset for teaching Qwen to translate TensorFlow 1.x code
(the style used in the well-cost project: placeholders, tf.Variable,
tf.Session, feed_dict, AdamOptimizer...) into PyTorch.

Every example is a pair:   TensorFlow snippet  ->  PyTorch snippet.

Output (folder data/):
    train.jsonl   used for training
    val.jsonl     used for validation during training (never trained on)
    test.jsonl    used ONLY for the final comparison. It has three groups:
        seen_patterns    same kinds of snippet as training, but new names/sizes
        unseen_patterns  kinds of snippet that never appear in training
        real_code        pieces copied from the well-cost project itself

Run:   ../llm_env/bin/python build_dataset.py
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from exec_check import run_snippet

SYSTEM = ("You convert TensorFlow 1.x code to PyTorch. Keep the same variable "
          "names. Reply with only the PyTorch code.")
MAX_TOKENS = 512
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
rng = random.Random(3407)
pick = rng.choice

# ---------------------------------------------------------------- name pools
XY = [("xs", "ys"), ("x", "y"), ("inputs", "targets"), ("x_in", "y_true"), ("X", "Y")]
DATA = [("x_data", "y_data"), ("train_x", "train_y"), ("X_train", "y_train"), ("features", "labels")]
WB = [("Weights_1", "biases_1", "Weights_2", "biases_2"), ("W1", "b1", "W2", "b2"),
      ("w_hidden", "b_hidden", "w_out", "b_out"), ("weights1", "bias1", "weights2", "bias2")]
SRC = [("w1", "b1", "w2", "b2"), ("ww1", "bb1", "ww2", "bb2"),
       ("W1_train", "B1_train", "W2_train", "B2_train"), ("init_w1", "init_b1", "init_w2", "init_b2")]
HID = ["l1", "hidden", "h1", "layer1"]
PRED = ["prediction", "pred", "output", "y_pred"]
SESS = ["sess", "model", "session"]
LOSS = ["loss", "cost"]
ERR = ["error", "err", "final_loss", "train_error"]
ACT = [("tf.nn.relu", "torch.relu"), ("tf.nn.tanh", "torch.tanh"), ("tf.tanh", "torch.tanh"),
       ("tf.nn.sigmoid", "torch.sigmoid"), ("tf.sigmoid", "torch.sigmoid")]
LR_VARS = ["Learning_rate", "learning_rate", "lr"]
STEP_VARS = ["Train_itteration", "Object_itteration", "n_steps", "epochs"]

ADD_LAYER_PT = '''def add_layer(inputs, Weights, biases, activation_function=None):
    Wx_plus_b = torch.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
'''


def distinct(*names):
    return len(set(names)) == len(names)


def loss_form(Y, P, which=None):
    forms = [
        (f"tf.reduce_mean(tf.reduce_sum(tf.square({Y} - {P}), reduction_indices=[1]))",
         f"torch.mean(torch.sum(torch.square({Y} - {P}), dim=1))"),
        (f"tf.reduce_mean(tf.reduce_sum(tf.square({Y} - {P}), axis=1))",
         f"torch.mean(torch.sum(torch.square({Y} - {P}), dim=1))"),
        (f"tf.reduce_mean(tf.square({Y} - {P}))", f"torch.mean(torch.square({Y} - {P}))"),
        (f"tf.reduce_mean(tf.abs({Y} - {P}))", f"torch.mean(torch.abs({Y} - {P}))"),
        (f"tf.losses.mean_squared_error({Y}, {P})", f"torch.nn.functional.mse_loss({P}, {Y})"),
        (f"tf.reduce_sum(tf.square({Y} - {P}))", f"torch.sum(torch.square({Y} - {P}))"),
        (f"tf.sqrt(tf.reduce_mean(tf.square({Y} - {P})))", f"torch.sqrt(torch.mean(torch.square({Y} - {P})))"),
    ]
    return forms[which] if which is not None else pick(forms)


def optimizer_form(lr, params):
    forms = [
        (f"tf.train.AdamOptimizer({lr})", f"torch.optim.Adam([{params}], lr={lr})"),
        (f"tf.train.AdamOptimizer({lr})", f"torch.optim.Adam([{params}], lr={lr})"),
        (f"tf.train.GradientDescentOptimizer({lr})", f"torch.optim.SGD([{params}], lr={lr})"),
        (f"tf.train.MomentumOptimizer({lr}, momentum=0.9)", f"torch.optim.SGD([{params}], lr={lr}, momentum=0.9)"),
        (f"tf.train.RMSPropOptimizer({lr})", f"torch.optim.RMSprop([{params}], lr={lr}, alpha=0.9, eps=1e-10)"),
    ]
    return pick(forms)


def variable_form(name, src, style):
    if style == "mat":
        return (f"{name} = tf.Variable(tf.cast(np.mat({src}), dtype=tf.float32), name='{name}')",
                f"{name} = torch.tensor(np.atleast_2d({src}), dtype=torch.float32, requires_grad=True)")
    if style == "array":
        return (f"{name} = tf.Variable(tf.cast(np.array({src}), dtype=tf.float32), name='{name}')",
                f"{name} = torch.tensor(np.array({src}), dtype=torch.float32, requires_grad=True)")
    return (f"{name} = tf.Variable({src}, dtype=tf.float32, name='{name}')",
            f"{name} = torch.tensor({src}, dtype=torch.float32, requires_grad=True)")


def placeholder(name, n):
    return pick([f"{name} = tf.placeholder(shape=[None, {n}], dtype=tf.float32)",
                 f"{name} = tf.placeholder(tf.float32, [None, {n}])",
                 f"{name} = tf.placeholder(tf.float32, shape=[None, {n}])"])


def weight_setup(names, n_in, n_h, n_out, as_tensor):
    a, b, c, d = names
    if as_tensor:
        return (f"{a} = torch.tensor(rs.rand({n_in}, {n_h}) - 0.5, dtype=torch.float32, requires_grad=True)\n"
                f"{b} = torch.tensor(rs.rand(1, {n_h}) * 0.1, dtype=torch.float32, requires_grad=True)\n"
                f"{c} = torch.tensor(rs.rand({n_h}, {n_out}) - 0.5, dtype=torch.float32, requires_grad=True)\n"
                f"{d} = torch.tensor(rs.rand(1, {n_out}) * 0.1, dtype=torch.float32, requires_grad=True)\n")
    return (f"{a} = (rs.rand({n_in}, {n_h}) - 0.5).tolist()\n"
            f"{b} = (rs.rand({n_h}) * 0.1).tolist()\n"
            f"{c} = (rs.rand({n_h}, {n_out}) - 0.5).tolist()\n"
            f"{d} = (rs.rand({n_out}) * 0.1).tolist()\n")


def sizes():
    return pick([2, 3, 4, 5, 6, 6, 8]), pick([4, 5, 8, 10, 16, 20, 20]), pick([1, 1, 1, 2])


# ------------------------------------------------------- training templates
def t_setup_lines():
    tf_imp = pick(["import tensorflow as tf", "import tensorflow.compat.v1 as tf",
                   "import tensorflow as tf\nimport tensorflow.compat.v1 as tf"])
    keep = rng.sample(["import numpy as np", "import csv", "import os", "from sklearn import preprocessing",
                       "import matplotlib.pyplot as plt", "import pandas as pd"], rng.randint(1, 3))
    drop = rng.sample(["tf.disable_v2_behavior()", "tf.disable_eager_execution()", "tf.reset_default_graph()",
                       "tf.compat.v1.reset_default_graph()", "tf.compat.v1.disable_v2_behavior()"], rng.randint(0, 2))
    pos = rng.randint(0, len(keep))
    tf_lines = keep[:pos] + [tf_imp] + keep[pos:] + drop
    pt_lines = keep[:pos] + ["import torch"] + keep[pos:]
    return dict(tf="\n".join(tf_lines), pt="\n".join(pt_lines))


def t_session_lines():
    s = pick(SESS)
    lines = rng.sample(["tf.reset_default_graph()", "tf.disable_v2_behavior()"], rng.randint(0, 2))
    lines += [pick(["init = tf.global_variables_initializer()", "init = tf.initialize_all_variables()"])]
    lines += [f"{s} = tf.Session()", f"{s}.run(init)"]
    return dict(tf="\n".join(lines),
                pt="# Not needed in PyTorch: tensors are ready as soon as they are created.")


def t_variables_np():
    while True:
        names, src = pick(WB), pick(SRC)
        if distinct(*names, *src):
            break
    n_in, n_h, n_out = sizes()
    style = pick(["mat", "mat", "array", "direct"])
    k = rng.randint(2, 4)
    pairs = [variable_form(n, s, style) for n, s in list(zip(names, src))[:k]]
    return dict(tf="\n".join(p[0] for p in pairs), pt="\n".join(p[1] for p in pairs),
                setup=weight_setup(src, n_in, n_h, n_out, False), outs=list(names[:k]))


def t_variables_init():
    names = pick(WB)
    n_in, n_h, n_out = sizes()
    std = pick([0.1, 0.01, 0.5])
    w_forms = [
        ("tf.random_normal([{a}, {b}])", "torch.randn({a}, {b}, requires_grad=True)"),
        ("tf.random_normal([{a}, {b}], stddev={s})", "(torch.randn({a}, {b}) * {s}).requires_grad_()"),
        ("tf.random_uniform([{a}, {b}])", "torch.rand({a}, {b}, requires_grad=True)"),
    ]
    b_forms = [
        ("tf.zeros([1, {b}])", "torch.zeros(1, {b}, requires_grad=True)"),
        ("tf.zeros([1, {b}]) + 0.1", "torch.full((1, {b}), 0.1, requires_grad=True)"),
        ("tf.ones([1, {b}])", "torch.ones(1, {b}, requires_grad=True)"),
    ]
    wf, bf = pick(w_forms), pick(b_forms)
    named = pick([True, False])
    shapes = [(n_in, n_h, wf), (1, n_h, bf), (n_h, n_out, wf), (1, n_out, bf)]
    tf_l, pt_l = [], []
    k = rng.randint(2, 4)
    for name, (a, b, form) in list(zip(names, shapes))[:k]:
        nm = f", name='{name}'" if named else ""
        tf_l.append(f"{name} = tf.Variable({form[0].format(a=a, b=b, s=std)}{nm})")
        pt_l.append(f"{name} = {form[1].format(a=a, b=b, s=std)}")
    return dict(tf="\n".join(tf_l), pt="\n".join(pt_l), setup="", outs=list(names[:k]), mode="shape")


def t_add_layer():
    fn = pick(["add_layer", "add_layer", "dense_layer", "layer"])
    i, w, b, act = pick([("inputs", "Weights", "biases", "activation_function"), ("x", "W", "b", "act"),
                         ("inputs", "weights", "bias", "activation")])
    comment = pick(["    # add one more layer and return the output of this layer\n", ""])
    if pick([True, False]):
        body = ("    Wx_plus_b = {mm}({i}, {w}) + {b}\n    if {act} is None:\n        outputs = Wx_plus_b\n"
                "    else:\n        outputs = {act}(Wx_plus_b)\n    return outputs")
    else:
        body = "    z = {mm}({i}, {w}) + {b}\n    return z if {act} is None else {act}(z)"
    head = f"def {fn}({i}, {w}, {b}, {act}=None):\n{comment}"
    return dict(tf=head + body.format(mm="tf.matmul", i=i, w=w, b=b, act=act),
                pt=head + body.format(mm="torch.matmul", i=i, w=w, b=b, act=act),
                setup="_x = torch.tensor(rs.rand(5, 3), dtype=torch.float32)\n"
                      "_W = torch.tensor(rs.rand(3, 4), dtype=torch.float32)\n"
                      "_b = torch.tensor(rs.rand(1, 4), dtype=torch.float32)\n",
                post=f"_r1 = {fn}(_x, _W, _b, torch.tanh)\n_r2 = {fn}(_x, _W, _b)\n", outs=["_r1", "_r2"])


def t_forward(hidden_layers=1):
    while True:
        (X, _), (xd, _), names = pick(XY), pick(DATA), pick(WB)
        h, p, s = pick(HID), pick(PRED), pick(SESS)
        out = pick(["prediction_value", "out", "result", "y_hat"])
        if distinct(X, xd, *names, h, p, s, out):
            break
    n_in, n_h, n_out = sizes()
    w1, b1, w2, b2 = names
    atf, apt = pick(ACT)
    use_fn = pick([True, False])
    setup = f"{xd} = rs.rand(7, {n_in})\n" + weight_setup(names, n_in, n_h, n_out, True)
    if use_fn:
        setup += ADD_LAYER_PT
        layers = (f"{h} = add_layer({X}, {w1}, {b1}, activation_function=%s)\n"
                  f"{p} = add_layer({h}, {w2}, {b2}, activation_function=None)\n")
    else:
        layers = (f"{h} = %s(MM({X}, {w1}) + {b1})\n"
                  f"{p} = MM({h}, {w2}) + {b2}\n")
    tf_code = (placeholder(X, n_in) + "\n" + (layers % atf).replace("MM", "tf.matmul")
               + f"{out} = {s}.run({p}, feed_dict={{{X}: {xd}}})")
    pt_code = (f"{X} = torch.tensor({xd}, dtype=torch.float32)\n" + (layers % apt).replace("MM", "torch.matmul")
               + f"{out} = {p}.detach().numpy()")
    return dict(tf=tf_code, pt=pt_code, setup=setup, outs=[out])


def t_loss():
    Y, P = pick([("ys", "prediction"), ("y", "pred"), ("targets", "output"), ("y_true", "y_pred"), ("Y", "P")])
    n_out = pick([1, 1, 2, 3])
    setup = (f"{Y} = torch.tensor(rs.rand(9, {n_out}), dtype=torch.float32)\n"
             f"{P} = torch.tensor(rs.rand(9, {n_out}), dtype=torch.float32)\n")
    kind = rng.random()
    if kind < 0.6:
        name = pick(LOSS + ["mse", "error"])
        ltf, lpt = loss_form(Y, P)
        return dict(tf=f"{name} = {ltf}", pt=f"{name} = {lpt}", setup=setup, outs=[name])
    if kind < 0.8:
        return dict(tf=f"rmse = tf.sqrt(tf.reduce_mean(tf.square({Y} - {P})))\nmae = tf.reduce_mean(tf.abs({Y} - {P}))",
                    pt=f"rmse = torch.sqrt(torch.mean(torch.square({Y} - {P})))\nmae = torch.mean(torch.abs({Y} - {P}))",
                    setup=setup, outs=["rmse", "mae"])
    return dict(tf=f"ss_res = tf.reduce_sum(tf.square({Y} - {P}))\n"
                   f"ss_tot = tf.reduce_sum(tf.square({Y} - tf.reduce_mean({Y})))\nr2 = 1 - ss_res / ss_tot",
                pt=f"ss_res = torch.sum(torch.square({Y} - {P}))\n"
                   f"ss_tot = torch.sum(torch.square({Y} - torch.mean({Y})))\nr2 = 1 - ss_res / ss_tot",
                setup=setup, outs=["r2"])


def t_optimizer():
    names = pick(WB)
    k = pick([2, 4, 4])
    params = ", ".join(names[:k])
    lr = pick([0.1, 0.01, 0.001, 0.05, 0.0005] + LR_VARS)
    loss = pick(LOSS)
    otf, opt = optimizer_form(lr, params)
    if pick([True, False]):
        tf_code = f"train_step = {otf}.minimize({loss}, var_list=[{params}])"
    else:
        tf_code = f"optimizer = {otf}\ntrain_step = optimizer.minimize({loss}, var_list=[{params}])"
    setup = weight_setup(names, 3, 4, 1, True) + (f"{lr} = 0.02\n" if isinstance(lr, str) else "")
    return dict(tf=tf_code, pt=f"optimizer = {opt}", setup=setup,
                post="_res = [type(optimizer).__name__, optimizer.param_groups[0]['lr'], "
                     "len(optimizer.param_groups[0]['params'])]\n", outs=["_res"])


def t_full_train():
    while True:
        (X, Y), (xd, yd), names, src = pick(XY), pick(DATA), pick(WB), pick(SRC)
        h, p, s, loss, err = pick(HID), pick(PRED), pick(SESS), pick(LOSS), pick(ERR)
        lr = pick([0.1, 0.01, 0.001, 0.05] + LR_VARS)
        steps = pick([20, 50, 100] + STEP_VARS)
        if distinct(X, Y, xd, yd, *names, h, p, s, loss, err, str(lr), str(steps)):
            break
    n_in, n_h, n_out = sizes()
    w1, b1, w2, b2 = names
    atf, apt = pick(ACT)
    use_fn = rng.random() < 0.3
    in_loop = pick([True, False])
    ltf, lpt = loss_form(Y, p)
    otf, opt = optimizer_form(lr, ", ".join(names))
    # The weights already exist (creating them is taught by the variables_* examples),
    # which keeps a whole training program inside the 512-token window.
    feed = f"feed_dict={{{X}: {xd}, {Y}: {yd}}}"

    setup = (f"{xd} = rs.rand(12, {n_in})\n{yd} = rs.rand(12, {n_out})\n"
             + weight_setup(names, n_in, n_h, n_out, True)
             + (f"{lr} = 0.02\n" if isinstance(lr, str) else "")
             + (f"{steps} = 30\n" if isinstance(steps, str) else "")
             + (ADD_LAYER_PT if use_fn else ""))
    if use_fn:
        tf_layers = (f"{h} = add_layer({X}, {w1}, {b1}, activation_function={atf})\n"
                     f"{p} = add_layer({h}, {w2}, {b2}, activation_function=None)\n")
        pt_fwd = (f"    {h} = add_layer(inp, {w1}, {b1}, activation_function={apt})\n"
                  f"    return add_layer({h}, {w2}, {b2}, activation_function=None)\n")
    else:
        tf_layers = f"{h} = {atf}(tf.matmul({X}, {w1}) + {b1})\n{p} = tf.matmul({h}, {w2}) + {b2}\n"
        pt_fwd = f"    {h} = {apt}(torch.matmul(inp, {w1}) + {b1})\n    return torch.matmul({h}, {w2}) + {b2}\n"

    tf_code = (placeholder(X, n_in) + "\n" + placeholder(Y, n_out) + "\n"
               + tf_layers
               + f"{loss} = {ltf}\ntrain_step = {otf}.minimize({loss})\n"
               + f"init = tf.global_variables_initializer()\n{s} = tf.Session()\n{s}.run(init)\n"
               + f"for i in range({steps}):\n    {s}.run(train_step, {feed})\n"
               + (f"    {err} = {s}.run({loss}, {feed})" if in_loop else f"{err} = {s}.run({loss}, {feed})"))
    pt_code = (f"{X} = torch.tensor({xd}, dtype=torch.float32)\n{Y} = torch.tensor({yd}, dtype=torch.float32)\n"
               + "\ndef forward(inp):\n" + pt_fwd + "\n"
               + f"def compute_loss():\n    {p} = forward({X})\n    return {lpt}\n\n"
               + f"optimizer = {opt}\n"
               + f"for i in range({steps}):\n    optimizer.zero_grad()\n    {loss} = compute_loss()\n"
               + f"    {loss}.backward()\n    optimizer.step()\n"
               + (f"    {err} = compute_loss().item()" if in_loop else f"{err} = compute_loss().item()"))
    return dict(tf=tf_code, pt=pt_code, setup=setup, outs=[err])


def t_fetch():
    while True:
        names, s = pick(WB), pick(SESS)
        outs = pick([("WW1", "BB1", "WW2", "BB2"), ("W1_out", "B1_out", "W2_out", "B2_out"),
                     ("final_w1", "final_b1", "final_w2", "final_b2")])
        if distinct(*names, s, *outs):
            break
    k = rng.randint(2, 4)
    pairs = list(zip(outs, names))[:k]
    if pick([True, False]):
        tf_code = "\n".join(f"{o} = {s}.run({n})" for o, n in pairs)
        pt_code = "\n".join(f"{o} = {n}.detach().numpy()" for o, n in pairs)
    else:
        tf_code = f"{', '.join(o for o, _ in pairs)} = {s}.run([{', '.join(n for _, n in pairs)}])"
        pt_code = f"{', '.join(o for o, _ in pairs)} = {', '.join(n + '.detach().numpy()' for _, n in pairs)}"
    return dict(tf=tf_code, pt=pt_code, setup=weight_setup(names, 3, 5, 1, True), outs=[o for o, _ in pairs])


def t_save():
    names, s = pick(WB), pick(SESS)
    k = pick([2, 4, 4])
    folder = pick(["saved_model", "checkpoints", "/tmp/bp_model", "models/train"])
    fname = pick(["bp_model", "model", "net", "well_cost"])
    d_tf = ", ".join(f"'{n}': {n}" for n in names[:k])
    d_pt = ", ".join(f"'{n}': {n}.detach()" for n in names[:k])
    if pick([True, True, False]):
        return dict(tf=f"saver = tf.train.Saver({{{d_tf}}})\nsave_path = saver.save({s}, \"{folder}/{fname}.ckpt\")",
                    pt=f"save_path = \"{folder}/{fname}.pt\"\ntorch.save({{{d_pt}}}, save_path)")
    load = "\n".join(f"{n} = state['{n}'].requires_grad_()" for n in names[:k])
    return dict(tf=f"saver = tf.train.Saver({{{d_tf}}})\nsaver.restore({s}, \"{folder}/{fname}.ckpt\")",
                pt=f"state = torch.load(\"{folder}/{fname}.pt\")\n{load}")


MATH = [
    ("tf.matmul({a}, {b})", "torch.matmul({a}, {b})"),
    ("tf.reduce_sum({a}, axis={k})", "torch.sum({a}, dim={k})"),
    ("tf.reduce_sum({a})", "torch.sum({a})"),
    ("tf.reduce_mean({a}, axis={k})", "torch.mean({a}, dim={k})"),
    ("tf.reduce_mean({a})", "torch.mean({a})"),
    ("tf.reduce_sum({a}, axis=1, keepdims=True)", "torch.sum({a}, dim=1, keepdim=True)"),
    ("tf.reduce_max({a}, axis={k})", "torch.max({a}, dim={k}).values"),
    ("tf.reduce_min({a})", "torch.min({a})"),
    ("tf.square({a})", "torch.square({a})"),
    ("tf.sqrt({a})", "torch.sqrt({a})"),
    ("tf.exp({a})", "torch.exp({a})"),
    ("tf.log({a} + 1.0)", "torch.log({a} + 1.0)"),
    ("tf.abs({a} - {c})", "torch.abs({a} - {c})"),
    ("tf.concat([{a}, {c}], axis={k})", "torch.cat([{a}, {c}], dim={k})"),
    ("tf.stack([{a}, {c}], axis=0)", "torch.stack([{a}, {c}], dim=0)"),
    ("tf.reshape({a}, [-1, 1])", "torch.reshape({a}, (-1, 1))"),
    ("tf.transpose({a})", "{a}.t()"),
    ("tf.argmax({a}, axis=1)", "torch.argmax({a}, dim=1)"),
    ("tf.cast({a} > 0.5, tf.float32)", "({a} > 0.5).float()"),
    ("tf.expand_dims({a}, axis={k})", "torch.unsqueeze({a}, dim={k})"),
    ("tf.clip_by_value({a}, 0.2, 0.8)", "torch.clamp({a}, 0.2, 0.8)"),
    ("tf.nn.softmax({a}, axis=1)", "torch.softmax({a}, dim=1)"),
    ("tf.maximum({a}, {c})", "torch.maximum({a}, {c})"),
    ("tf.minimum({a}, {c})", "torch.minimum({a}, {c})"),
    ("tf.nn.relu({a} - 0.5)", "torch.relu({a} - 0.5)"),
    ("tf.sigmoid({a})", "torch.sigmoid({a})"),
    ("tf.tanh({a})", "torch.tanh({a})"),
    ("tf.zeros_like({a})", "torch.zeros_like({a})"),
    ("tf.ones_like({a})", "torch.ones_like({a})"),
    ("tf.multiply({a}, {c})", "torch.mul({a}, {c})"),
    ("tf.add({a}, {c})", "torch.add({a}, {c})"),
    ("tf.subtract({a}, {c})", "torch.sub({a}, {c})"),
    ("tf.divide({a}, {c} + 1.0)", "torch.div({a}, {c} + 1.0)"),
    ("tf.norm({a})", "torch.norm({a})"),
    ("tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)", "torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)"),
    ("tf.zeros([{k2}, 3])", "torch.zeros({k2}, 3)"),
    ("tf.ones([{k2}, 3])", "torch.ones({k2}, 3)"),
]


def t_math():
    a, c, b = pick([("a", "c", "b"), ("t1", "t2", "m"), ("u", "v", "w"), ("p", "q", "k")])
    base = pick(["r", "out", "z", "val"])
    setup = (f"{a} = torch.tensor(rs.rand(4, 5), dtype=torch.float32)\n"
             f"{c} = torch.tensor(rs.rand(4, 5), dtype=torch.float32)\n"
             f"{b} = torch.tensor(rs.rand(5, 3), dtype=torch.float32)\n")
    tf_l, pt_l, outs = [], [], []
    for i, (ft, fp) in enumerate(rng.sample(MATH, rng.randint(1, 4)), 1):
        kw = dict(a=a, c=c, b=b, k=pick([0, 1]), k2=pick([2, 4]))
        tf_l.append(f"{base}{i} = {ft.format(**kw)}")
        pt_l.append(f"{base}{i} = {fp.format(**kw)}")
        outs.append(f"{base}{i}")
    return dict(tf="\n".join(tf_l), pt="\n".join(pt_l), setup=setup, outs=outs)


def t_dense():
    X, h, p = pick(XY)[0], pick(HID), pick(PRED)
    n_in, n_h, n_out = sizes()
    atf, apt = pick([("tf.nn.relu", "torch.nn.ReLU()"), ("tf.nn.tanh", "torch.nn.Tanh()"),
                     ("tf.nn.sigmoid", "torch.nn.Sigmoid()")])
    return dict(tf=f"{placeholder(X, n_in)}\n{h} = tf.layers.dense({X}, {n_h}, activation={atf})\n"
                   f"{p} = tf.layers.dense({h}, {n_out})",
                pt=f"model = torch.nn.Sequential(\n    torch.nn.Linear({n_in}, {n_h}),\n    {apt},\n"
                   f"    torch.nn.Linear({n_h}, {n_out}),\n)",
                setup="", post=f"_shape = list(model(torch.zeros(3, {n_in})).shape)\n", outs=["_shape"])


TRAIN_TEMPLATES = [(t_full_train, 190), (t_forward, 100), (t_math, 150), (t_variables_np, 70),
                   (t_variables_init, 50), (t_loss, 80), (t_optimizer, 60), (t_fetch, 40), (t_save, 30),
                   (t_add_layer, 24), (t_dense, 40), (t_setup_lines, 40), (t_session_lines, 12)]


# ------------------------------------------------------------------------------------------
# ADDED after looking inside the model (internals_demo.py): version 1 wrote
#     tf.nn.dropout(l1, keep_prob=0.9)  ->  torch.dropout(l1, keep_prob=0.9)
# because almost every training pair kept the function name ("tf.X -> torch.X"). These extra
# training pairs are cases where the PyTorch NAME (and sometimes the argument) must change.
# They are added to the training pool only, AFTER the test set is drawn, so the test set is
# exactly the same as in version 1. Dropout is no longer an "unseen pattern": the test rows
# use keep_prob 0.5 / 0.8 / 0.9, which are never used here. l2_loss and
# softmax_cross_entropy stay out of training on purpose.
# ------------------------------------------------------------------------------------------
def t_dropout_train():
    h = pick(HID)
    keep = pick([0.55, 0.6, 0.65, 0.7, 0.75, 0.85, 0.95])
    p = round(1 - keep, 2)
    arg_tf = pick([f"keep_prob={keep}", f"keep_prob={keep}", f"rate={p}"])
    setup = (f"{h} = torch.tensor(rs.rand(6, 8), dtype=torch.float32)\n"
             "xs = torch.tensor(rs.rand(6, 4), dtype=torch.float32)\n"
             "x = xs\ninputs = xs\n"
             "W1 = torch.tensor(rs.rand(4, 8), dtype=torch.float32)\nWeights_1 = W1\n"
             "b1 = torch.tensor(rs.rand(1, 8), dtype=torch.float32)\nbiases_1 = b1\n")
    if pick([True, False]):
        X, (W, b) = pick(["xs", "x", "inputs"]), pick([("W1", "b1"), ("Weights_1", "biases_1")])
        atf, apt = pick([("tf.nn.tanh", "torch.tanh"), ("tf.nn.sigmoid", "torch.sigmoid"), ("tf.nn.relu", "torch.relu")])
        if atf == "tf.nn.relu" and h == "l1" and X == "xs" and W == "W1":
            h = "hidden"                       # keep clear of the exact wording of the test rows
        return dict(tf=f"{h} = {atf}(tf.matmul({X}, {W}) + {b})\n{h} = tf.nn.dropout({h}, {arg_tf})",
                    pt=f"{h} = {apt}(torch.matmul({X}, {W}) + {b})\n{h} = torch.nn.functional.dropout({h}, p={p})",
                    setup=setup, outs=[h], mode="shape")
    out = pick(["dropped", "h_drop", h])
    return dict(tf=f"{out} = tf.nn.dropout({h}, {arg_tf})",
                pt=f"{out} = torch.nn.functional.dropout({h}, p={p})", setup=setup, outs=[out], mode="shape")


RENAME = [   # TensorFlow name -> a DIFFERENT PyTorch name
    ("tf.nn.leaky_relu({a}, alpha=0.2)", "torch.nn.functional.leaky_relu({a}, negative_slope=0.2)"),
    ("tf.nn.elu({a})", "torch.nn.functional.elu({a})"),
    ("tf.nn.softplus({a})", "torch.nn.functional.softplus({a})"),
    ("tf.nn.log_softmax({a}, axis=1)", "torch.log_softmax({a}, dim=1)"),
    ("tf.nn.l2_normalize({a}, axis=1)", "torch.nn.functional.normalize({a}, dim=1)"),
    ("tf.nn.bias_add({a}, {v})", "{a} + {v}"),
    ("tf.reduce_prod({a}, axis={k})", "torch.prod({a}, dim={k})"),
    ("tf.math.reduce_variance({a})", "torch.var({a}, unbiased=False)"),
    ("tf.math.log1p({a})", "torch.log1p({a})"),
    ("tf.nn.relu6({a})", "torch.nn.functional.relu6({a})"),
]


def t_rename():
    a, c = pick([("a", "c"), ("t1", "t2"), ("u", "w"), ("p", "q")])
    base = pick(["r", "out", "z", "val"])
    setup = (f"{a} = torch.tensor(rs.rand(4, 5), dtype=torch.float32)\n"
             f"{c} = torch.tensor(rs.rand(4, 5), dtype=torch.float32)\n"
             "v = torch.tensor(rs.rand(5), dtype=torch.float32)\n")
    forms = rng.sample(RENAME, rng.randint(1, 2)) + rng.sample(MATH[:6], rng.randint(0, 1))
    rng.shuffle(forms)
    tf_l, pt_l, outs = [], [], []
    for i, (ft, fp) in enumerate(forms, 1):
        kw = dict(a=pick([a, c]), c=c, b="v", v="v", k=pick([0, 1]), k2=2)
        if "matmul" in ft:
            continue
        tf_l.append(f"{base}{i} = {ft.format(**kw)}")
        pt_l.append(f"{base}{i} = {fp.format(**kw)}")
        outs.append(f"{base}{i}")
    return dict(tf="\n".join(tf_l), pt="\n".join(pt_l), setup=setup, outs=outs)


EXTRA_TRAIN_TEMPLATES = [(t_dropout_train, 50), (t_rename, 70)]

# ------------------------------------- patterns that training NEVER sees
def h_two_hidden():
    X, xd = "xs", "x_data"
    n_in, n_h, n_out = sizes()
    n_h2 = pick([4, 8, 10])
    atf, apt = pick(ACT)
    setup = (f"{xd} = rs.rand(7, {n_in})\n" + weight_setup(("W1", "b1", "W2", "b2"), n_in, n_h, n_h2, True)
             + f"W3 = torch.tensor(rs.rand({n_h2}, {n_out}) - 0.5, dtype=torch.float32)\n"
             + f"b3 = torch.tensor(rs.rand(1, {n_out}), dtype=torch.float32)\n")
    body = ("l1 = %s(MM(xs, W1) + b1)\nl2 = %s(MM(l1, W2) + b2)\nprediction = MM(l2, W3) + b3\n")
    return dict(tf=placeholder(X, n_in) + "\n" + (body % (atf, atf)).replace("MM", "tf.matmul")
                   + "out = sess.run(prediction, feed_dict={xs: x_data})",
                pt="xs = torch.tensor(x_data, dtype=torch.float32)\n" + (body % (apt, apt)).replace("MM", "torch.matmul")
                   + "out = prediction.detach().numpy()", setup=setup, outs=["out"])


def h_dropout():
    keep = pick([0.5, 0.8, 0.9])
    return dict(tf=f"l1 = tf.nn.relu(tf.matmul(xs, W1) + b1)\nl1 = tf.nn.dropout(l1, keep_prob={keep})",
                pt=f"l1 = torch.relu(torch.matmul(xs, W1) + b1)\n"
                   f"l1 = torch.nn.functional.dropout(l1, p={round(1 - keep, 2)})",
                setup="xs = torch.tensor(rs.rand(6, 4), dtype=torch.float32)\n"
                      "W1 = torch.tensor(rs.rand(4, 8), dtype=torch.float32)\n"
                      "b1 = torch.tensor(rs.rand(1, 8), dtype=torch.float32)\n", outs=["l1"], mode="shape")


def h_l2():
    lam = pick([0.01, 0.001, 0.1])
    return dict(tf=f"loss = tf.reduce_mean(tf.square(ys - prediction)) + {lam} * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2))",
                pt=f"loss = torch.mean(torch.square(ys - prediction)) + {lam} * "
                   f"(torch.sum(torch.square(W1)) / 2 + torch.sum(torch.square(W2)) / 2)",
                setup="ys = torch.tensor(rs.rand(6, 1), dtype=torch.float32)\n"
                      "prediction = torch.tensor(rs.rand(6, 1), dtype=torch.float32)\n"
                      "W1 = torch.tensor(rs.rand(4, 8), dtype=torch.float32)\n"
                      "W2 = torch.tensor(rs.rand(8, 1), dtype=torch.float32)\n", outs=["loss"])


def h_xent():
    n = pick([3, 4, 5])
    return dict(tf="loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=logits))",
                pt="loss = torch.mean(torch.sum(-ys * torch.log_softmax(logits, dim=1), dim=1))",
                setup=f"logits = torch.tensor(rs.rand(6, {n}), dtype=torch.float32)\n"
                      f"ys = torch.nn.functional.one_hot(torch.arange(6) % {n}, {n}).float()\n", outs=["loss"])


def h_misc():
    forms = [("r = tf.where(a > 0.5, a, c)", "r = torch.where(a > 0.5, a, c)"),
             ("r = tf.tile(a, [2, 1])", "r = a.repeat(2, 1)"),
             ("r = tf.one_hot(idx, 4)", "r = torch.nn.functional.one_hot(idx, 4).float()"),
             ("r = tf.gather(a, idx)", "r = a[idx]"),
             ("r = tf.range(0, 10, 2)", "r = torch.arange(0, 10, 2)"),
             ("r = tf.linspace(0.0, 1.0, 5)", "r = torch.linspace(0.0, 1.0, 5)"),
             ("r = tf.squeeze(tf.expand_dims(a, 0), axis=0)", "r = torch.squeeze(torch.unsqueeze(a, 0), dim=0)"),
             ("r = tf.math.reduce_std(a)", "r = torch.std(a, unbiased=False)")]
    ftf, fpt = pick(forms)
    return dict(tf=ftf, pt=fpt, setup="a = torch.tensor(rs.rand(4, 5), dtype=torch.float32)\n"
                                      "c = torch.tensor(rs.rand(4, 5), dtype=torch.float32)\n"
                                      "idx = torch.tensor([0, 2, 3, 1])\n", outs=["r"])


UNSEEN_TEMPLATES = [(h_two_hidden, 6), (h_dropout, 3), (h_l2, 3), (h_xent, 3), (h_misc, 8)]

# ------------------------------ pieces copied from the well-cost project
W_SETUP = ("w1 = (rs.rand(6, 20) - 0.5).tolist()\nb1 = (rs.rand(20) * 0.1).tolist()\n"
           "w2 = (rs.rand(20, 1) - 0.5).tolist()\nb2 = 0.05\n")
REAL = [
    dict(name="imports (bp_object.py)",
         tf="import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' \nimport tensorflow as tf\n"
            "import tensorflow.compat.v1 as tf\nimport numpy as np\nimport csv\nimport shutil\n"
            "from sklearn import preprocessing\ntf.compat.v1.disable_v2_behavior()\ntf.compat.v1.reset_default_graph()",
         pt="import os\nimport torch\nimport numpy as np\nimport csv\nimport shutil\nfrom sklearn import preprocessing"),
    dict(name="add_layer (all bp_*.py files)",
         tf="#Add layer\ndef add_layer(inputs, Weights, biases, activation_function=None):\n"
            "    # add one more layer and return the output of this layer\n"
            "    Wx_plus_b = tf.matmul(inputs, Weights) + biases\n    if activation_function is None:\n"
            "        outputs = Wx_plus_b\n    else:\n        outputs = activation_function(Wx_plus_b)\n    return outputs",
         pt="#Add layer\n" + ADD_LAYER_PT.replace("\n    Wx_plus_b", "\n    # add one more layer and return the output of this layer\n    Wx_plus_b", 1).rstrip(),
         setup="_x = torch.tensor(rs.rand(5, 3), dtype=torch.float32)\n_W = torch.tensor(rs.rand(3, 4), dtype=torch.float32)\n"
               "_b = torch.tensor(rs.rand(1, 4), dtype=torch.float32)\n",
         post="_r1 = add_layer(_x, _W, _b, torch.tanh)\n_r2 = add_layer(_x, _W, _b)\n", outs=["_r1", "_r2"]),
    dict(name="weights from the GA individual (bp_train.py)",
         tf="Weights_1 = tf.Variable(tf.cast(np.mat(w1), dtype=tf.float32),name='Weights_1')\n"
            "biases_1  = tf.Variable(tf.cast(np.mat(b1), dtype=tf.float32),name='biases_1')\n"
            "Weights_2 = tf.Variable(tf.cast(np.mat(w2), dtype=tf.float32),name='Weights_2')\n"
            "biases_2  = tf.Variable(tf.cast(np.mat(b2), dtype=tf.float32),name='biases_2')   ",
         pt="\n".join(f"{n} = torch.tensor(np.atleast_2d({s}), dtype=torch.float32, requires_grad=True)"
                      for n, s in zip(WB[0], SRC[0])),
         setup=W_SETUP, outs=list(WB[0])),
    dict(name="random weights (bp_test.py)",
         tf="Weights_1  = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights_1')\n"
            "Weights_2 = tf.Variable(tf.random_normal([out_size, 1]),name='Weights_2')\n"
            "biases_1 = tf.Variable(tf.zeros([1, out_size]) + 0.1,name='biases_1')\n"
            "biases_2 = tf.Variable(tf.zeros([1, 1]) + 0.1,name='biases_2')",
         pt="Weights_1 = torch.randn(in_size, out_size, requires_grad=True)\n"
            "Weights_2 = torch.randn(out_size, 1, requires_grad=True)\n"
            "biases_1 = torch.full((1, out_size), 0.1, requires_grad=True)\n"
            "biases_2 = torch.full((1, 1), 0.1, requires_grad=True)",
         setup="in_size = 6\nout_size = 20\n", outs=["Weights_1", "Weights_2", "biases_1", "biases_2"], mode="shape"),
    dict(name="network + loss (bp_train.py)",
         tf="xs = tf.placeholder(shape=[None, 6], dtype=tf.float32)\nys = tf.placeholder(shape=[None, 1], dtype=tf.float32)\n"
            "l1 = add_layer(xs, Weights_1, biases_1, activation_function=tf.nn.relu)\n"
            "prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)\n"
            "loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),\n                 reduction_indices=[1]))\n"
            "error = sess.run(loss, feed_dict={xs: x_data, ys: y_data})",
         pt="xs = torch.tensor(x_data, dtype=torch.float32)\nys = torch.tensor(y_data, dtype=torch.float32)\n"
            "l1 = add_layer(xs, Weights_1, biases_1, activation_function=torch.relu)\n"
            "prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)\n"
            "loss = torch.mean(torch.sum(torch.square(ys - prediction), dim=1))\nerror = loss.item()",
         setup="x_data = rs.rand(12, 6)\ny_data = rs.rand(12, 1)\n" + weight_setup(WB[0], 6, 20, 1, True) + ADD_LAYER_PT,
         outs=["error"]),
    dict(name="variable initialisation + session (bp_object.py)",
         tf="if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:\n"
            "   init = tf.initialize_all_variables()\nelse:\n   init = tf.global_variables_initializer()\n"
            "model = tf.Session()\nmodel.run(init)",
         pt="# Not needed in PyTorch: tensors are ready as soon as they are created."),
    dict(name="read trained weights (bp_object.py)",
         tf="W1 = model.run(Weights_1)\nW2 = model.run(Weights_2)\nB1 = model.run(biases_1)\nB2 = model.run(biases_2)",
         pt="W1 = Weights_1.detach().numpy()\nW2 = Weights_2.detach().numpy()\n"
            "B1 = biases_1.detach().numpy()\nB2 = biases_2.detach().numpy()",
         setup=weight_setup(WB[0], 6, 20, 1, True), outs=["W1", "W2", "B1", "B2"]),
    dict(name="complete training step (bp_train.py)",
         tf="xs = tf.placeholder(shape=[None, 6], dtype=tf.float32)\nys = tf.placeholder(shape=[None, 1], dtype=tf.float32)\n"
            "Weights_1 = tf.Variable(tf.cast(np.mat(w1), dtype=tf.float32),name='Weights_1')\n"
            "biases_1  = tf.Variable(tf.cast(np.mat(b1), dtype=tf.float32),name='biases_1')\n"
            "Weights_2 = tf.Variable(tf.cast(np.mat(w2), dtype=tf.float32),name='Weights_2')\n"
            "biases_2  = tf.Variable(tf.cast(np.mat(b2), dtype=tf.float32),name='biases_2')\n"
            "l1 = add_layer(xs, Weights_1, biases_1, activation_function=tf.nn.relu)\n"
            "prediction = add_layer(l1, Weights_2, biases_2, activation_function=None)\n"
            "loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),\n                 reduction_indices=[1]))\n"
            "train_step = tf.train.AdamOptimizer(Learning_rate).minimize(loss)\n"
            "with tf.Session() as sess:\n    sess.run(tf.global_variables_initializer())\n"
            "    for i in range(Train_itteration):\n"
            "        train = sess.run(train_step, feed_dict={xs: x_data, ys: y_data})\n"
            "        error = sess.run(loss, feed_dict={xs: x_data, ys: y_data})",
         pt="xs = torch.tensor(x_data, dtype=torch.float32)\nys = torch.tensor(y_data, dtype=torch.float32)\n"
            + "\n".join(f"{n} = torch.tensor(np.atleast_2d({s}), dtype=torch.float32, requires_grad=True)"
                        for n, s in zip(WB[0], SRC[0]))
            + "\n\ndef forward(inp):\n    l1 = add_layer(inp, Weights_1, biases_1, activation_function=torch.relu)\n"
              "    return add_layer(l1, Weights_2, biases_2, activation_function=None)\n\n"
              "def compute_loss():\n    prediction = forward(xs)\n"
              "    return torch.mean(torch.sum(torch.square(ys - prediction), dim=1))\n\n"
              "optimizer = torch.optim.Adam([Weights_1, biases_1, Weights_2, biases_2], lr=Learning_rate)\n"
              "for i in range(Train_itteration):\n    optimizer.zero_grad()\n    loss = compute_loss()\n"
              "    loss.backward()\n    optimizer.step()\n    error = compute_loss().item()",
         setup="x_data = rs.rand(12, 6)\ny_data = rs.rand(12, 1)\nLearning_rate = 0.01\nTrain_itteration = 30\n"
               + W_SETUP + ADD_LAYER_PT, outs=["error"]),
]


# ------------------------------------------------------------------ assembly
def finish(ex, cat, group, idx):
    ex = dict(ex)
    ex.setdefault("setup", "")
    ex.setdefault("post", "")
    ex.setdefault("outs", [])
    ex.setdefault("mode", "value")
    ex.update(id=f"{group}-{cat}-{idx}", category=cat, group=group)
    ex["messages"] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["tf"]},
                      {"role": "assistant", "content": ex["pt"]}]
    return ex


def generate(templates, group, seen, per_template=None):
    out = []
    for fn, n in templates:
        n = per_template or n
        cat = fn.__name__[2:]
        made = tries = 0
        while made < n and tries < n * 60:
            tries += 1
            ex = fn()
            if ex["tf"] in seen:
                continue
            seen.add(ex["tf"])
            out.append(finish(ex, cat, group, made))
            made += 1
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(here, "data"), exist_ok=True)
    seen = set(r["tf"] for r in REAL)

    # The test set is generated FIRST so that nothing in it can end up in training.
    test = generate(TRAIN_TEMPLATES, "seen_patterns", seen, per_template=5)
    test += generate(UNSEEN_TEMPLATES, "unseen_patterns", seen)
    test += [finish({k: v for k, v in r.items() if k != "name"} | {"name": r["name"]}, "real", "real_code", i)
             for i, r in enumerate(REAL)]
    pool = generate(TRAIN_TEMPLATES, "train", seen)
    pool += generate(EXTRA_TRAIN_TEMPLATES, "train", seen)      # added in version 2

    # Drop training examples that do not fit in the 512-token window.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    def n_tokens(ex):
        text = tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
        return len(tok(text)["input_ids"])
    lengths = [n_tokens(ex) for ex in pool]
    kept = [ex for ex, n in zip(pool, lengths) if n <= MAX_TOKENS]
    print(f"generated {len(pool)} training pairs, longest {max(lengths)} tokens, "
          f"dropped {len(pool) - len(kept)} longer than {MAX_TOKENS}")

    rng.shuffle(kept)
    n_val = len(kept) // 10
    val, train = kept[:n_val], kept[n_val:]

    # Prove every reference answer runs (and is deterministic where we compare numbers).
    def verify(ex):
        if not ex["outs"]:
            compile(ex["pt"], ex["id"], "exec")
            return None
        res, err = run_snippet(ex["setup"], ex["pt"], ex["post"], ex["outs"])
        return f"{ex['id']}: {err}" if res is None else None
    if "--no-verify" not in sys.argv:
        with ThreadPoolExecutor(8) as pool_exec:
            problems = [p for p in pool_exec.map(verify, train + val + test) if p]
        if problems:
            print("\n".join(problems[:20]))
            sys.exit(f"{len(problems)} reference answers failed to run")
        print("every reference answer runs")

    for name, rows in [("train", train), ("val", val), ("test", test)]:
        with open(os.path.join(here, "data", f"{name}.jsonl"), "w") as f:
            for ex in rows:
                f.write(json.dumps(ex) + "\n")
        cats = {}
        for ex in rows:
            key = ex["group"] if name == "test" else ex["category"]
            cats[key] = cats.get(key, 0) + 1
        print(f"{name}: {len(rows)} examples  {cats}")


if __name__ == "__main__":
    main()
