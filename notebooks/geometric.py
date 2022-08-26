"""
# Core code for GeometricConvolutions

## License:
Copyright 2022 David W. Hogg and contributors.
The code in GeometricConvolutions is licensed under the open-source MIT License.
See the file `LICENSE` for more details.

## Authors:
- David W. Hogg (NYU)

## To-do items:
- Drop the k-tensor and geometric filter classes; there can be only one class.
- Move over to jax.
- Make the geometric_filter inherit geometric_image to reduce repeated code.
- Is it possible to make the data a big block (not a dictionary) but have the block addressable with keys()? I bet it is...?
- Create tests for group operations on k-tensor images.
- Fix sizing of multi-filter plots.
- Switch over to jax so this is useful for ML people.
- Switch the structure of the image and filter so they make better use of jax.numpy array objects.
- Need to implement index permutation operation.
- Need to implement Levi-Civita contraction for general dimensions.
- Need to build tests for the contractions.
- Need to implement bin-down and bin-up operators.
"""

import itertools as it
import numpy as np
import pylab as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import cmastro

TINY = 1.e-5

# ------------------------------------------------------------------------------
# PART 1: Make and test a complete group

def make_all_generators(D):

    # Make the flip operator
    foo = np.ones(D).astype(int)
    foo[0] = -1
    gg = np.diag(foo).astype(int)
    generators = [gg, ]

    # Make the 90-degree rotation operators
    for i in range(D):
        for j in range(i + 1, D):
            gg = np.eye(D).astype(int)
            gg[i, i] = 0
            gg[j, j] = 0
            gg[i, j] = -1
            gg[j, i] = 1
            generators.append(gg)

    return np.array(generators)

def make_all_operators(D):
    generators = make_all_generators(D)
    operators = np.array([np.eye(D).astype(int), ])
    foo = 0
    while len(operators) != foo:
        foo = len(operators)
        operators = make_new_operators(operators, generators)
    return(operators)

def make_new_operators(operators, generators):
    for op in operators:
        for gg in generators:
            op2 = (gg @ op).astype(int)
            operators = np.unique(np.append(operators, op2[None, :, :], axis=0), axis=0)
    return operators

def test_group(operators):
    D = len(operators[0])
    # Check that the list of group operators is closed
    for gg in operators:
        for gg2 in operators:
            if ((gg @ gg2).astype(int) not in operators):
                return False
    print("group is closed under multiplication")
    # Check that gg.T is gg.inv for all gg in group?
    for gg in operators:
        if not np.allclose(gg @ gg.T, np.eye(D)):
            return False
    print("group operators are the transposes of their inverses")
    return True

# ------------------------------------------------------------------------------
# PART 2: Define a k-tensor.

class ktensor:

    def name(k, parity):
        nn = "tensor"
        if k == 0:
            nn = "scalar"
        if k == 1:
            nn = "vector"
        if parity % 2 == 1:
            nn = "pseudo" + nn
        if k > 1:
            nn = "${}$-".format(k) + nn
        return nn

    def __init__(self, data, parity, D):
        self.levi_civita = None
        self.D = D
        assert self.D > 1, \
        "ktensor: geometry makes no sense if D<2."
        self.parity = parity % 2
        if len(np.atleast_1d(data)) == 1:
            self.data = data
            self.k = 0
        else:
            self.data = np.array(data)
            self.k = len(data.shape)
            assert np.all(np.array(data.shape) == self.D), \
            "ktensor: shape must be (D, D, D, ...), but instead it's {}".format(data.shape)

    def __getitem__(self, key):
        return self.data[key]

    def __add__(self, other):
        assert self.k == other.k, \
        "ktensor: can't add objects of different k"
        assert self.parity == other.parity, \
        "ktensor: can't add objects of different parity"
        return ktensor(self.data + other.data, self.parity, self.D)

    def __mul__(self, other):
        if self.k == 0 or other.k == 0:
            return ktensor(self.data * other.data,
                           self.parity + other.parity, self.D)
        return ktensor(np.multiply.outer(self.data, other.data),
                       self.parity + other.parity, self.D)

    def __str__(self):
        return "<k-tensor object in D={} with k={} and parity={}>".format(
            self.D, self.k, self.parity)

    def norm(self):
        if self.k == 0:
            return np.abs(self.data)
        return np.linalg.norm(self.data)

    def times_scalar(self, scalar):
        return ktensor(scalar * self.data, self.parity, self.D)

    def times_group_element(self, gg):
        """
        # Notes / Bugs:
        - THIS IS UNTESTED.
        - This is incomprehensible.
        """
        assert self.k < 14
        assert gg.shape == (self.D, self.D)
        sign, logdet = np.linalg.slogdet(gg)
        assert logdet == 0.
        if self.k == 0:
            newdata = 1. * self.data * sign ** self.parity
        else:
            firstletters  = "abcdefghijklm"
            secondletters = "nopqrstuvwxyz"
            einstr = "".join([firstletters[i] for i in range(self.k)]) +"," + \
            ",".join([secondletters[i] + firstletters[i] for i in range(self.k)])
            foo = (self.data, ) + self.k * (gg, )
            newdata = np.einsum(einstr, *foo) * sign ** self.parity
        return ktensor(newdata, self.parity, self.D)

    def contract(self, i, j):
        assert self.k < 27
        assert self.k >= 2
        assert i < j
        assert i < self.k
        assert j < self.k
        letters  = "bcdefghijklmnopqrstuvwxyz"
        einstr = letters[:i] + "a" + letters[i:j-1] + "a" + letters[j-1:self.k-2]
        return ktensor(np.einsum(einstr, self.data), self.parity, self.D)

    def levi_civita_contract(self, index):
        assert self.D in [2, 3] # BECAUSE WE SUCK
        assert (self.k + 1) >= self.D # so we have enough indices work on
        if self.D == 2:
            otherdata = np.zeros_like(self.data)
            otherdata[..., 0] =  1. * np.take(self.data, 1, axis=index)
            otherdata[..., 1] = -1. * np.take(self.data, 0, axis=index)
            return ktensor(otherdata, self.parity + 1, self.D)
        if self.D == 3:
            assert len(index) == 2
            i, j = index
            assert i < j
            otherdata = np.zeros_like(self.data[..., 0])
            otherdata[..., 0] = np.take(np.take(self.data, 2, axis=j), 1, axis=i) \
                              - np.take(np.take(self.data, 1, axis=j), 2, axis=i)
            otherdata[..., 1] = np.take(np.take(self.data, 0, axis=j), 2, axis=i) \
                              - np.take(np.take(self.data, 2, axis=j), 0, axis=i)
            otherdata[..., 2] = np.take(np.take(self.data, 1, axis=j), 0, axis=i) \
                              - np.take(np.take(self.data, 0, axis=j), 1, axis=i)
            return ktensor(otherdata, self.parity + 1, self.D)
        return

# Now test group actions on k-tensors:
def test_group_actions(operators):
    """
    # Notes:
    - This only does minimal tests!
    """
    D = len(operators[0])

    for parity in [0, 1]:

        # vector dot vector
        v1 = ktensor(np.random.normal(size=D), parity, D)
        v2 = ktensor(np.random.normal(size=D), parity, D)
        dots = [(v1.times_group_element(gg)
                 * v2.times_group_element(gg)).contract(0, 1).data
                for gg in operators]
        dots = np.array(dots)
        if not np.allclose(dots, np.mean(dots)):
            print("failed (parity = {}) vector dot test.".format(parity))
            return False
        print("passed (parity = {}) vector dot test.".format(parity))

        # tensor times tensor
        T3 = ktensor(np.random.normal(size=(D, D)), parity, D)
        T4 = ktensor(np.random.normal(size=(D, D)), parity, D)
        dots = [(T3.times_group_element(gg)
                 * T4.times_group_element(gg)).contract(1, 2).contract(0, 1).data
                for gg in operators]
        dots = np.array(dots)
        if not np.allclose(dots, np.mean(dots)):
            print("failed (parity = {}) tensor times tensor test".format(parity))
            return False
        print("passed (parity = {}) tensor times tensor test".format(parity))

        # vectors dotted through tensor
        v5 = ktensor(np.random.normal(size=D), 0, D)
        dots = [(v5.times_group_element(gg) * T3.times_group_element(gg)
                 * v2.times_group_element(gg)).contract(1, 2).contract(0, 1).data
                for gg in operators]
        dots = np.array(dots)
        if not np.allclose(dots, np.mean(dots)):
            print("failed (parity = {}) v T v test.".format(parity))
            return False
        print("passed (parity = {}) v T v test.".format(parity))
    
    return True

# ------------------------------------------------------------------------------
# PART 3: Define a geometric (k-tensor) filter.

class geometric_filter:

    def zeros(M, k, parity, D):
        """
        WARNING: NO `self`; static method maybe?
        """
        shape = (M ** D, ) + k * (D, )
        return geometric_filter(np.zeros(shape), parity, D)

    def hash(self, pixel):
        return tuple(pixel.astype(int))

    def make_pixels_and_keys(self):
        foo = range(-self.m, self.m + 1)
        self._pixels = np.array([pp for pp in it.product(foo, repeat=self.D)]).astype(int)
        self._keys = [self.hash(pp) for pp in self._pixels]
        return

    def __init__(self, data, parity, D):
        self.D = D
        self.M = np.round(len(data) ** (1. / D)).astype(int)
        assert len(data) == self.M ** self.D, \
        "geometric_filter: data doesn't seem to be the right length?"
        self.m = (self.M - 1) // 2
        assert self.M == 2 * self.m + 1, \
        "geometric_filter: M needs to be odd."
        self.make_pixels_and_keys()
        self.parity = parity % 2
        self.data = {kk: ktensor(ff, self.parity, self.D)
                     for kk, ff in zip(self.keys(), data)}
        self.k = self[self.keys()[0]].k
        return

    def copy(self):
        return geometric_filter(self.unpack(), self.parity, self.D)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, thing):
        self.data[key] = thing
        return

    def keys(self):
        return self._keys

    def pixels(self):
        return self._pixels

    def __add__(self, other):
        assert self.D == other.D
        assert self.M == other.M
        newfilter = self.copy()
        for kk in self.keys():
            newfilter[kk] = self[kk] + other[kk]
        return newfilter

    def __str__(self):
        return "<geometric filter object in D={} with M={} (m={}), k={}, and parity={}>".format(
            self.D, self.M, self.m, self.k, self.parity)

    def times_group_element(self, gg):
        newfilter = self.copy()
        for pp, kk in zip(self.pixels(), self.keys()):
            newfilter[kk] = self[self.hash(gg.T @ pp)].times_group_element(gg)
        return newfilter

    def unpack(self):
        return np.array([self[kk].data for kk in self.keys()])

    def bigness(self):
        numerator, denominator = 0., 0.
        for pp, kk in zip(self.pixels(), self.keys()):
            numerator += np.linalg.norm(pp * self[kk].norm(), ord=2)
            denominator += self[kk].norm()
        return numerator / denominator

    def normalize(self):
        max_norm = np.max([self[kk].norm() for kk in self.keys()])
        if max_norm <= TINY:
            return self
        return self.times_scalar(1. / max_norm)

    def times_scalar(self, scalar):
        newfilter = self.copy()
        for kk in self.keys():
            newfilter[kk] = self[kk].times_scalar(scalar)
        return newfilter

    def rectify(self):
        if self.k == 0:
            if np.sum([self[kk].data for kk in self.keys()]) < 0:
                return self.times_scalar(-1)
            return self
        if self.k == 1:
            if self.parity % 2 == 0:
                if np.sum([np.dot(pp, self[kk].data) for kk, pp in zip(self.keys(), self.pixels())]) < 0:
                    return self.times_scalar(-1)
                return self
            elif self.D == 2:
                if np.sum([np.cross(pp, self[kk].data) for kk, pp in zip(self.keys(), self.pixels())]) < 0:
                    return self.times_scalar(-1)
                return self
        return self

    def contract(self, i, j):
        assert self.k >= 2
        newfilter = geometric_filter.zeros(self.M, self.k - 2, self.parity, self.D)
        for kk in newfilter.keys():
            newfilter[kk] = self[kk].contract(i, j)
        return newfilter

    def levi_civita_contract(self, index):
        assert (self.k + 1) >= self.D
        newfilter = geometric_filter.zeros(self.N, self.k - self.D + 2, self.parity + 1, self.D)
        for kk in self.keys():
            newfilter[kk] = self[kk].levi_civita_contract(index)
        return newimage

# Visualize the filters.

FIGSIZE = (4, 3)
XOFF, YOFF = 0.15, -0.1

def setup_plot():
    fig = plt.figure(figsize=FIGSIZE)
    return fig

def nobox(ax):
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.axis("off")
    return

def finish_plot(ax, title, pixels, D):
    ax.set_title(title)
    if D == 2:
        ax.set_xlim(np.min(pixels)-0.55, np.max(pixels)+0.55)
        ax.set_ylim(np.min(pixels)-0.55, np.max(pixels)+0.55)
    if D == 3:
        ax.set_xlim(np.min(pixels)-0.75, np.max(pixels)+0.75)
        ax.set_ylim(np.min(pixels)-0.75, np.max(pixels)+0.75)
    ax.set_aspect("equal")
    nobox(ax)
    return

def plot_boxes(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.plot([x-0.5, x-0.5, x+0.5, x+0.5, x-0.5],
                 [y-0.5, y+0.5, y+0.5, y-0.5, y-0.5], "k-", lw=0.5, zorder=10)
    return

def fill_boxes(ax, xs, ys, ws, vmin, vmax, cmap, zorder=-100, colorbar=False, alpha=1.):
    cmx = cm.ScalarMappable(cmap=cm.get_cmap(cmap))
    cmx.set_clim(vmin, vmax)
    cs = cmx.to_rgba(ws)
    if colorbar:
        plt.colorbar(cmx, ax=ax)
    for x, y, c in zip(xs, ys, cs):
        ax.fill_between([x - 0.5, x + 0.5], [y - 0.5, y - 0.5], [y + 0.5, y + 0.5],
                             color=c, zorder=zorder, alpha=alpha)
    return

def plot_scalars(ax, M, xs, ys, ws, boxes=True, fill=True, symbols=True,
                 vmin=-2., vmax=2., cmap="cma:unph", colorbar=False):
    if boxes:
        plot_boxes(ax, xs, ys)
    if fill:
        fill_boxes(ax, xs, ys, ws, vmin, vmax, cmap, colorbar=colorbar)
    if symbols:
        height = ax.get_window_extent().height
        ss = (5 * height / M) * np.abs(ws)
        ax.scatter(xs[ws > TINY], ys[ws > TINY],
                   marker="+", c="k",
                   s=ss[ws > TINY], zorder=100)
        ax.scatter(xs[ws < -TINY], ys[ws < -TINY],
                   marker="_", c="k",
                   s=ss[ws < -TINY], zorder=100)
    return

def plot_scalar_filter(filter, title, ax=None):
    assert filter.k == 0
    if filter.D not in [2, 3]:
        print("plot_scalar_filter(): Only works for D in [2, 3].")
        return
    if ax is None:
        fig = setup_plot()
        ax = fig.gca()
    MtotheD = filter.M ** filter.D
    xs, ys, zs = np.zeros(MtotheD), np.zeros(MtotheD), np.zeros(MtotheD)
    ws = np.zeros(MtotheD)
    for i, (kk, pp) in enumerate(zip(filter.keys(), filter.pixels())):
        ws[i] = filter[kk].data
        if filter.D == 2:
            xs[i], ys[i] = pp
        elif filter.D == 3:
            xs[i], ys[i] = pp[0] + XOFF * pp[2], pp[1] + XOFF * pp[2]
    plot_scalars(ax, filter.M, xs, ys, ws, vmin=-3., vmax=3.)
    finish_plot(ax, title, filter.pixels(), filter.D)
    return ax

def plot_vectors(ax, xs, ys, ws, boxes=True, fill=True,
                 vmin=0., vmax=2., cmap="cma:lacerta_r", scaling=0.33):
    if boxes:
        plot_boxes(ax, xs, ys)
    if fill:
        fill_boxes(ax, xs, ys, np.linalg.norm(ws, axis=-1), vmin, vmax, cmap, alpha=0.25)
    for x, y, w in zip(xs, ys, ws):
        normw = np.linalg.norm(w)
        if normw > TINY:
            ax.arrow(x - scaling * w[0], y - scaling * w[1],
                     2 * scaling * w[0], 2 * scaling * w[1],
                     length_includes_head=True,
                     head_width= 0.24 * scaling * normw,
                     head_length=0.72 * scaling * normw,
                     color="k", zorder=100)
    return

def plot_vector_filter(filter, title, ax=None):
    assert filter.k == 1
    if filter.D not in [2, 3]:
        print("plot_vector_filter(): Only works for D in [2, 3].")
        return
    if ax is None:
        fig = setup_plot()
        ax = fig.gca()
    MtotheD = filter.M ** filter.D
    xs, ys, zs = np.zeros(MtotheD), np.zeros(MtotheD), np.zeros(MtotheD)
    ws = np.zeros((MtotheD, filter.D))
    for i, (kk, pp) in enumerate(zip(filter.keys(), filter.pixels())):
        ws[i] = filter[kk].data
        if filter.D == 2:
            xs[i], ys[i] = pp
        elif filter.D == 3:
            xs[i], ys[i] = pp[0] + XOFF * pp[2], pp[1] + YOFF * pp[2]
    plot_vectors(ax, xs, ys, ws, vmin=0., vmax=3.)
    finish_plot(ax, title, filter.pixels(), filter.D)
    return ax

Rx = np.array([[-1.0, -1.0, 1.0, 1.0, -1.0, 0.0,  1.0],
               [-1.5,  1.5, 1.5, 0.2, -0.2, 0.0, -1.5]])

def plot_one_tensor(ax, x, y, T, color="k", zorder=0, scaling=0.2):
    dx, dy = scaling * T @ Rx
    ax.plot(x + dx, y + dy, ls="-", color=color)
    return

def plot_tensors(ax, xs, ys, ws, boxes=True, fill=True,
                 vmin=0., vmax=2., cmap="cma:hesperia_r", scaling=0.33):
    if boxes:
        plot_boxes(ax, xs, ys)
    if fill:
        fill_boxes(ax, xs, ys, np.linalg.norm(ws, axis=-1), vmin, vmax, cmap, alpha=0.25)
    for x, y, w in zip(xs, ys, ws):
        normw = np.linalg.norm(w)
        if normw > TINY:
            plot_one_tensor(ax, x, y, w, color="k", zorder=100)
    return

def plot_tensor_filter(filter, title, ax=None):
    assert filter.k == 2, "plot_tensor_filter(): Only 2-tensors (for now)."
    if filter.D not in [2, ]:
        print("plot_vector_filter(): Only works for D in [2, ].")
        return
    if ax is None:
        fig = setup_plot()
        ax = fig.gca()
    MtotheD = filter.M ** filter.D
    xs, ys = np.zeros(MtotheD), np.zeros(MtotheD)
    ws = np.zeros((MtotheD, filter.D, filter.D))
    for i, (kk, pp) in enumerate(zip(filter.keys(), filter.pixels())):
        ws[i] = filter[kk].data
        if filter.D == 2:
            xs[i], ys[i] = pp
        elif filter.D == 3:
            xs[i], ys[i] = pp[0] + XOFF * pp[2], pp[1] + YOFF * pp[2]
    plot_tensors(ax, xs, ys, ws, vmin=0., vmax=3.)
    finish_plot(ax, title, filter.pixels(), filter.D)
    return ax

def plot_nothing(ax):
    ax.set_title(" ")
    nobox(ax)
    return

def plot_filters(filters, names, n):
    assert len(filters) <= n
    bar = 10. # figure width in inches?
    fig, axes = plt.subplots(1, n, figsize = (bar, 0.03 * bar + bar / n)) # magic
    if n == 1:
        axes = [axes, ]
    plt.subplots_adjust(left=0.001, right=0.999, wspace=0.2/n,
                        bottom=0.001, top=0.999-0.1, hspace=0)
    for i, (ff, name) in enumerate(zip(filters, names)):
        if ff.k == 0:
            plot_scalar_filter(ff, name, ax=axes[i])
        if ff.k == 1:
            plot_vector_filter(ff, name, ax=axes[i])
        if ff.k == 2:
            plot_tensor_filter(ff, name, ax=axes[i])
    for i in range(len(filters), n):
        plot_nothing(axes[i])
    return fig

# ------------------------------------------------------------------------------
# PART 4: Use group averaging to find unique invariant filters.

def get_unique_invariant_filters(M, k, parity, D, operators):

    # make the seed filters
    tmp = geometric_filter.zeros(M, k, parity, D)
    M, D, keys, shape = tmp.M, tmp.D, tmp.keys(), tmp.unpack().shape
    allfilters = []
    if k == 0:
        for kk in keys:
            thisfilter = geometric_filter.zeros(M, k, parity, D)
            thisfilter[kk].data = 1
            allfilters.append(thisfilter)
    else:
        for kk in keys:
            thisfilter = geometric_filter.zeros(M, k, parity, D)
            for indices in it.product(range(D), repeat=k):
                thisfilter[kk].data[indices] = 1
                allfilters.append(thisfilter)

    # do the group averaging
    bigshape = (len(allfilters), ) + thisfilter.unpack().flatten().shape
    filter_matrix = np.zeros(bigshape)
    for i, f1 in enumerate(allfilters):
        ff = geometric_filter.zeros(M, k, parity, D)
        for gg in operators:
            ff = ff + f1.times_group_element(gg)
        filter_matrix[i] = ff.unpack().flatten()

    # do the SVD
    u, s, v = np.linalg.svd(filter_matrix)
    sbig = s > TINY
    if not np.any(sbig):
        return []

    # normalize the amplitudes so they max out at +/- 1.
    amps = v[sbig] / np.max(np.abs(v[sbig]), axis=1)[:, None]
    # make sure the amps are positive, generally
    for i in range(len(amps)):
        if np.sum(amps[i]) < 0:
            amps[i] *= -1
    # make sure that the zeros are zeros.
    amps[np.abs(amps) < TINY] = 0.

    # order them
    filters = [geometric_filter(aa.reshape(shape), parity, D).normalize() for aa in amps]
    norms = [ff.bigness() for ff in filters]
    I = np.argsort(norms)
    filters = [filters[i] for i in I]

    # now do k-dependent rectification:
    filters = [ff.rectify() for ff in filters]

    return filters

# ------------------------------------------------------------------------------
# PART 5: Define geometric (k-tensor, torus) images.

class geometric_image:

    def zeros(N, k, parity, D):
        """
        WARNING: No `self`; static method.
        """
        shape = D * (N, ) + k * (D, )
        return geometric_image(np.zeros(shape), parity, D)

    def hash(self, pixel):
        """
        ## Note:
        - Deals with torus by modding (with `np.remainder()`).
        """
        return tuple(np.remainder(pixel.astype(int), self.N))

    def make_pixels_and_keys(self):
        self._pixels = np.array([pp for pp in
                                 it.product(range(self.N),
                                            repeat=self.D)]).astype(int)
        self._keys = [self.hash(pp) for pp in self._pixels]
        return

    def __init__(self, data, parity, D):
        self.D = D
        self.N = len(data)
        assert data.shape[:D] == self.D * (self.N, ), \
        "geometric_filter: data must be square."
        self.make_pixels_and_keys()
        self.parity = parity % 2
        self.data = {kk: ktensor(data[kk], self.parity, self.D)
                     for kk in self.keys()}
        self.k = self[self.keys()[0]].k
        return

    def copy(self):
        return geometric_image(self.unpack(), self.parity, self.D)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, thing):
        self.data[key] = thing
        return

    def keys(self):
        return self._keys

    def pixels(self):
        return self._pixels

    def __add__(self, other):
        assert self.D == other.D
        assert self.N == other.N
        assert self.k == other.k
        assert self.parity == other.parity
        newimage = self.copy()
        for kk in self.keys():
            newimage[kk] = self[kk] + other[kk]
        return newimage

    def __mul__(self, other):
        assert self.D == other.D
        assert self.N == other.N
        newimage = geometric_image.zeros(self.N, self.k + other.k,
                                         self.parity + other.parity, self.D)
        for kk in self.keys():
            newimage[kk] = self[kk] * other[kk] # handled by ktensor
        return newimage

    def __str__(self):
        return "<geometric image object in D={} with N={}, k={}, and parity={}>".format(
            self.D, self.N, self.k, self.parity)

    def unpack(self):
        shape = self.D * (self.N, ) + self.k * (self.D, )
        package = np.zeros(shape)
        for kk in self.keys():
            package[kk] = self[kk].data
        return package

    def convolve_with(self, filter):
        newimage = geometric_image.zeros(self.N, self.k + filter.k,
                                         self.parity + filter.parity, self.D)
        for kk, pp in zip(self.keys(), self.pixels()):
            for dk, dp in zip(filter.keys(), filter.pixels()):
                newimage[kk] += self[self.hash(pp + dp)] * filter[dk]
        return newimage

    def times_scalar(self, scalar):
        newimage = self.copy()
        for kk in newimage.keys():
            newimage[kk] = self[kk].times_scalar(scalar)
        return newimage

    def normalize(self):
        max_norm = np.max([self[kk].norm() for kk in self.keys()])
        if max_norm > TINY:
            return self.times_scalar(1. / max_norm)
        else:
            return self.times_scalar(1.)

    def contract(self, i, j):
        assert self.k >= 2
        newk = self.k - 2
        newimage = geometric_image.zeros(self.N, newk, self.parity, self.D)
        for kk in self.keys():
            newimage[kk] = self[kk].contract(i, j)
        return newimage

    def levi_civita_contract(self, index):
        assert (self.k + 1) >= self.D
        newimage = geometric_image.zeros(self.N, self.k - self.D + 2,
                                         self.parity + 1, self.D)
        for kk in self.keys():
            newimage[kk] = self[kk].levi_civita_contract(index)
        return newimage

# Visualize geometric images

def setup_image_plot():
    ff = plt.figure(figsize=(8, 6))
    return ff

def plot_scalar_image(image, vmin=-1., vmax=1., ax=None, colorbar=False):
    assert image.D == 2
    assert image.k == 0
    if ax is None:
        ff = setup_image_plot()
        ax = ff.gca()
    plotdata = np.array([[pp[0], pp[1], image[kk].data]
                         for kk, pp in zip(image.keys(), image.pixels())])
    ax.set_aspect("equal", adjustable="box")
    if vmin is None:
        vmin = np.percentile(plotdata[:, 2],  2.5)
    if vmax is None:
        vmax = np.percentile(plotdata[:, 2], 97.5)
    plot_scalars(ax, image.N, plotdata[:, 0], plotdata[:, 1], plotdata[:, 2],
                 symbols=False, vmin=vmin, vmax=vmax, colorbar=colorbar)
    image_axis(ax, plotdata)
    return ax

def image_axis(ax, plotdata):
    ax.set_xlim(np.min(plotdata[:, 0])-0.5, np.max(plotdata[:, 0])+0.5)
    ax.set_ylim(np.min(plotdata[:, 1])-0.5, np.max(plotdata[:, 1])+0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def plot_vector_image(image, ax=None):
    assert image.D == 2
    assert image.k == 1
    if ax is None:
        ff = setup_image_plot()
        ax = ff.gca()
    plotdata = np.array([[pp[0], pp[1], image[kk].data[0], image[kk].data[1]]
                         for kk, pp in zip(image.keys(), image.pixels())])
    plot_vectors(ax, plotdata[:, 0], plotdata[:, 1], plotdata[:, 2:4],
                 boxes=True, fill=True, scaling=0.5)
    image_axis(ax, plotdata)
    return ax

def plot_image(image, **kwargs):
    assert image.D == 2
    if image.k == 0:
        plot_scalar_image(image, **kwargs)
    if image.k == 1:
        plot_vector_image(image, **kwargs)

def plot_images(images):
    """
    # Notes:
    - This takes a list of lists, each inner list is an image and
      a LaTeX expression.
    """
    nim = len(images)
    n = np.floor(np.sqrt(nim)).astype(int)
    m = np.ceil(nim / n).astype(int)
    print(len(images), n, m)
    bar = 10. # inches?
    fig, axes = plt.subplots(m, n, figsize = (bar, 0.2 * m + bar * m / n)) # magic
    axes = np.atleast_1d(axes).flatten()
    plt.subplots_adjust(left=0.001, right=0.999, wspace=0.2/n,
                        bottom=0.001, top=0.999-0.07/m, hspace=0.2/m)
    for ax, (image, latex) in zip(axes, images):
        plot_image(image, ax=ax)
        finish_plot(ax, "$" + latex + "$", image.pixels(), image.D)
    for i in range(nim, n * m):
        plot_nothing(axes[i])
    return fig
