# coding=utf-8
"""
The implementation of the k-Bags feature vector proposed by the paper:

Chen, X., Jørgensen, M. S., Li, J., Hammer, B. (2018). J. Chem. Theory. Comput.

"""
from __future__ import print_function, absolute_import

import numpy as np
import tensorflow as tf
import time
import sys
import json
from os.path import splitext, join, dirname, basename
from tensorflow.python.training.training import Features, Example
from ase import Atoms
from scipy.special import comb
from itertools import combinations, product, chain, repeat
from collections import Counter, namedtuple
from sklearn.metrics import pairwise_distances

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# The ghost (virtual) atom
GHOST = "X"

# The pyykko radius of each element.
pyykko = {
  'Ac': 1.86, 'Ag': 1.28, 'Al': 1.26, 'Am': 1.66, 'Ar': 0.96, 'As': 1.21,
  'At': 1.47, 'Au': 1.24, 'B': 0.85, 'Ba': 1.96, 'Be': 1.02, 'Bh': 1.41,
  'Bi': 1.51, 'Bk': 1.68, 'Br': 1.14, 'C': 0.75, 'Ca': 1.71, 'Cd': 1.36,
  'Ce': 1.63, 'Cf': 1.68, 'Cl': 0.99, 'Cm': 1.66, 'Co': 1.11, 'Cr': 1.22,
  'Cs': 2.32, 'Cu': 1.12, 'Db': 1.49, 'Ds': 1.28, 'Dy': 1.67, 'Er': 1.65,
  'Es': 1.65, 'Eu': 1.68, 'F': 0.64, 'Fe': 1.16, 'Fm': 1.67, 'Fr': 2.23,
  'Ga': 1.24, 'Gd': 1.69, 'Ge': 1.21, 'H': 0.32, 'He': 0.46, 'Hf': 1.52,
  'Hg': 1.33, 'Ho': 1.66, 'Hs': 1.34, 'I': 1.33, 'In': 1.42, 'Ir': 1.22,
  'K': 1.96, 'Kr': 1.17, 'La': 1.8, 'Li': 1.33, 'Lu': 1.62, 'Md': 1.73,
  'Mg': 1.39, 'Mn': 1.19, 'Mo': 1.38, 'Mt': 1.29, 'N': 0.71, 'Na': 1.55,
  'Nb': 1.47, 'Nd': 1.74, 'Ne': 0.67, 'Ni': 1.1, 'No': 1.76, 'Np': 1.71,
  'O': 0.63, 'Os': 1.29, 'P': 1.11, 'Pa': 1.69, 'Pb': 1.44, 'Pd': 1.2,
  'Pm': 1.73, 'Po': 1.45, 'Pr': 1.76, 'Pt': 1.23, 'Pu': 1.72, 'Ra': 2.01,
  'Rb': 2.1, 'Re': 1.31, 'Rf': 1.57, 'Rh': 1.25, 'Rn': 1.42, 'Ru': 1.25,
  'S': 1.03, 'Sb': 1.4, 'Sc': 1.48, 'Se': 1.16, 'Sg': 1.43, 'Si': 1.16,
  'Sm': 1.72, 'Sn': 1.4, 'Sr': 1.85, 'Ta': 1.46, 'Tb': 1.68, 'Tc': 1.28,
  'Te': 1.36, 'Th': 1.75, 'Ti': 1.36, 'Tl': 1.44, 'Tm': 1.64, 'U': 1.7,
  'V': 1.34, 'W': 1.37, 'Xe': 1.31, 'Y': 1.63, 'Yb': 1.7,
  'Zn': 1.18, 'Zr': 1.54,
  # X represents virtual atoms
  GHOST: 0.32,
}

# Important: this `_safe_log` must be very very small. The previous e^-6 is
# large enough to cause significant numeric errors.
_safe_log = np.e**(-20)


# A data structure for storing transformed features and auxiliary parameters.
KcnnSample = namedtuple("KcnnSample", (
  "features",
  "split_dims",
  "binary_weights",
  "occurs",
  "compress_stats"
))


# A data structure for storing decoded TFRecord examples.
EnergyExample = namedtuple("EnergyExample", (
  "features",
  "energy",
  "occurs",
  "weights",
  "y_weight",
))


def get_formula(species):
  """
  Return the molecular formula given a list of atomic species.
  """
  return "".join(species)


def get_atoms_from_kbody_term(kbody_term):
  """
  Return the atoms in the given k-body term.

  Args:
    kbody_term: a `str` as the k-body term.

  Returns:
    atoms: a `list` of `str` as the chemical symbols of the atoms.

  """
  sel = [0]
  for i in range(len(kbody_term)):
    if kbody_term[i].isupper():
      sel.append(i + 1)
    else:
      sel[-1] += 1
  atoms = []
  for i in range(len(sel) - 1):
    atoms.append(kbody_term[sel[i]: sel[i + 1]])
  return atoms


def get_kbody_terms_from_species(species, k_max):
  """
  Return the k-body terms given the chemical symbols and `many_body_k`.

  Args:
    species: a `list` of `str` as the chemical symbols.
    k_max: a `int` as the maximum k-body terms that we should consider.

  Returns:
    kbody_terms: a `list` of `str` as the k-body terms.

  """
  return sorted(list(set(
      ["".join(sorted(c)) for c in combinations(species, k_max)])))


def _get_pyykko_bonds_matrix(species, factor=1.0, flatten=True):
  """
  Return the pyykko-bonds matrix given a list of atomic symbols.

  Args:
    species: a `List[str]` as the atomic symbols.
    factor: a `float` as the normalization factor.
    flatten: a `bool` indicating whether the bonds matrix is flatten or not.

  Returns:
    bonds: the bonds matrix (or vector if `flatten` is True).

  """
  rr = np.asarray([pyykko[specie] for specie in species])[:, np.newaxis]
  lmat = np.multiply(factor, rr + rr.T)
  if flatten:
    return lmat.flatten()
  else:
    return lmat


def exponential_norm(x, unit=1.0, order=1):
  """
  Normalize the inputs `x` with the exponential function:
    f(x) = exp(-x / unit)

  Args:
    x: Union[float, np.ndarray] as the inputs to scale.
    unit: a `float` or an array with the same shape of `inputs` as the scaling
      factor(s).
    order: a `int` as the exponential order. If `order` is 0, the inputs will
      not be scaled by `factor`.

  Returns:
    scaled: the scaled unitless inputs.

  """
  if order == 0:
    return np.exp(-x)
  else:
    return np.exp(-(x / unit) ** order)


def safe_divide(a, b):
  """
  Safe division while ignoring / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0].

  References:
     https://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero

  """
  with np.errstate(divide='ignore', invalid='ignore'):
    c = np.true_divide(a, b)
    c[~ np.isfinite(c)] = 0  # -inf inf NaN
  return c


def _compute_lr_weights(coef, y, num_real_atom_types, factor=1.0):
  """
  Solve the linear equation system of Ax = b.

  Args:
    coef: a `float` array of shape `[num_examples, num_atom_types]`.
    y: a `float` array of shape `[num_examples, ]`.
    num_real_atom_types: an `int` as the number of atom types excluding the
      ghost atoms.
    factor: a `float` as a scaling factor for the weights.

  Returns:
    x: a `float` array of shape `[num_atom_types, ]` as the solution.

  """
  rank = np.linalg.matrix_rank(coef[:, :num_real_atom_types])
  diff = num_real_atom_types - rank

  # The coef matrix is full rank. So the linear equation system can be solved.
  if diff == 0:
    x = np.negative(np.dot(np.linalg.pinv(coef), y))

  # The rank is 1, so all structures have the same stoichiometry. Then all types
  # of atoms can be treated equally.
  elif rank == 1:
    x = np.negative(np.mean(y / coef[:, :num_real_atom_types].sum(axis=1)))

  else:
    raise ValueError(
      "Coefficients matrix rank {} of {} is not supported!".format(
        rank, num_real_atom_types))

  return x * factor


def _bytes_feature(value):
  """
  Convert the `value` to Protobuf bytes.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
  """
  Convert the `value` to Protobuf float32.
  """
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class KbagsTransformer:
  """
  Compute the `k-Bags` feature vector for a given `ase.Atoms` object.
  """

  def __init__(self, species, k_max=3, kbody_terms=None, split_dims=None,
               periodic=False, cutoff=None):
    """
    Initialization method.

    Args:
      species: a `List[str]` as the ordered atomic symboles.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      kbody_terms: a `List[str]` as the k-body terms.
      split_dims: a `List[int]` as the dimensions for splitting inputs. If this
        is given, the `kbody_terms` must also be set and their lengths should be
        equal.
      periodic: a `bool` indicating whether this transformer is used for
        periodic structures or not.
      cutoff: a `float` as the cutoff.

    """
    if split_dims is not None:
      assert len(split_dims) == len(kbody_terms)

    kbody_terms = kbody_terms or get_kbody_terms_from_species(species, k_max)
    num_ghosts = self._get_num_ghosts(species, k_max)
    mapping, selections = self._get_mapping(species, kbody_terms)

    # Internal initialization.
    offsets, real_dim, kbody_sizes = [0], 0, []
    if split_dims is None:
      # If `split_dims` is not given, we shall construct it by our own.
      # `real_dim` should be always not smaller than `sum(kbody_sizes)` because
      # every excluded k-body term is represented by a single row vector of all
      # zeros.
      for kbody_term in kbody_terms:
        if kbody_term in mapping:
          size = mapping[kbody_term].shape[1]
        else:
          size = 0
        real_dim += max(size, 1)
        kbody_sizes.append(size)
        offsets.append(real_dim)
      split_dims = np.diff(offsets).tolist()
    else:
      offsets = [0] + np.cumsum(split_dims).tolist()
      real_dim = offsets[-1]
      kbody_sizes = []
      for kbody_term in kbody_terms:
        if kbody_term in mapping:
          size = mapping[kbody_term].shape[1]
        else:
          size = 0
        kbody_sizes.append(size)

    # Initialize internal variables.
    self._k_max = k_max
    self._kbody_terms = kbody_terms
    self._offsets = offsets
    self._kbody_sizes = kbody_sizes
    self._species = species
    self._mapping = mapping
    self._selections = selections
    self._split_dims = split_dims
    self._ck2 = int(comb(k_max, 2, exact=True))
    self._cond_sort = self._get_conditional_sorting_indices(kbody_terms)
    self._cmatrix = _get_pyykko_bonds_matrix(species)
    self._num_ghosts = num_ghosts
    self._periodic = periodic
    self._real_dim = real_dim
    self._binary_weights = self._get_binary_weights()
    self._norm_fn = exponential_norm
    self._cutoff = cutoff or np.inf
    self._cutoff_table = self._get_cutoff_table()

  @property
  def species(self):
    """
    Return the species of this transformer excluding all ghosts.
    """
    return [symbol for symbol in self._species if symbol != GHOST]

  @property
  def shape(self):
    """
    Return the shape of the transformed input feature matrix.
    """
    return self._real_dim, self._ck2

  @property
  def ck2(self):
    """
    Return the value of C(k,2) for this transformer.
    """
    return self._ck2

  @property
  def k_max(self):
    """
    Return the maximum order for the many-body expansion.
    """
    return self._k_max

  @property
  def split_dims(self):
    """
    Return the dims for spliting the inputs.
    """
    return self._split_dims

  @property
  def kbody_terms(self):
    """
    Return the kbody terms for this transformer.
    """
    return self._kbody_terms

  @property
  def kbody_sizes(self):
    """
    Return the real sizes of each kbody term of this transformer. Typically this
    is equal to `split_dims` but when `kbody_terms` is manually set, this may be
    different.
    """
    return self._kbody_sizes

  @property
  def binary_weights(self):
    """
    Return the binary weights for the all k-body contribs.
    """
    return self._binary_weights

  @property
  def kbody_selections(self):
    """
    Return the kbody selections.
    """
    return self._selections

  @property
  def num_ghosts(self):
    """
    Return the number of ghosts atoms.
    """
    return self._num_ghosts

  @property
  def is_periodic(self):
    """
    Return True if this transformer is used for periodic structures.
    """
    return self._periodic

  @property
  def cutoff(self):
    """
    Return the cutoff.
    """
    return self._cutoff

  def get_bond_types(self):
    """
    Return the ordered bond types for each k-body term.
    """
    bonds = {}
    for kbody_term in self._kbody_terms:
      atoms = get_atoms_from_kbody_term(kbody_term)
      bonds[kbody_term] = ["-".join(ab) for ab in combinations(atoms, r=2)]
    return bonds

  def _get_cutoff_table(self):
    """
    Return the cutoff table.
    """
    table = np.zeros((self._real_dim, self._ck2), dtype=np.float32)
    cutoff = self._cutoff
    if cutoff < np.inf:
      thres = self._norm_fn(cutoff)
    else:
      thres = 0.0
    for i, kbody_term in enumerate(self._kbody_terms):
      istart, istop = self._offsets[i], self._offsets[i + 1]
      if GHOST in kbody_term:
        table[istart: istop, 0] = thres
      else:
        table[istart: istop, :] = thres
    return table

  @staticmethod
  def _get_num_ghosts(species, many_body_k):
    """
    Return and check the number of ghost atoms.

    Args:
      species: a `list` of `str` as the chemical symbols.
      many_body_k: a `int` as the maximum k-body terms that we should consider.

    Returns:
      num_ghosts: a `int` as the number of ghost atoms.

    """
    num_ghosts = list(species).count(GHOST)
    if num_ghosts != 0 and (num_ghosts > 2 or many_body_k - num_ghosts != 2):
      raise ValueError("The number of ghosts is wrong!")
    return num_ghosts

  @staticmethod
  def _get_mapping(species, kbody_terms):
    """
    Build the mapping from interatomic distance matrix of shape `[N, N]` to the
    input feature matrix of shape `[C(N, k), C(k, 2)]`.

    Args:
      species: a `list` of `str` as the ordered atomic symbols.
      kbody_terms: a `list` of `str` as the ordered k-body terms.

    Returns:
      mapping: a `Dict[str, Array]` as the mapping from the N-by-N interatomic
        distance matrix to the input feature matrix for each k-body term.
      selection: a `Dict[str, List[List[int]]]` as the indices of the k-atoms
        selections for each k-body term.

    """
    natoms = len(species)
    mapping = {}
    selections = {}

    # Determine the indices of each type of atom and store them in a dict.
    atom_index = {}
    for i in range(len(species)):
      atom = species[i]
      atom_index[atom] = atom_index.get(atom, []) + [i]

    for kbody_term in kbody_terms:
      # Extract atoms from this k-body term
      atoms = get_atoms_from_kbody_term(kbody_term)
      # Count the occurances of the atoms.
      counter = Counter(atoms)
      # If an atom appears more in the k-body term, we should discard this
      # k-body term. For example the `CH4` molecule can not have `CCC` or `CCH`
      # interactions.
      if any([counter[e] > len(atom_index.get(e, [])) for e in atoms]):
        continue
      # ck2 is the number of bond types in this k-body term.
      ck2 = int(comb(len(atoms), 2, exact=True))
      # Sort the atoms
      sorted_atoms = sorted(counter.keys())
      # Build up the k-atoms selection candidates. For each type of atom we draw
      # N times where N is equal to `counter[atom]`. Thus, the candidate list
      # can be constructed:
      # [[[1, 2], [1, 3], [1, 4], ...], [[8], [9], [10], ...]]
      # The length of the candidates is equal to the number of atom types.
      k_atoms_candidates = [
        [list(o) for o in combinations(atom_index[e], counter[e])]
        for e in sorted_atoms
      ]
      # Build up the k-atoms selections. First, we get the `product` (See Python
      # official document for more info), eg [[1, 2], [8]]. Then `chain` it to
      # get flatten lists, eg [[1, 2, 8]].
      k_atoms_selections = [list(chain(*o)) for o in
                            product(*k_atoms_candidates)]
      selections[kbody_term] = k_atoms_selections
      # cnk is the number of k-atoms selections.
      cnk = len(k_atoms_selections)
      # Construct the mapping from the interatomic distance matrix to the input
      # matrix. This procedure can greatly increase the transformation speed.
      # The basic idea is to fill the input feature matrix with broadcasting.
      # The N-by-N interatomic distance matrix is flatten to 1D vector. Then we
      # can fill the matrix like this:
      #   feature_matrix[:, col] = flatten_dist[[1,2,8,10,9,2,1,1]]
      mapping[kbody_term] = np.zeros((ck2, cnk), dtype=int)
      for i in range(cnk):
        for j, (vi, vj) in enumerate(combinations(k_atoms_selections[i], 2)):
          mapping[kbody_term][j, i] = vi * natoms + vj
    return mapping, selections

  @staticmethod
  def _get_conditional_sorting_indices(kbody_terms):
    """
    Generate the indices of the columns for the conditional sorting scheme.

    Args:
      kbody_terms: a `List[str]` as the ordered k-body terms.

    Returns:
      indices: a `dict` of indices for sorting along the last axis of the input
        features.

    """
    indices = {}
    for kbody_term in kbody_terms:
      # Extract atoms from this k-body term
      atoms = get_atoms_from_kbody_term(kbody_term)
      # All possible bonds from the given atom types
      bonds = list(combinations(atoms, r=2))
      n = len(bonds)
      counter = Counter(bonds)
      # If the bonds are unique, there is no need to sort because columns of the
      # formed feature matrix will not be interchangable.
      if max(counter.values()) == 1:
        continue
      # Determine the indices of duplicated bonds.
      indices[kbody_term] = []
      for bond, times in counter.items():
        if times > 1:
          indices[kbody_term].append([i for i in range(n) if bonds[i] == bond])
    return indices

  def _get_binary_weights(self):
    """
    Return the binary weights.
    """
    weights = np.zeros(self._real_dim, dtype=np.float32)
    offsets = self._offsets
    for i in range(len(self._split_dims)):
      weights[offsets[i]: offsets[i] + self._kbody_sizes[i]] = 1.0
    return weights

  def _get_coords(self, atoms):
    """
    Return the N-by-3 coordinates matrix for the given `ase.Atoms`.

    Notes:
      Auxiliary vectors may be appended if `num_ghosts` is non-zero.

    """
    if self._num_ghosts > 0:
      # Append `num_ghosts` rows of zeros to the positions. We can not directly
      # use `inf` because `pairwise_distances` and `get_all_distances` do not
      # support `inf`.
      aux_vecs = np.zeros((self._num_ghosts, 3))
      coords = np.vstack((atoms.get_positions(), aux_vecs))
    else:
      coords = atoms.get_positions()
    return coords

  def _get_interatomic_distances(self, coords, cell, pbc):
    """
    Return the interatomic distances matrix and its associated coordinates
    differences matrices.

    Returns:
      dists: a `float32` array of shape `[N, N]` where N is the number of atoms
        as the interatomic distances matrix.

    """

    if not self.is_periodic:
      dists = pairwise_distances(coords)
    else:
      atoms = Atoms(
        symbols=self._species,
        positions=coords,
        cell=cell,
        pbc=pbc
      )
      dists = atoms.get_all_distances(mic=True)
      del atoms

    # Manually set the distances between ghost atoms and real atoms to inf.
    if self._num_ghosts > 0:
      dists[:, -self._num_ghosts:] = np.inf
      dists[-self._num_ghosts:, :] = np.inf
    return dists

  def _assign(self, dists, features=None):
    """
    Assign the normalized distances to the input feature matrix and build the
    auxiliary matrices.

    Args:
      dists: a `float32` array of shape `[N**2, ]` as the scaled flatten
        interatomic distances matrix.
      features: a 2D `float32` array or None as the location into which the
        result is stored. If not provided, a new array will be allocated.

    Returns:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.

    """
    if features is None:
      features = np.zeros((self._real_dim, self._ck2))
    elif features.shape != self.shape:
      raise ValueError("The shape should be {}".format(self.shape))

    for i, kbody_term in enumerate(self._kbody_terms):
      if kbody_term not in self._mapping:
        continue
      # The index matrix was transposed because typically C(N, k) >> C(k, 2).
      # See `_get_mapping`.
      mapping = self._mapping[kbody_term]
      istart = self._offsets[i]
      # Manually adjust the step size because the offset length may be larger if
      # `split_dims` is fixed.
      istep = min(self._offsets[i + 1] - istart, mapping.shape[1])
      istop = istart + istep
      for k in range(self._ck2):
        features[istart: istop, k] = dists[mapping[k]]

    return features

  def _conditionally_sort(self, features):
    """
    Apply the conditional sorting algorithm.

    Args:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.

    Returns:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.

    """
    for i, kbody_term in enumerate(self._kbody_terms):
      if kbody_term not in self._mapping:
        continue
      for ix in self._cond_sort.get(kbody_term, []):
        z = features[self._offsets[i]: self._offsets[i + 1], ix]
        # `samples` is a 2D array, the Python advanced slicing will make the
        # returned `z` a copy but not a view. The shape of `z` is transposed.
        # So we should sort along axis 0 here!
        z.sort()
        features[self._offsets[i]: self._offsets[i + 1], ix] = z
    return features

  def compress(self, features):
    """
    Apply the soft compressing algorithm. The compression is implemented by
    adjusting the binary weights.

    Args:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.

    Returns:
      weights: a `float32` array as the updated binary weights.
      counter: a `dict` to count the number of kept contribs for each k-body
        term. This may be an empty `dict` indicating all contribs are kept.

    """
    if self._cutoff == np.inf:
      return self._binary_weights, {}

    else:
      results = np.sum(features >= self._cutoff_table, axis=1, dtype=int)
      weights = np.ones_like(self._binary_weights)
      weights[results < 3] = 0.0
      counter = {}
      for i, kbody_term in enumerate(self._kbody_terms):
        istart, istop = self._offsets[i], self._offsets[i + 1]
        counter[kbody_term] = np.sum(results[istart: istop] == 3)
      return weights, counter

  def transform(self, atoms, features=None):
    """
    Transform the given `ase.Atoms` object to an input feature matrix.

    Args:
      atoms: an `ase.Atoms` object.
      features: a 2D `float32` array or None as the location into which the
        result is stored. If not provided, a new array will be allocated.

    Returns:
      features: a `float32` array of shape `self.shape` as the input feature
        matrix.
      coef: a `float32` array as the coefficients matrix. The shape of `coef` is
        `[self.shape[0], self.shape[1] * 6]`.
      indexing: an `int` array of shape `[3N, C(N, k) * C(k, 2) * 2 / N]` as the
        indices of the entries for each atomic force component.

    """

    # Get the coordinates matrix.
    coords = self._get_coords(atoms)

    # Compute the interatomic distances. For non-periodic molecules we use the
    # faster method `pairwise_distances`.
    dists = self._get_interatomic_distances(
      coords, cell=atoms.get_cell(), pbc=atoms.get_pbc()
    )

    # Normalize the interatomic distances with the exponential function so that
    # shorter bonds have larger normalized weights.
    dists = dists.flatten()
    norm_dists = self._norm_fn(dists, unit=self._cmatrix.flatten())

    # Assign the normalized distances to the input feature matrix.
    features = self._assign(norm_dists, features=features)

    # Apply the conditional sorting algorithm
    features = self._conditionally_sort(features)

    # Convert the data types
    return features.astype(np.float32)


class MultiTransformer:
  """
  A flexible k-Bags transformer targeting on AxByCz ... molecular compositions.
  """

  def __init__(self, atom_types, k_max=3, max_occurs=None, include_all_k=True,
               periodic=False, cutoff=None):
    """
    Initialization method.

    Args:
      atom_types: a `List[str]` as the atomic species.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      max_occurs: a `Dict[str, int]` as the maximum appearances for a specie.
        If an atom is explicitly specied, it can appear infinity times.
      include_all_k: a `bool` indicating whether we shall include all k-body
        terms where k = 1, ..., k_max. If False, only k = 1 and k = k_max are
        considered.
      periodic: a `bool` indicating whether we shall use periodic boundary
        conditions.
      cutoff: a `float` as the cutoff.

    """
    # Make sure the ghost atom is always the last one!
    if include_all_k and k_max == 3:
      num_ghosts = 1
      atom_types = list(atom_types)
      if GHOST in atom_types:
        if atom_types[-1] != GHOST:
          atom_types.remove(GHOST)
          atom_types = sorted(atom_types) + [GHOST]
        else:
          atom_types = sorted(atom_types[:-1]) + [GHOST]
      else:
        atom_types = sorted(atom_types) + [GHOST]
    elif k_max > 3:
      raise ValueError("k_max > 3 is not supported!")
    else:
      num_ghosts = 0
      if GHOST in atom_types:
        raise ValueError("GHOST is not allowed when k_max == 2!")

    # Determine the species and maximum occurs.
    species = []
    max_occurs = {} if max_occurs is None else dict(max_occurs)
    max_occurs[GHOST] = num_ghosts
    for specie in atom_types:
      species.extend(list(repeat(specie, max_occurs.get(specie, k_max))))
      if specie not in max_occurs:
        max_occurs[specie] = np.inf

    self._include_all_k = include_all_k
    self._k_max = k_max
    self._atom_types = atom_types
    self._species = species
    self._num_atom_types = len(atom_types)
    self._num_ghosts = num_ghosts
    self._kbody_terms = get_kbody_terms_from_species(species, k_max)
    self._transformers = {}
    self._max_occurs = max_occurs
    self._periodic = periodic
    self._cutoff = cutoff

    # The global split dims is None so that internal `_Transformer` objects will
    # construct their own `splid_dims`.
    self._split_dims = None

  @property
  def k_max(self):
    """
    Return the many-body expansion factor.
    """
    return self._k_max

  @property
  def included_k(self):
    """
    Return the included k.
    """
    if self._include_all_k:
      return list(range(1, self._k_max + 1))
    else:
      return [1, self._k_max]

  @property
  def ck2(self):
    """
    Return the value of C(k,2).
    """
    return comb(self._k_max, 2, exact=True)

  @property
  def kbody_terms(self):
    """
    Return the ordered k-body terms for this transformer.
    """
    return self._kbody_terms

  @property
  def species(self):
    """
    Return the ordered species of this transformer.
    """
    return self._species

  @property
  def atom_types(self):
    """
    Return the supported atom types.
    """
    return self._atom_types

  @property
  def number_of_atom_types(self):
    """
    Return the number of atom types in this transformer.
    """
    return self._num_atom_types

  @property
  def include_all_k(self):
    """
    Return True if a standalone two-body term is included.
    """
    return self._include_all_k

  @property
  def is_periodic(self):
    """
    Return True if this is a periodic transformer.
    """
    return self._periodic

  @property
  def max_occurs(self):
    """
    Return the maximum occurances of each type of atom.
    """
    return self._max_occurs

  @property
  def cutoff(self):
    """
    Return the cutoff.
    """
    return self._cutoff

  def accept_species(self, species):
    """
    Return True if the given species can be handled.

    Args:
      species: a `List[str]` as the ordered species of a molecule.

    Returns:
      accepted: True if the given species can be handled by this transformer.

    """
    counter = Counter(species)
    return all(counter[e] <= self._max_occurs.get(e, 0) for e in counter)

  def _get_transformer(self, species):
    """
    Return the `Transformer` for the given list of species.

    Args:
      species: a `List[str]` as the atomic species.

    Returns:
      clf: a `Transformer`.

    """
    species = list(species) + [GHOST] * self._num_ghosts
    formula = get_formula(species)
    clf = self._transformers.get(
      formula, KbagsTransformer(species=species,
                                k_max=self._k_max,
                                kbody_terms=self._kbody_terms,
                                split_dims=self._split_dims,
                                periodic=self._periodic,
                                cutoff=self._cutoff)
    )
    self._transformers[formula] = clf
    return clf

  def transform_trajectory(self, trajectory):
    """
    Transform the given trajectory (a list of `ase.Atoms` with the same chemical
    symbols or an `ase.io.TrajectoryReader`) to input features.

    Args:
      trajectory: a `list` of `ase.Atoms` or a `ase.io.TrajectoryReader`. All
        objects should have the same chemical symbols.

    Returns:
      sample: a `KcnnSample` object.

    """
    ntotal = len(trajectory)
    assert ntotal > 0

    species = trajectory[0].get_chemical_symbols()
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    clf = self._get_transformer(species)
    split_dims = np.asarray(clf.split_dims)
    nrows, ncols = clf.shape
    features = np.zeros((ntotal, nrows, ncols), dtype=np.float32)

    occurs = np.zeros((ntotal, self._num_atom_types), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._atom_types.index(specie)
      if loc < 0 or loc >= self._num_atom_types:
        raise ValueError("The loc of {:s} is {:d}!".format(specie, loc))
      occurs[:, loc] = float(times)

    weights = np.zeros((ntotal, nrows), dtype=np.float32)
    compress_stats = {}

    for i, atoms in enumerate(trajectory):
      _, coef_, indexing_ = clf.transform(atoms, features=features[i])
      if self._cutoff is None:
        weights[i] = clf.binary_weights
      else:
        weights[i], stats = clf.compress(features[i])
        for k, v in stats.items():
          compress_stats[k] = max(compress_stats.get(k, 0), v)

    return KcnnSample(features=features,
                      split_dims=split_dims,
                      binary_weights=weights,
                      occurs=occurs,
                      compress_stats=compress_stats)

  def transform(self, atoms):
    """
    Transform a single `ase.Atoms` object to input features.

    Args:
      atoms: an `ase.Atoms` object as the target to transform.

    Returns:
      sample: a `KcnnSample` object.

    Raises:
      ValueError: if the `species` is not supported by this transformer.

    """
    species = atoms.get_chemical_symbols()
    if not self.accept_species(species):
      raise ValueError(
        "This transformer does not support {}!".format(get_formula(species)))

    clf = self._get_transformer(species)
    split_dims = np.asarray(clf.split_dims)
    features = clf.transform(atoms)
    occurs = np.zeros((1, self._num_atom_types), dtype=np.float32)
    for specie, times in Counter(species).items():
      loc = self._atom_types.index(specie)
      if loc < 0:
        raise ValueError("The loc of %s is -1!" % specie)
      occurs[0, loc] = float(times)
    if self._cutoff is not None:
      weights, compress_stats = clf.compress(features)
    else:
      weights = np.array(clf.binary_weights)
      compress_stats = {}
    return KcnnSample(features=features,
                      split_dims=split_dims,
                      binary_weights=weights,
                      occurs=occurs,
                      compress_stats=compress_stats)


class OneBodyCalculator:
  """
  A helper class to compute the initial one-body weights.
  """

  def __init__(self, atom_types, num_examples, algorithm='default', factor=1.0,
               include_perturbations=True):
    """
    Initialization method.

    Args:
      atom_types: a list of `str` as the types of atoms.
      num_examples: an `int` as the total number of examples.
      algorithm: a `str` as the algorithm to compute the one-body weights.
      factor: a `float` as the scaling factor as the one-body weights.
      include_perturbations: a `bool`. If True, the higher-order perturbations
        terms will be included in the coefficients matrix as well.

    """
    self.atom_types = atom_types
    self.num_atom_types = len(atom_types)
    if atom_types[-1] == GHOST:
      self.num_real_atom_types = self.num_atom_types - 1
    else:
      self.num_real_atom_types = self.num_atom_types
    self.minima = {}
    self.b = np.zeros((num_examples, ))
    self.algorithm = algorithm.lower()
    self.factor = factor
    self.include_perturbations = include_perturbations
    self.mp2 = self.num_real_atom_types
    self.mp3 = self.num_real_atom_types + 1
    if not include_perturbations:
      self.coef = np.zeros((num_examples, self.num_real_atom_types))
    else:
      self.coef = np.zeros((num_examples, self.num_real_atom_types + 2))

  def add(self, index, chemical_symbols, y_true):
    """
    Add an example.

    Args:
      index: an `int` as the index of this sample.
      chemical_symbols: a `list` of `str` as the chemical symbols of this
        example.
      y_true: a `float` as the total energy of this example.

    """
    counter = Counter(chemical_symbols)
    for loc, atom in enumerate(self.atom_types[:self.num_real_atom_types]):
      self.coef[index, loc] = counter[atom]
    if self.include_perturbations:
      self.coef[index, self.mp2] = comb(len(chemical_symbols), 2)
      self.coef[index, self.mp3] = comb(len(chemical_symbols), 3)
    self.b[index] = y_true
    sch = self.get_stoichiometry(self.coef[index, :self.num_real_atom_types])
    if sch not in self.minima or y_true < self.b[self.minima[sch]]:
      self.minima[sch] = index

  def compute(self):
    """
    Compute the one-body weights.
    """
    if self.algorithm == 'minimal':
      # Only select the values from the global minima.
      selected = np.ix_(list(self.minima.values()))
      coef = self.coef[selected]
      b = self.b[selected]
    else:
      coef = self.coef
      b = self.b
    # The size of `x` is always equal to `self.num_real_atom_types`. We may need
    # to pad an zero at the end.
    x = _compute_lr_weights(
      coef, b,
      num_real_atom_types=self.num_real_atom_types,
      factor=self.factor
    )
    x = np.resize(x, self.num_atom_types)
    x[self.num_real_atom_types:] = 0.0
    return x

  def get_stoichiometry(self, atoms_counts):
    """
    A helper function to get the stoichiometry of a structure.
    """
    return ";".join(["{},{}".format(self.atom_types[j], int(atoms_counts[j]))
                     for j in range(self.num_real_atom_types)])


class FixedLenMultiTransformer(MultiTransformer):
  """
  This is also a flexible k-Bags transformer targeting on AxByCz ... molecular
  compositions. But the length of the transformed features are the same so that
  they can be used for training.
  """

  def __init__(self, max_occurs, periodic=False, k_max=3, include_all_k=True,
               cutoff=None):
    """
    Initialization method.

    Args:
      max_occurs: a `Dict[str, int]` as the maximum appearances for each kind of
        atomic specie.
      periodic: a `bool` indicating whether this transformer is used for
        periodic structures or not.
      k_max: a `int` as the maximum k for the many body expansion scheme.
      include_all_k: a `bool` indicating whether we shall include all k-body
        terms where k = 1, ..., k_max. If False, only k = 1 and k = k_max are
        considered.
      cutoff: a `float` as the cutoff.

    """
    super(FixedLenMultiTransformer, self).__init__(
      atom_types=list(max_occurs.keys()),
      k_max=k_max,
      max_occurs=max_occurs,
      include_all_k=include_all_k,
      periodic=periodic,
      cutoff=cutoff
    )
    self._split_dims = self._get_fixed_split_dims()
    self._total_dim = sum(self._split_dims)

  @property
  def shape(self):
    """
    Return the shape of input feature matrix of this transformer.
    """
    return self._total_dim, comb(self._k_max, 2, exact=True)

  @property
  def split_dims(self):
    """
    Return the fixed `split_dims` for all internal transformers.
    """
    return self._split_dims

  def _get_fixed_split_dims(self):
    """
    The `split_dims` of all `_Transformer` should be the same.
    """
    split_dims = []
    for kbody_term in self._kbody_terms:
      atoms = get_atoms_from_kbody_term(kbody_term)
      counter = Counter(atoms)
      dims = [comb(self._max_occurs[e], k, True) for e, k in counter.items()]
      split_dims.append(np.prod(dims))
    return [int(x) for x in split_dims]

  def _log_compression_results(self, results):
    """
    A helper function to log the compression result.
    """
    print("The soft compression algorithm is applied with cutoff = "
          "{:.2f}".format(self._cutoff))

    num_loss_total = 0
    num_total = self._total_dim
    for j, kbody_term in enumerate(self._kbody_terms):
      num_full = self._split_dims[j]
      num_kept = results.get(kbody_term, 0)
      num_loss = num_full - num_kept
      num_loss_total += num_loss
      print("{:<12s} : {:5d} / {:5d}, compression = {:.2f}%".format(
        kbody_term, num_loss, num_full, num_loss / num_full * 100))
    print("Final result : {:5d} / {:5d}, compression = {:.2f}%".format(
      num_loss_total, num_total, num_loss_total / num_total * 100))

  def transform_and_save(self, filename, examples, num_examples,
                         loss_fn=None, verbose=True, one_body_kwargs=None):
    """
    Transform the given atomic coordinates to input features and save them to
    tfrecord files using `tf.TFRecordWriter`.

    Args:
      filename: a `str` as the file to save examples.
      examples: a iterator which iterates through all examples.
      num_examples: an `int` as the number of examples.
      verbose: boolean indicating whether.
      loss_fn: a `Callable` for transforming the calculated raw loss.
      one_body_kwargs: a `dict` as the key-value args for computing initial
        one-body weights.

    Returns:
      weights: a `float32` array as the weights for linear fit of the energies.

    """

    def _identity(_):
      """
      An identity function which returns the input directly.
      """
      return 1.0

    # Setup the loss function.
    loss_fn = loss_fn or _identity

    # Setup the one-body weights calculator
    one_body = OneBodyCalculator(
      self._atom_types, num_examples, **(one_body_kwargs or {}))

    # Start the timer
    tic = time.time()

    with tf.python_io.TFRecordWriter(filename) as writer:
      if verbose:
        print("Start transforming {} ... ".format(filename))

      compress_stats = {}

      for i, atoms in enumerate(examples):

        species = atoms.get_chemical_symbols()
        y_true = atoms.get_total_energy()
        sample = self.transform(atoms)

        x = _bytes_feature(sample.features.tostring())
        y = _bytes_feature(np.atleast_2d(-y_true).tostring())
        z = _bytes_feature(sample.occurs.tostring())
        w = _bytes_feature(sample.binary_weights.tostring())
        y_weight = _float_feature(loss_fn(y_true))

        example = Example(
          features=Features(feature={'energy': y, 'features': x, 'occurs': z,
                                     'weights': w, 'loss_weight': y_weight}))

        writer.write(example.SerializeToString())

        # Add this example to the one-body database
        one_body.add(i, species, y_true)

        # Save the compress stats for this example
        for k, v in sample.compress_stats.items():
          compress_stats[k] = max(compress_stats.get(k, 0), v)

        if verbose and (i + 1) % 100 == 0:
          sys.stdout.write("\rProgress: {:7d} / {:7d} | Speed = {:6.1f}".format(
            i + 1, num_examples, (i + 1) / (time.time() - tic)))

      if verbose:
        print("")
        print("Transforming {} finished!".format(filename))

        if self._cutoff is not None:
          self._log_compression_results(compress_stats)

      return one_body.compute()

  def save_auxiliary_for_file(self, filename, lookup_indices=None,
                              initial_one_body_weights=None):
    """
    Save auxiliary data for the given dataset.

    Args:
      filename: a `str` as the tfrecords file.
      initial_one_body_weights: a `float32` array of shape `[num_atom_types, ]`
        as the initial weights for the one-body convolution kernels.
      lookup_indices: a `List[int]` as the indices of each given example.

    """
    if lookup_indices is not None:
      lookup_indices = list(lookup_indices)
    else:
      lookup_indices = []

    if initial_one_body_weights is not None:
      initial_one_body_weights = list(initial_one_body_weights)
    else:
      initial_one_body_weights = []

    max_occurs = {atom: times for atom, times in self._max_occurs.items()
                  if times < self._k_max}

    auxiliary_properties = {
      "kbody_terms": self._kbody_terms,
      "split_dims": self._split_dims,
      "shape": self.shape,
      "lookup_indices": list([int(i) for i in lookup_indices]),
      "atom_types": self._atom_types,
      "num_atom_types": self._num_atom_types,
      "species": self._species,
      "include_all_k": self._include_all_k,
      "periodic": self._periodic,
      "k_max": self._k_max,
      "max_occurs": max_occurs,
      "initial_one_body_weights": initial_one_body_weights,
      "cutoff": self._cutoff
    }

    with open(join(dirname(filename),
                   "{}.json".format(splitext(basename(filename))[0])),
              "w+") as fp:
      json.dump(auxiliary_properties, fp=fp, indent=2)


def decode_protobuf(example_proto, cnk=None, ck2=None, num_atom_types=None):
  """
  Decode the protobuf into a tuple of tensors.

  Args:
    example_proto: A scalar string Tensor, a single serialized Example.
      See `_parse_single_example_raw` documentation for more details.
    cnk: an `int` as the value of C(N,k).
    ck2: an `int` as the value of C(k,2).
    num_atom_types: an `int` as the number of atom types.

  Returns:
    example: a decoded `TFExample` from the TFRecord file.

  """
  example = tf.parse_single_example(
    example_proto,
    # Defaults are not specified since both keys are required.
    features={
      'features': tf.FixedLenFeature([], tf.string),
      'energy': tf.FixedLenFeature([], tf.string),
      'occurs': tf.FixedLenFeature([], tf.string),
      'weights': tf.FixedLenFeature([], tf.string),
      'loss_weight': tf.FixedLenFeature([], tf.float32)
    })

  features = tf.decode_raw(example['features'], tf.float32)
  features.set_shape([cnk * ck2])
  features = tf.reshape(features, [1, cnk, ck2])

  energy = tf.decode_raw(example['energy'], tf.float64)
  energy.set_shape([1])
  energy = tf.squeeze(energy)

  occurs = tf.decode_raw(example['occurs'], tf.float32)
  occurs.set_shape([num_atom_types])
  occurs = tf.reshape(occurs, [1, 1, num_atom_types])

  weights = tf.decode_raw(example['weights'], tf.float32)
  weights.set_shape([cnk, ])
  weights = tf.reshape(weights, [1, cnk, 1])

  y_weight = tf.cast(example['loss_weight'], tf.float32)

  return EnergyExample(features=features,
                       energy=energy,
                       occurs=occurs,
                       weights=weights,
                       y_weight=y_weight)
