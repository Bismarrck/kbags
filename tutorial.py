# coding=utf-8
"""
A simple tutorial of the k-bags transformer.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
from ase.db import connect
from kbags import FixedLenMultiTransformer, decode_protobuf
from collections import Counter
from functools import partial

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'

# Connect to an `ase.db`.
db = connect('../datasets/qm7.db')

# Loop through all `ase.Atoms` and compute the `max_occurs`.
examples = []
max_occurs = {}

for row in db.select('id>=1'):
  atoms = row.toatoms()
  c = Counter(atoms.get_chemical_symbols())
  for symbol, times in c.items():
    max_occurs[symbol] = max(max_occurs.get(symbol, 0), times)
  examples.append(atoms)

# Due to the `GHOST`, we need add one to `num_atom_types`.
num_real_atom_types = len(max_occurs.keys())
num_atom_types = num_real_atom_types + 1

# Initialize a fixed-length transformer
clf = FixedLenMultiTransformer(max_occurs)

# Transform all `ase.Atoms` to k-Bags features and compute the initial one-body
# weights using the RMS algorithm.
weights = clf.transform_and_save("qm7_all.tfrecords", examples, len(examples))

# Save the auxiliary info to a JSON file.
clf.save_auxiliary_for_file(
  "qm7_all.json",
  initial_one_body_weights=weights,
  lookup_indices=list(range(len(db)))
)

# Decode the transformed TFRecords.
dataset = tf.data.TFRecordDataset(["qm7_all.tfrecords"])
dataset = dataset.map(
  partial(decode_protobuf, cnk=4495, ck2=3, num_atom_types=num_atom_types),
  num_parallel_calls=64
)

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
  result = sess.run(next_element)
  print(result.energy)
  print(result.features)
