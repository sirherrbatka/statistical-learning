(cl:in-package #:statistical-learning.data)


(deftype data-matrix ()
  '(simple-array double-float (* *)))


(deftype split-vector ()
  'simple-vector)
