(cl:in-package #:statistical-learning.data)


(deftype double-float-data-matrix ()
  '(simple-array double-float (* *)))


(deftype universal-data-matrix ()
  '(simple-array t (* *)))


(deftype data-matrix ()
  '(or double-float-data-matrix universal-data-matrix))


(deftype split-vector ()
  'simple-vector)
