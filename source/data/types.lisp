(cl:in-package #:statistical-learning.data)


(defstruct double-float-data-matrix
  (data (make-array '(0 0) :element-type 'double-float) :type (simple-array double-float (* *)))
  (index (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*))))


(defstruct universal-data-matrix
  (data (make-array '(0 0) :element-type t) :type (simple-array t (* *)))
  (index (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*))))


(deftype data-matrix ()
  '(or double-float-data-matrix universal-data-matrix))


(deftype split-vector ()
  'simple-vector)
