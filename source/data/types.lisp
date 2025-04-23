(cl:in-package #:statistical-learning.data)


(defstruct single-float-data-matrix
  (data (make-array '(0 0) :element-type 'single-float) :type (simple-array single-float (* *)))
  (index (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (missing-mask (make-array '(0 0) :element-type 'bit :initial-element 1) :type (simple-array bit (* *))))


(defstruct universal-data-matrix
  (data (make-array '(0 0) :element-type t) :type (simple-array t (* *)))
  (index (make-array 0 :element-type 'fixnum) :type (simple-array fixnum (*)))
  (missing-mask (make-array '(0 0) :element-type 'bit :initial-element 1) :type (simple-array bit (* *))))


(deftype data-matrix ()
  '(or single-float-data-matrix universal-data-matrix))


(deftype split-vector ()
  'simple-vector)
