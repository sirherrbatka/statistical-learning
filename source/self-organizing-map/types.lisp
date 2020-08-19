(cl:in-package #:sl.som)


(defclass self-organizing-map (sl.mp:fundamental-model-parameters)
  ((%initial-alpha
    :initarg :initial-alpha
    :reader initial-alpha)
   (%initial-sigma
    :initarg :initial-sigma
    :reader initial-sigma)
   (%decay
    :initarg :decay
    :reader decay)
   (%grid-dimensions
    :initarg :grid-dimensions
    :reader grid-dimensions)
   (%parallel
    :initarg :parallel
    :reader parallel)
   (%number-of-iterations
    :initarg :number-of-iterations
    :reader number-of-iterations))
  (:default-initargs :parallel nil))


(defclass self-organizing-map-training-state (sl.mp:fundamental-training-state)
  ((%data
    :initarg :data
    :accessor data)
   (%all-distances
    :initarg :all-distances
    :reader all-distances)
   (%all-indexes
    :initarg :all-indexes
    :reader all-indexes)
   (%units
    :initarg :units
    :accessor units)
   (%weights
    :initarg :weights
    :accessor weights)))


(defclass self-organizing-map-model (sl.mp:fundamental-model)
  ((%units :initarg :units
           :reader units)))


(defclass decay ()
  ())


(defclass hill-decay (decay)
  ())


(defclass linear-decay (decay)
  ())


(deftype unit ()
  '(simple-array double-float (*)))


(deftype grid ()
  '(simple-array unit))
