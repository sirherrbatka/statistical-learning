(cl:in-package #:sl.som)


(defclass fundamental-matching-unit-selector ()
  ())


(defclass euclid-matching-unit-selector (fundamental-matching-unit-selector)
  ())


(defclass fundamental-self-organizing-map (sl.mp:fundamental-model-parameters)
  ())


(defclass units-container ()
  ((%data :initarg :data
          :reader data)
   (%index :initarg :index
           :reader index)
   (%units :initarg :units
           :reader units)
   (%parameters :initarg :parameters
                :reader sl.mp:parameters)))


(defclass units-container-with-unit-leafs (units-container)
  ((%unit-leafs :initarg :unit-leafs
                :reader unit-leafs)))


(defclass decay ()
  ())


(defclass hill-decay (decay)
  ())


(defclass linear-decay (decay)
  ())


(def <linear-decay> (make 'linear-decay))
(def <hill-decay> (make 'hill-decay))
(def <euclid-matching-unit-selector> (make 'euclid-matching-unit-selector))


(defclass abstract-self-organizing-map (fundamental-self-organizing-map)
  ((%initial-alpha
    :initarg :initial-alpha
    :reader initial-alpha)
   (%random-ranges
    :initarg :random-ranges
    :reader random-ranges)
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
  (:default-initargs
   :random-ranges nil
   :parallel nil))


(defclass self-organizing-map (abstract-self-organizing-map)
  ((%matching-unit-selector
    :initarg :matching-unit-selector
    :reader matching-unit-selector))
  (:default-initargs :matching-unit-selector <euclid-matching-unit-selector>))


(defclass random-forest-self-organizing-map (abstract-self-organizing-map)
  ((%forest :initarg :forest
            :reader forest)))


(defclass self-organizing-map-training-state (sl.mp:fundamental-training-state)
  ((%initial-sigma
    :initarg :initial-sigma
    :reader initial-sigma)
   (%data
    :initarg :data
    :reader sl.mp:train-data)
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


(defclass random-forest-self-organizing-map-model (self-organizing-map)
  ((%unit-leafs :initarg :unit-leafs
                :reader unit-leafs)
   (%units-leafs :initarg :units-leafs
                 :reader units-leafs)))


(deftype unit ()
  '(simple-array double-float (*)))


(deftype grid ()
  '(simple-array unit))
