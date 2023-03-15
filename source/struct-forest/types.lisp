(cl:in-package #:statistical-learning.struct-forest)


(defclass struct-state (sl.tp:tree-training-state)
  ((%relabaled :initform nil
               :accessor relabaled)))


(defclass struct-training-implementation (sl.dt:classification
                                          sl.tp:basic-tree-training-parameters)
  ((%original :initarg :original
              :reader original))
  (:default-initargs
   :optimized-function (sl.opt:gini-impurity 2)))


(defclass euclid-distance-relabaler ()
  ((%relabel-iterations :initarg :relabel-iterations
                        :reader relabel-iterations)
   (%relabel-repeats :initarg :relabel-repeats
                     :reader relabel-repeats))
  (:default-initargs
   :relabel-iterations 4
   :relabel-repeats 4))


(defclass struct (sl.perf:regression
                  sl.tp:standard-tree-training-parameters)
  ((%optimized-function :initarg :optimized-function
                        :reader sl.opt:optimized-function)
   (%relabeler :initarg :relabeler
               :reader relabeler))
  (:default-initargs
   :relabeler (make 'euclid-distance-relabaler)
   :splitter (make 'sl.tp:random-attribute-splitter)
   :optimized-function (sl.opt:squared-error)))
