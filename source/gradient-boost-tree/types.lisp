(cl:in-package #:statistical-learning.gradient-boost-tree)


(defclass fundamental-gradient-boost-tree-parameters
    (sl.tp:fundamental-tree-training-parameters)
  ((%optimized-function :initarg :optimized-function
                        :reader optimized-function)))


(defclass classification (sl.perf:classification
                          fundamental-gradient-boost-tree-parameters)
  ())


(defclass regression (sl.perf:regression
                      fundamental-gradient-boost-tree-parameters)
  ())


(defclass classification-implementation (sl.dt:regression)
  ()
  (:default-initargs :optimized-function sl.opt:<squared-error>))


(defclass regression-implementation (sl.dt:regression)
  ()
  (:default-initargs :optimized-function sl.opt:<squared-error>))


(defclass gradient-boost-model (statistical-learning.tp:tree-model)
  ((%expected-value :initarg :expected-value
                    :reader expected-value)
   (%shrinkage :initarg :shrinkage
               :reader shrinkage)))
