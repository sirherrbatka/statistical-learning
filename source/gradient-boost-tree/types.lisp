(cl:in-package #:statistical-learning.gradient-boost-tree)


(defclass fundamental-gradient-boost-tree-parameters
    (sl.tp:standard-tree-training-parameters)
  ((%optimized-function :initarg :optimized-function
                        :reader sl.opt:optimized-function
                        :reader optimized-function)))


(defclass classification (sl.perf:classification
                          fundamental-gradient-boost-tree-parameters)
  ())


(defclass regression (sl.perf:regression
                      fundamental-gradient-boost-tree-parameters)
  ())


(defclass gradient-boosting-implementation (sl.dt:regression)
  ((%gradient-parameters :initarg :gradient-parameters
                         :reader gradient-parameters)
   (%expected-value :initarg :expected-value
                    :reader expected-value)
   (%shrinkage :initarg :shrinkage
               :reader shrinkage))
  (:default-initargs :optimized-function sl.opt:<squared-error>))


(defclass classification-implementation (gradient-boosting-implementation)
  ())


(defclass regression-implementation (gradient-boosting-implementation)
  ())


(defclass gradient-boost-model (statistical-learning.tp:tree-model)
  ((%expected-value :initarg :expected-value
                    :reader expected-value)
   (%shrinkage :initarg :shrinkage
               :reader shrinkage)))
