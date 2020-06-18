(cl:in-package #:statistical-learning.ensemble)


(defclass ensemble (sl.mp:fundamental-model-parameters)
  ((%trees-count :initarg :trees-count
                 :reader trees-count
                 :type positive-integer)
   (%parallel :initarg :parallel
              :reader parallel)
   (%tree-attributes-count :initarg :tree-attributes-count
                           :reader tree-attributes-count)
   (%tree-batch-size :initarg :tree-batch-size
                     :reader tree-batch-size
                     :type positive-integer)
   (%tree-sample-rate :initarg :tree-sample-rate
                      :reader tree-sample-rate)
   (%tree-parameters :initarg :tree-parameters
                     :reader tree-parameters)))


(defclass random-forest (ensemble)
  ())


(defclass gradient-boost-ensemble (ensemble)
  ((%shrinkage :initarg :shrinkage
               :reader shrinkage)
   (%shrinkage-change :initarg :shrinkage-change
                      :reader shrinkage-change))
  (:default-initargs
   :shrinkage 0.1d0
   :shrinkage-change 0.0d0))


(defclass ensemble-model (statistical-learning.mp:fundamental-model)
  ((%trees :initarg :trees
           :reader trees
           :type simple-vector)
   (%target-attributes-count :initarg :target-attributes-count
                             :reader target-attributes-count)))


(defclass random-forest-model (ensemble-model)
  ())


(defclass gradient-boost-ensemble-model (ensemble-model)
  ())
