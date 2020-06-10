(cl:in-package #:cl-grf.ensemble)


(defclass ensemble-parameters ()
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


(defclass random-forest-parameters (ensemble-parameters)
  ())


(defclass gradient-boost-ensemble-parameters (random-forest-parameters)
  ((%learning-rate :initarg :learning-rate
                   :reader learning-rate)
   (%learning-rate-change :initarg :learning-rate-change
                          :reader learning-rate-change))
  (:default-initargs
   :learning-rate 0.1d0
   :learning-rate-change 0.0d0))


(defclass ensemble (cl-grf.mp:fundamental-model)
  ((%trees :initarg :trees
           :reader trees
           :type simple-vector)
   (%target-attributes-count :initarg :target-attributes-count
                             :reader target-attributes-count)))


(defclass random-forest (ensemble)
  ())


(defclass gradient-boost-ensemble (ensemble)
  ())
