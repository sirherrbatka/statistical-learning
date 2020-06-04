(cl:in-package #:cl-grf.forest)


(defclass random-forest-parameters ()
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


(defclass classification-random-forest-parameters (random-forest-parameters)
  ())


(defclass regression-random-forest-parameters (random-forest-parameters)
  ())


(defclass fundamental-random-forest (cl-grf.mp:fundamental-model)
  ((%trees :initarg :trees
           :reader trees
           :type simple-vector)
   (%target-attributes-count :initarg :target-attributes-count
                             :reader target-attributes-count)
   (%attributes :initarg :attributes
                :reader attributes
                :type simple-vector)))


(defclass classification-random-forest (fundamental-random-forest)
  ())


(defclass regression-random-forest (fundamental-random-forest)
  ())
