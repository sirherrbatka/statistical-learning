(cl:in-package #:statistical-learning.ensemble)


(defclass fundamental-weights-calculator ()
  ((%weights :initarg :weights
             :reader weights)
   (%train-data :initarg :train-data
                :reader train-data)
   (%target-data :initarg :target-data
                 :reader target-data)
   (%parallel :initarg :parallel
              :reader parallel)))


(defclass dynamic-weights-calculator (fundamental-weights-calculator)
  ((%indexes :initarg :indexes
             :accessor indexes)
   (%counts :initarg :counts
            :accessor counts))
  (:default-initargs
   :indexes nil
   :counts nil))


(defclass static-weights-calculator (fundamental-weights-calculator)
  ())


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
  ((%weights-calculator-class :initarg :weights-calculator-class
                              :reader weights-calculator-class))
  (:default-initargs
   :weights-calculator-class 'static-weights-calculator))


(defclass ensemble-state (sl.mp:fundamental-training-state)
  ((%all-args :initarg :all-args
              :reader all-args)
   (%train-data :initarg :train-data
                :reader sl.mp:train-data)
   (%target-data :initarg :target-data
                 :reader sl.mp:target-data)
   (%weights :initarg :weights
             :reader sl.mp:weights)))


(defclass gradient-boost-ensemble (ensemble)
  ((%shrinkage :initarg :shrinkage
               :reader shrinkage))
  (:default-initargs :shrinkage 0.01d0))


(defclass ensemble-model (statistical-learning.mp:supervised-model)
  ((%trees :initarg :trees
           :reader trees
           :type simple-vector)
   (%target-attributes-count :initarg :target-attributes-count
                             :reader target-attributes-count)))


(defclass random-forest-model (ensemble-model)
  ())


(defclass gradient-boost-ensemble-model (ensemble-model)
  ())
