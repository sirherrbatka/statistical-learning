(cl:in-package #:statistical-learning.ensemble)


(defclass fundamental-weights-calculator ()
  ((%weights :initarg :weights
             :reader weights)
   (%ensemble :initarg :ensemble
              :reader ensemble)
   (%train-data :initarg :train-data
                :reader train-data)
   (%target-data :initarg :target-data
                 :reader target-data)
   (%parallel :initarg :parallel
              :reader parallel)))


(defclass dynamic-weights-calculator (fundamental-weights-calculator)
  ((%counts :initarg :counts
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


(defclass isolation-forest (ensemble)
  ())


(defclass random-forest (ensemble)
  ((%weights-calculator-class :initarg :weights-calculator-class
                              :reader weights-calculator-class))
  (:default-initargs
   :weights-calculator-class 'static-weights-calculator))


(defclass ensemble-state (sl.mp:fundamental-training-state)
  ((%all-args :initarg :all-args
              :reader all-args)
   (%parameters :initarg :parameters
                :reader sl.mp:parameters)
   (%trees :initarg :trees
           :accessor trees)
   (%attributes :initarg :attributes
                :reader attributes)
   (%samples :initarg :samples
             :reader samples)
   (%trees-view :initarg :trees-view
                :accessor trees-view)
   (%attributes-view :initarg :attributes-view
                     :accessor attributes-view)
   (%samples-view :initarg :samples-view
                  :accessor samples-view)
   (%train-data :initarg :train-data
                :reader sl.mp:train-data)
   (%indexes :initarg :indexes
             :reader indexes))
  (:default-initargs
   :samples-view nil
   :attributes-view nil
   :trees-view nil))


(defclass supervised-ensemble-state (ensemble-state)
  ((%target-data :initarg :target-data
                 :reader sl.mp:target-data)
   (%leafs-assigned-p :initarg :leafs-assigned-p
                      :accessor leafs-assigned-p)
   (%assigned-leafs :initarg :assigned-leafs
                    :accessor assigned-leafs
                    :documentation "For the weights calculator")
   (%weights :initarg :weights
             :accessor sl.mp:weights))
  (:default-initargs
   :leafs-assigned-p nil))


(defclass isolation-forest-ensemble-state (ensemble-state)
  ())


(defclass gradient-boost-ensemble (ensemble)
  ((%shrinkage :initarg :shrinkage
               :reader shrinkage))
  (:default-initargs :shrinkage 0.01d0))


(defclass ensemble-model (sl.mp:fundamental-model)
  ((%trees :initarg :trees
           :reader trees
           :type simple-vector)))


(defclass random-forest-model (sl.mp:supervised-model
                               ensemble-model)
  ((%target-attributes-count :initarg :target-attributes-count
                             :reader target-attributes-count)))


(defclass gradient-boost-ensemble-model (sl.mp:supervised-model
                                         ensemble-model)
  ((%target-attributes-count :initarg :target-attributes-count
                             :reader target-attributes-count)))


(defclass isolation-forest-model (ensemble-model)
  ())
