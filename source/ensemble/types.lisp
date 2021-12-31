(cl:in-package #:statistical-learning.ensemble)


(defclass fundamental-pruning-algorithm ()
  ())


(defclass fundamental-refinement-algorithm ()
  ())


(defclass fundamental-weights-calculator ()
  ())


(defclass static-weights-calculator (fundamental-weights-calculator)
  ())


(defclass fundamental-data-points-sampler ()
  ())


(defclass weights-based-data-points-sampler (fundamental-data-points-sampler)
  ((%sampling-rate :initarg :sampling-rate
                   :reader sampling-rate)))


(defclass gradient-based-one-side-sampler (fundamental-data-points-sampler)
  ((%large-gradient-sampling-rate :initarg :large-gradient-sampling-rate
                                  :reader large-gradient-sampling-rate)
   (%small-gradient-sampling-rate :initarg :small-gradient-sampling-rate
                                  :reader small-gradient-sampling-rate)))


(defclass dynamic-weights-calculator (fundamental-weights-calculator)
  ())


(defclass dynamic-weights-calculator-state ()
  ((%counts :initarg :counts
            :accessor counts))
  (:default-initargs
   :counts nil))


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
   (%tree-parameters :initarg :tree-parameters
                     :reader tree-parameters)))


(defclass isolation-forest (ensemble)
  ((%tree-sample-rate :initarg :tree-sample-rate
                      :reader tree-sample-rate)))


(defclass random-forest (ensemble)
  ((%weights-calculator :initarg :weights-calculator
                        :reader weights-calculator)
   (%pruning :initarg :pruning
             :reader pruning)
   (%refinement :initarg :refinement
                :reader refinement)
   (%data-points-sampler :initarg :data-points-sampler
                         :reader data-points-sampler))
  (:default-initargs
   :pruning nil
   :refinement nil
   :weights-calculator (make 'static-weights-calculator)
   :data-points-sampler (make 'weights-based-data-points-sampler :sampling-rate 0.1d0)))


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
             :reader indexes)
   (%sampler-state :initarg :sampler-state
                   :accessor sampler-state))
  (:default-initargs
   :samples-view nil
   :attributes-view nil
   :sampler-state nil
   :trees-view nil))


(defclass supervised-ensemble-state (ensemble-state)
  ((%target-data :initarg :target-data
                 :reader sl.mp:target-data)
   (%leafs-assigned-p :initarg :leafs-assigned-p
                      :accessor leafs-assigned-p
                      :documentation "For the weights calculator.")
   (%assigned-leafs :initarg :assigned-leafs
                    :accessor assigned-leafs
                    :documentation "For the weights calculator"))
  (:default-initargs
   :leafs-assigned-p nil))


(defclass random-forest-state (supervised-ensemble-state)
  ((%weights :initarg :weights
             :accessor sl.mp:weights)
   (%weights-calculator-state :initarg :weights-calculator-state
                              :accessor weights-calculator-state))
  (:default-initargs
   :weights nil
   :weights-calculator-state nil))


(defclass gradient-boost-ensemble-state-mixin ()
  ((%gradients :initarg :gradients
               :accessor gradients))
  (:default-initargs :gradients nil))


(defclass gradient-boost-ensemble-state (gradient-boost-ensemble-state-mixin
                                         supervised-ensemble-state)
  ())


(defclass supervised-gradient-boost-ensemble-state (gradient-boost-ensemble-state-mixin
                                                    supervised-ensemble-state)
  ())


(defclass isolation-forest-ensemble-state (ensemble-state)
  ())


(defclass gradient-boost-ensemble (ensemble)
  ((%shrinkage :initarg :shrinkage
               :reader shrinkage)
   (%pruning :initarg :pruning
             :reader pruning)
   (%data-points-sampler :initarg :data-points-sampler
                         :reader data-points-sampler))
  (:default-initargs
   :pruning nil
   :shrinkage 0.01d0
   :data-points-sampler (make 'weights-based-data-points-sampler :sampling-rate 0.1d0)))


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


(defmethod make-data-points-sampler-state ((data-point-sampler fundamental-data-points-sampler)
                                           ensemble-state)
  nil)
