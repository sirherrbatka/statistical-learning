(cl:in-package #:statistical-learning.tree-protocol)


(defclass split-result ()
  ((%split-vector :initarg :split-vector
                  :reader split-vector)
   (%split-point :initarg :split-point
                 :reader split-point)
   (%left-length :initarg :left-length
                 :reader left-length)
   (%right-length :initarg :right-length
                  :reader right-length)
   (%left-score :initarg :left-score
                :reader left-score)
   (%right-score :initarg :right-score
                 :reader right-score)))


(defclass fundamental-splitter (sl.common:proxy-enabled)
  ())


(defclass data-point-oriented-splitter (fundamental-splitter)
  ())


(defclass random-splitter (sl.common:lifting-proxy)
  ((%trials-count :initarg :trials-count
                  :reader trials-count)))


(defclass random-attribute-splitter (data-point-oriented-splitter)
  ())


(defclass hyperplane-splitter (data-point-oriented-splitter)
  ())


(defclass distance-splitter (data-point-oriented-splitter)
  ((%distance-function :initarg :distance-function
                       :reader distance-function)
   (%repeats :initarg :repeats
             :reader repeats)
   (%iterations :initarg :iterations
                :reader iterations))
  (:default-initargs :iterations 2))


(defclass set-splitter (data-point-oriented-splitter)
  ()
  (:documentation "Splitter which can be used for unordered sets of tuples."))


(defclass fundamental-node ()
  ())


(defclass fundamental-tree-node (fundamental-node)
  ((%left-node :initarg :left-node
               :accessor left-node)
   (%right-node :initarg :right-node
                :accessor right-node)
   (%point :initarg :point
           :accessor point)))


(defclass fundamental-leaf-node (fundamental-node)
  ())


(defclass standard-leaf-node (fundamental-leaf-node fundamental-node)
  ((%predictions :initarg :predictions
                 :accessor predictions))
  (:default-initargs :predictions nil))


(defclass fundamental-tree-training-parameters
    (statistical-learning.mp:fundamental-model-parameters)
  ())


(defclass basic-tree-training-parameters
    (fundamental-tree-training-parameters)
  ((%tree-node-class :initarg :tree-node-class
                     :reader tree-node-class)
   (%leaf-node-class :initarg :leaf-node-class
                     :reader leaf-node-class))
  (:default-initargs
   :tree-node-class 'fundamental-tree-node
   :leaf-node-class 'fundamental-leaf-node))


(defclass standard-tree-training-parameters (basic-tree-training-parameters)
  ((%maximal-depth :initarg :maximal-depth
                   :reader maximal-depth)
   (%minimal-difference :initarg :minimal-difference
                        :reader minimal-difference)
   (%minimal-size :initarg :minimal-size
                  :reader minimal-size)
   (%parallel :initarg :parallel
              :reader parallel)
   (%splitter :initarg :splitter
              :reader splitter))
  (:default-initargs
   :splitter (sl.common:lift (make 'random-attribute-splitter)
                             'random-splitter
                             :trials-count 20)))


(defclass tree-training-state (sl.mp:fundamental-training-state)
  ((%attribute-indexes :initarg :attributes
                       :accessor attribute-indexes)
   (%data-points :initarg :data-points
                 :accessor sl.mp:data-points)
   (%depth :initarg :depth
           :reader depth)
   (%loss :initarg :loss
          :reader loss)
   (%target-data :initarg :target-data
                 :reader sl.mp:target-data)
   (%weights :initarg :weights
             :reader sl.mp:weights)
   (%split-point :initarg :split-point
                 :accessor split-point)
   (%train-data :initarg :train-data
                :reader sl.mp:train-data)
   (%spritter-state :initarg :splitter-state
                    :accessor splitter-state)
   (%parent-state :initarg :parent-state
                  :reader parent-state))
  (:default-initargs :depth 0
                     :attributes nil
                     :split-point nil
                     :splitter-state nil
                     :weights nil
                     :parent-state nil
                     :data-points nil))


(defclass tree-model (statistical-learning.mp:supervised-model)
  ((%root :initarg :root
          :writer write-root
          :reader root)
   (%attribute-indexes :initarg :attribute-indexes
                       :reader attribute-indexes)
   (%forced :initarg :forced
            :accessor forced)
   (%weight :initarg :weight
            :accessor weight))
  (:default-initargs
   :forced nil
   :weight 1.0))


(defclass contributed-predictions ()
  ((%training-parameters :initarg :training-parameters
                         :reader sl.mp:training-parameters)
   (%predictions-lock :initarg :predictions-lock
                      :reader predictions-lock)
   (%contributions-count :initarg :contributions-count
                         :accessor contributions-count)
   (%indexes :initarg :indexes
             :reader indexes)
   (%sums :initarg :sums
          :accessor sums))
  (:default-initargs
   :contributions-count 0.0
   :predictions-lock (bt:make-lock)))
