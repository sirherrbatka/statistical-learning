(cl:in-package #:statistical-learning.tree-protocol)


(defclass fundamental-node ()
  ())


(defclass fundamental-tree-node (fundamental-node)
  ((%left-node :initarg :left-node
               :accessor left-node)
   (%right-node :initarg :right-node
                :accessor right-node)
   (%attribute :initarg :attribute
               :accessor attribute)
   (%attribute-value :initarg :attribute-value
                     :accessor attribute-value)))


(defclass fundamental-leaf-node (fundamental-node)
  ())


(defclass fundamental-tree-training-parameters
    (statistical-learning.mp:fundamental-model-parameters)
  ((%maximal-depth :initarg :maximal-depth
                   :reader maximal-depth)
   (%minimal-size :initarg :minimal-size
                  :reader minimal-size)
   (%trials-count :initarg :trials-count
                  :reader trials-count)
   (%parallel :initarg :parallel
              :reader parallel)))


(defclass fundamental-split-candidate ()
  ((%needs-split-p :initarg :needs-split-p
                   :reader needs-split-p)
   (%attribute :initarg :attribute
               :accessor attribute)
   (%attribute-value :initarg :attribute-value
                     :accessor attribute-value)
   (%left-node :initarg :left-node
               :accessor left-node)
   (%right-node :initarg :right-node
                :accessor right-node)))


(defclass fundamental-training-state ()
  ((%training-parameters :initarg :training-parameters
                         :accessor training-parameters)
   (%attribute-indexes :initarg :attribute-indexes
                       :accessor attribute-indexes)
   (%target-data :initarg :target-data
                 :accessor target-data)
   (%depth :initarg :depth
           :accessor depth)
   (%loss :initarg :loss
          :accessor loss)
   (%weights :initarg :weights
             :accessor weights)
   (%training-data :initarg :training-data
                   :accessor training-data))
  (:default-initargs :depth 0
                     :weights nil))


(defclass tree-model (statistical-learning.mp:fundamental-model)
  ((%root :initarg :root
          :writer write-root
          :reader root)))


(defclass contributed-predictions ()
  ((%training-parameters :initarg :training-parameters
                         :reader training-parameters)))
