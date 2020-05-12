(cl:in-package #:cl-grf.tree-protocol)


(defclass fundamental-node (cl-grf.mp:fundamental-model)
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
    (cl-grf.mp:fundamental-model-parameters)
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
   (%training-data :initarg :training-data
                   :accessor training-data))
  (:default-initargs :depth 0))


(defclass fundamental-tree ()
  ())


(defclass decision-tree (fundamental-tree)
  ())


(defclass regression-tree (fundamental-tree)
  ())
