(cl:in-package #:cl-grf.tree-protocol)


(defclass fundamental-node ()
  ((%feature :initarg :feature
             :accessor feature))
  (:default-initargs :feature nil))


(defclass fundamental-tree-node (fundamental-node)
  ((%left-node :initarg :left-node
               :accessor left-node)
   (%right-node :initarg :right-node
                :accessor right-node))
  (:default-initargs :left-node nil
                     :right-node nil))


(defclass fundamental-leaf-node (fundamental-node)
  ())


(defclass fundamental-training-parameters ()
  ((%maximal-depth :initarg :maximal-depth
                   :accessor maximal-depth)
   (%trees-count :initarg :trees-count
                 :accessor trees-count)
   (%trails-count :initarg :trials-count
                 :accessor trails-count)
   (%leaf-class :initarg :leaf-class
                :accessor leaf-class)))


(defclass fundamental-training-state ()
  ((%training-parameters :initarg :training-parameters
                         :accessor training-parameters)
   (%split-mode :initarg :split-mode
                :accessor split-mode)
   (%needs-split-p-mode :initarg :needs-split-p-mode
                        :accessor needs-split-p-mode)
   (%depth :initarg :depth
           :accessor depth)
   (%training-data :initarg :training-data
                   :accessor training-data)))


(defclass fundamental-tree ()
  ())


(defclass decision-tree (fundamental-tree)
  ())


(defclass regression-tree (fundamental-tree)
  ())
