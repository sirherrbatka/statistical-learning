(cl:in-package #:cl-grf.algorithms)


(defclass scored-training (cl-grf.tp:fundamental-tree-training-parameters)
  ((%minimal-difference :initarg :minimal-difference
                        :reader minimal-difference)))


(defclass classification ()
  ())


(defclass scored-classification (classification scored-training)
  ())


(defclass impurity-classification (scored-classification)
  ())


(defclass single-impurity-classification
    (impurity-classification)
  ((%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(defclass regression ()
  ())


(defclass basic-regression (regression scored-training)
  ())


(defclass gradient-boost-training-state (cl-grf.tp:fundamental-training-state)
  ((%learning-rate :initarg :learning-rate
                   :reader learning-rate)
   (%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(defmethod cl-ds.utils:cloning-information
    append ((state gradient-boost-training-state))
  '((:learning-rate learning-rate)))


(defclass gradient-boost-regression (regression scored-training)
  ())


(defclass gradient-boost-model (cl-grf.tp:tree-model)
  ((%expected-value :initarg :expected-value
                    :reader expected-value)
   (%learning-rate :initarg :learning-rate
                   :reader learning-rate)))


(defclass score ()
  ((%score :initarg :score
           :accessor score)))


(defclass scored-node (score)
  ((%support :initarg :support
             :accessor support)))


(defclass scored-leaf-node (scored-node
                            cl-grf.tp:fundamental-leaf-node)
  ((%predictions :initarg :predictions
                 :type cl-grf.data:data-matrix
                 :accessor predictions)))


(defclass scored-tree-node (scored-node
                            cl-grf.tp:fundamental-tree-node)
  ())
