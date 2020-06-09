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


(defclass regression (scored-training)
  ())


(defclass basic-regression (regression)
  ())


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
