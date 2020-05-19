(cl:in-package #:cl-grf.algorithms)


(defclass scored-classification (cl-grf.tp:fundamental-tree-training-parameters)
  ())


(defclass impurity-classification (scored-classification)
  ((%minimal-difference :initarg :minimal-difference
                        :reader minimal-difference)))


(defclass single-impurity-classification
    (impurity-classification)
  ((%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


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
