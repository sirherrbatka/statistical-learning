(cl:in-package #:cl-grf.algorithms)


(defclass scored-classification (cl-grf.tp:fundamental-training-parameters)
  ())


(defclass information-gain-classification (scored-classification)
  ((%minimal-difference :initarg :minimal-difference
                        :accessor minimal-difference)))


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
