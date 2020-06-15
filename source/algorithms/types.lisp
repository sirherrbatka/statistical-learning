(cl:in-package #:statistical-learning.algorithms)


(defclass scored-training (statistical-learning.tp:fundamental-tree-training-parameters)
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


(defclass gradient-boost-training-state (statistical-learning.tp:fundamental-training-state)
  ((%shrinkage :initarg :shrinkage
               :reader shrinkage)
   (%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(defmethod cl-ds.utils:cloning-information
    append ((state gradient-boost-training-state))
  '((:shrinkage shrinkage)
    (:number-of-classes number-of-classes)))


(defclass gradient-boost (scored-training)
  ())


(defclass gradient-boost-regression (regression gradient-boost)
  ())


(defclass gradient-boost-classification (classification gradient-boost)
  ((%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(defclass gradient-boost-model (statistical-learning.tp:tree-model)
  ((%expected-value :initarg :expected-value
                    :reader expected-value)
   (%shrinkage :initarg :shrinkage
               :reader shrinkage)))


(defclass score ()
  ((%score :initarg :score
           :accessor score)))


(defclass scored-node (score)
  ((%support :initarg :support
             :accessor support)))


(defclass scored-leaf-node (scored-node
                            statistical-learning.tp:fundamental-leaf-node)
  ((%predictions :initarg :predictions
                 :type statistical-learning.data:data-matrix
                 :accessor predictions)))


(defclass scored-tree-node (scored-node
                            statistical-learning.tp:fundamental-tree-node)
  ())


(defclass gathered-predictions (statistical-learning.tp:contributed-predictions)
  ((%contributions-count :initarg :contributions-count
                         :accessor contributions-count)
   (%indexes :initarg :indexes
             :reader indexes)
   (%sums :initarg :sums
          :reader sums))
  (:default-initargs :contributions-count 0))
