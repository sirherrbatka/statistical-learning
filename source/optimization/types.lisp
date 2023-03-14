(cl:in-package #:statistical-learning.optimization)


(defclass fundamental-loss-function ()
  ())


(defclass squared-error-function (fundamental-loss-function)
  ())


(defclass gini-impurity-function (fundamental-loss-function)
  ((%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(defclass k-logistic-function (fundamental-loss-function)
  ((%number-of-classes :initarg :number-of-classes
                       :reader number-of-classes)))


(deftype weights-data-matrix ()
  `sl.data:double-float-data-matrix)
