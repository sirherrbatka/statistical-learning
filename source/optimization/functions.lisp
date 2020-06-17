(cl:in-package #:statistical-learning.optimization)


(defun make-split-array (length)
  (make-array length :element-type t
                     :initial-element nil))


(defun k-logistic (number-of-classes)
  (make 'k-logistic-function
        :number-of-classes number-of-classes))


(defun gini-impurity (number-of-classes)
  (make 'gini-impurity-function
        :number-of-classes number-of-classes))


(defun squared-error ()
  <squared-error>)
