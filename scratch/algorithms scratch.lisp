(cl:in-package #:statistical-learning.algorithms)

(defparameter *data*
  (make-array (list 20 1) :element-type 'double-float
                          :initial-element 0.0))

(defparameter *target*
  (make-array (list 20 1) :element-type 'double-float
                          :initial-element 0.0))

(iterate
  (for i from 0 below 20)
  (setf (aref *target* i 0) (if (oddp i) 1.0 0.0)))

(iterate
  (for i from 0 below 20)
  (setf (aref *data* i 0) (if (oddp i)
                                (random-uniform 0.7 1.0)
                                (random-uniform 0.0 0.8))))

(defparameter *training-parameters*
  (make 'information-gain-classification
        :maximal-depth 3
        :minimal-difference 0.001
        :minimal-size 1
        :trials-count 500
        :parallel nil))


(defparameter *tree* (statistical-learning.mp:make-model *training-parameters* *data* *target*))

(iterate
  (with attribute-value = (statistical-learning.tp:attribute-value *tree*))
  (with result = (make-array 20))
  (for i from 0 below 20)
  (setf (aref result i) (list (> (aref *data* i 0)
                                 attribute-value)
                              (aref *target* i 0)))
  (finally
   (print result)))
