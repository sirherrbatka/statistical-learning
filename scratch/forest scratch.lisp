(cl:in-package #:statistical-learning.forest)

(defparameter *data*
  (make-array (list 20 1) :element-type 'double-float
                          :initial-element 0.0d0))

(defparameter *target*
  (make-array (list 20 1) :element-type 'double-float
                          :initial-element 0.0d0))

(iterate
  (for i from 0 below 20)
  (setf (aref *target* i 0) (if (oddp i) 1.0d0 0.0d0)))

(iterate
  (for i from 0 below 20)
  (setf (aref *data* i 0)
        (if (oddp i)
            (statistical-learning.algorithms::random-uniform 0.7d0 1.0d0)
            (statistical-learning.algorithms::random-uniform 0.0d0 0.8d0))))

(defparameter *training-parameters*
  (make 'statistical-learning.algorithms:information-gain-classification
        :maximal-depth 3
        :minimal-difference 0.001d0
        :minimal-size 1
        :trials-count 500
        :parallel nil))


(defparameter *forest-parameters*
  (make 'random-forest-parameters
        :trees-count 3
        :forest-class 'classification-random-forest
        :parallel nil
        :tree-attributes-count 1
        :tree-sample-size 5
        :tree-parameters *training-parameters*))


(defparameter *forest* (statistical-learning.mp:make-model *forest-parameters*
                                             *data*
                                             *target*))

(iterate
  (with attribute-value = (statistical-learning.tp:attribute-value *tree*))
  (with result = (make-array 20))
  (for i from 0 below 20)
  (setf (aref result i) (list (> (aref *data* i 0)
                                 attribute-value)
                              (aref *target* i 0)))
  (finally
   (print result)))
