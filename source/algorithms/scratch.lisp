(cl:in-package #:cl-grf.algorithms)

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
  (setf (aref *data* i 0) (if (oddp i)
                                (random-uniform 0.7d0 1.0d0)
                                (random-uniform 0.0d0 0.8d0))))

(defparameter *training-parameters*
  (make 'information-gain-classification
        :maximal-depth 3
        :minimal-difference 0.001d0
        :minimal-size 1
        :trials-count 500
        :parallel nil))


(defparameter *training-state*
  (make 'cl-grf.tp:fundamental-training-state
        :training-parameters *training-parameters*
        :attribute-indexes (make-array 1 :element-type 'fixnum
                                         :initial-element 0)
        :target-data *target*
        :training-data *data*))


(defparameter *leaf* (cl-grf.tp:make-leaf *training-state*))

(defparameter *tree* (cl-grf.tp:split *training-state* *leaf*))
