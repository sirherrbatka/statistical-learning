(cl:in-package #:cl-user)

(ql:quickload :statistical-learning)
(ql:quickload :vellum)

(defpackage #:promotions-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:promotions-example)


(defvar *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                                  (merge-pathnames "examples/promotions.data"))
                        :header t)
      (vellum:to-table :columns '((:alias id)
                                  (:alias promotion) ; file has yes or no as values
                                  (:alias purchase :type integer) ; either 0 or 1
                                  (:alias v1 :type float)
                                  (:alias v2 :type float)
                                  (:alias v3 :type float)
                                  (:alias v4 :type float)
                                  (:alias v5 :type float)
                                  (:alias v6 :type float)
                                  (:alias v7 :type float))
                       :body (vellum:body (promotion)
                               (setf promotion (econd ((string= promotion "Yes") 1)
                                                      ((string= promotion "No") 0)))))
      (vellum:select :columns '(:take-from promotion :take-to v7)))) ; who cares about id?


(defvar *train-data*
  (~> (vellum:select *data* :columns '(:take-from v1 :take-to v7))
      (vellum:to-matrix :element-type 'double-float)))


(defvar *target-data*
  (~> (vellum:select *data* :columns '(:v purchase))
      (vellum:to-matrix :element-type 'double-float)))


(defvar *treatment-data*
  (~> (vellum:select *data* :columns '(:v promotion))
      (vellum:to-matrix :element-type 'double-float)))


(defparameter *training-parameters*
  (~> (make 'statistical-learning.dt:classification
            :optimized-function (sl.opt:gini-impurity 2)
            :maximal-depth 3
            :minimal-difference 0.0d0
            :minimal-size 50
            :trials-count 15
            :parallel nil)
      (sl.pt:causal 10 2) ; 10 data points for promotion + 10 data points for no promotions required, 0 designates no promotion, 1 designates promotion
      sl.pt:honest))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 500
        :parallel nil
        :tree-batch-size 100
        :tree-attributes-count 3
        :tree-sample-rate 0.5
        :tree-parameters *training-parameters*))

(defparameter *model*
  (sl.mp:make-supervised-model *forest-parameters*
                               *train-data*
                               *target-data*
                               :treatment *treatment-data*))

(defparameter *predictions* (sl.mp:predict *model* *train-data* t))

(defparameter *gains*
  (iterate
    (with result = (sl.data:make-data-matrix-like (aref *predictions* 0)))
    (for i from 0 below (~> (aref *predictions* 0) array-total-size))
    (setf (row-major-aref result i)
          (- (row-major-aref (aref *predictions* 1) i)
             (row-major-aref (aref *predictions* 0) i)))
    (finally (return result))))

(defparameter *purchase-profit* 10.0d0)
(defparameter *promotion-cost* 0.10d0)

;; and here are the results. If profit from the purchase is $10 and the cost of the promotion is $0.10 we need to have at least 1% increase probability of purchase (1% of $10 is $0.10) to break even.
(defparameter *expected-promotion-gain*
  (iterate
    (with data-points-count = (sl.data:data-points-count *gains*))
    (with result = (make-array data-points-count
                               :element-type 'double-float))
    (for i from 0 below data-points-count)
    (setf (aref result i) (- (* *purchase-profit*
                                (sl.data:mref *gains* i 1))
                             *promotion-cost*))
    (finally (return result))))
