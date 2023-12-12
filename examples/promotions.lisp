(cl:in-package #:cl-user)

(ql:quickload '(:vellum :vellum-csv :statistical-learning))

(cl:defpackage #:promotions-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:promotions-example)

(defvar *data*
  (~> (vellum:copy-from :csv (~>> (asdf:system-source-directory :statistical-learning)
                                  (merge-pathnames "examples/promotions.data"))
                        :columns '((:name id)
                                   (:name promotion) ; file has yes or no as values
                                   (:name purchase :type integer) ; either 0 or 1
                                   (:name v1 :type float)
                                   (:name v2 :type float)
                                   (:name v3 :type float)
                                   (:name v4 :type float)
                                   (:name v5 :type float)
                                   (:name v6 :type float)
                                   (:name v7 :type float))
                        :body (vellum:bind-row (promotion)
                                (setf promotion (econd ((string= promotion "Yes") 1)
                                                       ((string= promotion "No") 0)))))
      (vellum:select :columns (vellum:s (vellum:between :from 'promotion))))) ; who cares about id?


(defvar *train-data*
  (~> (vellum:select *data* :columns (vellum:s (vellum:between :from 'v1)))
      (vellum:to-matrix :element-type 'double-float)
      sl.data:wrap))


(defvar *target-data*
  (~> (vellum:select *data* :columns '(purchase))
      (vellum:to-matrix :element-type 'double-float)))


(defvar *treatment-data*
  (~> (vellum:select *data* :columns '(promotion))
      (vellum:to-matrix :element-type 'double-float)
      sl.data:wrap))


(defparameter *training-parameters*
  (~> (make 'statistical-learning.dt:classification
            :optimized-function (sl.opt:gini-impurity 2)
            :maximal-depth 3
            :minimal-difference 0.0d0
            :minimal-size 50
            :parallel nil)
      (sl.pt:causal 10 2) ; 10 data points for promotion + 10 data points for no promotions required, 0 designates no promotion, 1 designates promotion
      ))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 500
        :parallel nil
        :tree-batch-size 100
        :tree-attributes-count 3
        :data-points-sampler (make-instance 'sl.ensemble:weights-based-data-points-sampler
                                            :sampling-rate 0.5)
        :tree-parameters *training-parameters*))

(defparameter *model*
  (sl.mp:make-supervised-model *forest-parameters*
                               *train-data*
                               *target-data*
                               :treatment *treatment-data*))

(defparameter *predictions* (sl.mp:predict *model* (sl.data:wrap *train-data*) nil))

(defparameter *gains*
  (iterate
    (with result = (copy-array (sl.data:mref *predictions* 0 0)))
    (for i from 0 below (~> (sl.data:mref *predictions* 0 0) array-total-size))
    (setf (row-major-aref result i)
          (- (row-major-aref (sl.data:mref *predictions* 1 0) i)
             (row-major-aref (sl.data:mref *predictions* 0 0) i)))
    (finally (return result))))

(defparameter *purchase-profit* 10.0d0)
(defparameter *promotion-cost* 0.10d0)

;; and here are the results. If profit from the purchase is $10 and the cost of the promotion is $0.10 we need to have at least 1% increase probability of purchase (1% of $10 is $0.10) to break even.
(defparameter *expected-promotion-gain*
  (iterate
    (with data-points-count = (array-dimension *gains* 0))
    (with result = (make-array data-points-count
                               :element-type 'double-float))
    (for i from 0 below data-points-count)
    (setf (aref result i) (- (* *purchase-profit* (aref *gains* i 1)) *promotion-cost*))
    (finally (return result))))
