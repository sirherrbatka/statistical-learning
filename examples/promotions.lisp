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
  (make 'statistical-learning.dt:classification
        :optimized-function (sl.opt:gini-impurity 2)
        :maximal-depth 5
        :minimal-difference 0.00000001d0
        :minimal-size 25
        :trials-count 25
        :parallel nil))

(defparameter *forest-parameters*
  (make 'statistical-learning.ensemble:random-forest
        :trees-count 500
        :parallel t
        :tree-batch-size 20
        :tree-attributes-count 5
        :tree-sample-rate 0.5
        :tree-parameters (~> *training-parameters*
                             (sl.pt:causal 10 10)
                             sl.pt:honest)))

(defparameter *model*
  (sl.mp:make-model *forest-parameters*
                    *train-data*
                    *target-data*
                    :treatment *treatment-data*))

;; and here are the results. If cost of the purchase is $10 and the cost of the promotion is $0.10 we need to have at least 1% increase probability of purchase (1% of $10 is $0.10) to get an yield from the promotion.
(defparameter *gains* (sl.mp:predict *model* *train-data* t))
