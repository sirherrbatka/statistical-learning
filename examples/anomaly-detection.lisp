(cl:in-package #:cl-user)

(ql:quickload :statistical-learning)
(ql:quickload :vellum)

(defpackage #:anomaly-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:anomaly-example)

(defvar *data*
  (lret ((data (vellum:make-table
                :columns '((:alias x :type double-float)
                           (:alias y :type double-float)))))
    ;; build normal population
    (let ((state (sl.common:make-gauss-random-state)))
      (vellum:transform data
                        (vellum:body (x y)
                          (let ((r (sl.common:gauss-random state)))
                            (setf x (- r 2.0d0)
                                  y (+ r 2.0d0))))
                        :end 100
                        :in-place t))
    ;; add abnormal observations
    (vellum:transform data
                      (vellum:body (x y)
                        (setf x #1=(random-in-range -5.0d0 5.0d0)
                              y #1#))
                      :start 100
                      :end 110
                      :in-place t)))

(defparameter *tree-parameters*
  (make 'sl.if:isolation
        :maximal-depth 2
        :minimal-size 3))

(defparameter *isolation-forest*
  (make 'sl.ensemble:isolation-forest
        :trees-count 20
        :parallel nil
        :tree-batch-size 5
        :tree-parameters *tree-parameters*
        :tree-attributes-count 2
        :tree-sample-rate 0.2))

(defparameter *model*
  (sl.mp:make-unsupervised-model *isolation-forest*
                                 (vellum:to-matrix *data*
                                                   :element-type 'double-float)))

;; (clouseau:inspect *model*)
