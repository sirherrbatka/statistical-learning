(cl:in-package #:cl-user)

(ql:quickload '(:vellum :vellum-csv :statistical-learning))

(defpackage #:anomaly-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:anomaly-example)

(defparameter *data*
  (lret ((data (vellum:make-table
                :columns '((:name x :type double-float)
                           (:name y :type double-float)))))
    ;; build normal population
    (vellum:transform data
                      (vellum:bind-row (x y)
                        (let ((re (+ 5 (sl.random:random-gauss))))
                          (setf x (+ re (random-in-range -0.1 0.1))
                                y (+ re (random-in-range -0.1 0.1)))))
                      :end 500
                      :in-place t)
    (vellum:transform data
                      (vellum:bind-row (x y)
                        (setf x (random-in-range -1.5d0 1.5d0)
                              y (random-in-range -1.5d0 1.5d0)))
                      :start 500
                      :end 520
                      :in-place t)))

(defparameter *tree-parameters*
  (make 'sl.if:isolation
        :maximal-depth 30
        :minimal-size 3
        :splitter (make-instance 'sl.tp:hyperplane-splitter)))

(defparameter *isolation-forest*
  (make 'sl.ensemble:isolation-forest
        :trees-count 500
        :parallel nil
        :tree-batch-size 5
        :tree-parameters *tree-parameters*
        :tree-attributes-count 2
        :tree-sample-rate 0.2))

(defparameter *model*
  (sl.mp:make-unsupervised-model *isolation-forest*
                                 (vellum:to-matrix *data*
                                                   :element-type 'double-float)))

(defparameter *anomaly-scores*
  (sl.mp:predict *model*
                 (vellum:to-matrix *data*
                                   :element-type 'double-float)))
