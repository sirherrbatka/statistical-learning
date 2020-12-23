(cl:in-package #:cl-user)

(ql:quickload :statistical-learning)
(ql:quickload :vellum)

(defpackage #:anomaly-example
  (:use #:cl #:statistical-learning.aux-package))

(cl:in-package #:anomaly-example)

(defparameter *data*
  (lret ((data (vellum:make-table
                :columns '((:alias x :type double-float)
                           (:alias y :type double-float)))))
    ;; build normal population
    (vellum:transform data
                      (vellum:body (x y)
                        (let ((re (sl.common:gauss-random)))
                          (setf x (+ re (random-in-range -0.2 0.2))
                                y (+ re (random-in-range -0.2 0.2)))))
                      :end 200
                      :in-place t)
    (vellum:transform data
                      (vellum:body (x y)
                        (setf x (random-in-range -2.0d0 2.0d0)
                              y (random-in-range -2.0d0 2.0d0)))
                      :start 200
                      :end 220
                      :in-place t)))

(vellum.plot:visualize
 :plotly
 (vellum.plot:stack *data*
                    (vellum.plot:aesthetics
                     :label "Heatmap"
                     :x (vellum.plot:axis :label "X" :scale-ratio 1 :scale-anchor :y)
                     :y (vellum.plot:axis :label "Y" :scale-ratio 1))
                    (vellum.plot:points
                     :aesthetics (vellum.plot:aesthetics)
                     :mapping (vellum.plot:mapping :x 'x :y 'y)))
 "~/dist.html")

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

(clouseau:inspect *model*)
