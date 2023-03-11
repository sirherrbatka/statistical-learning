(cl:in-package #:statistical-learning.struct-forest)


(defclass struct-state (sl.tp:tree-training-state)
  ((%relabaled :initform nil
               :accessor relabaled)))


(defclass struct-training-implementation (sl.dt:classification)
  ((%original :initarg :original
              :reader original))
  (:default-initargs
   :optimized-function (sl.opt:gini-impurity 2)))


(defclass struct (sl.perf:regression
                  sl.tp:standard-tree-training-parameters)
  ((%optimized-function :initarg :optimized-function
                        :reader sl.opt:optimized-function))
  (:default-initargs
   :splitter (make 'sl.tp:random-attribute-splitter)
   :optimized-function (sl.opt:squared-error)))
