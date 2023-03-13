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
                        :reader sl.opt:optimized-function)
   (%relabel-iterations :initarg :relabel-iterations
                        :reader relabel-iterations)
   (%relabel-repeats :initarg :relabel-repeats
                     :reader relabel-repeats))
  (:default-initargs
   :relabel-iterations 4
   :relabel-repeats 4
   :splitter (make 'sl.tp:random-attribute-splitter)
   :optimized-function (sl.opt:squared-error)))
