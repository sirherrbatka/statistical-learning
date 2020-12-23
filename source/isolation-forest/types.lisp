(cl:in-package #:statistical-learning.isolation-forest)


(defclass isolation-splitter (sl.tp:fundamental-splitter)
  ((%normals :initarg :normals
             :reader normals)))


(defclass isolation (sl.tp:basic-tree-training-parameters)
  ((%maximal-depth :initarg :maximal-depth
                   :reader sl.tp:maximal-depth)
   (%parallel :initarg :parallel
              :reader sl.tp:parallel)
   (%splitter :initarg :splitter
              :reader sl.tp:splitter)
   (%minimal-size :initarg :minimal-size
                  :reader sl.tp:minimal-size))
  (:default-initargs
   :splitter (make-instance 'isolation-splitter)))


(defclass isolation-leaf (sl.tp:fundamental-leaf-node)
  ())


(defclass isolation-training-state (sl.mp:fundamental-training-state)
  ((%parameters :initarg :parameters
                :reader sl.mp:training-parameters)
   (%global-min :initarg :global-min
                :reader global-min)
   (%global-max :initarg :global-max
                :reader global-max)
   (%depth :initarg :depth
           :reader sl.tp:depth)
   (%averages :initarg :averages
              :reader averages)
   (%mins :initarg :mins
          :reader mins)
   (%maxs :initarg :maxs
          :reader maxs)
   (%train-data :initarg :train-data
          :reader sl.mp:train-data)
   (%attributes :initarg :attributes
                :reader sl.tp:attribute-indexes)
   (%data-points :initarg :data-points
                 :reader sl.mp:data-points)
   (%gaussian-state :initarg :gaussian-state
                    :reader gaussian-state)))


(defstruct isolation-forest-split-point
  (dot-product 0.0d0 :type double-float)
  attributes)


(defclass isolation-prediction ()
  ((%trees-count :initarg :trees-count
                 :accessor trees-count)
   (%trees-sum :initarg :trees-sum
               :accessor trees-sum)
   (%indexes :initarg :indexes
             :reader sl.tp:indexes)
   (%predictions-lock :initarg :predictions-lock
                      :reader sl.tp:predictions-lock)
   (%c :initarg :c
       :reader c))
  (:default-initargs
   :trees-count 0
   :predictions-lock (bt:make-lock)))
