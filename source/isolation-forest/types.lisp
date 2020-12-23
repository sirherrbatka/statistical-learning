(cl:in-package #:statistical-learning.isolation-forest)


(defclass isolation-splitter (sl.tp:fundamental-splitter)
  ())


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
   :splitter (make 'isolation-splitter)
   :parallel nil))


(defclass isolation-model (sl.tp:tree-model)
  ((%normals :initarg :normals
             :reader normals)
   (%attributes :initarg :attributes
                :reader attributes)
   (%c :initarg :c
       :reader c)
   (%mins :initarg :mins
          :reader mins)
   (%maxs :initarg :maxs
          :reader maxs)
   (%global-min :initarg :global-min
                :reader global-min)
   (%global-max :initarg :global-max
                :reader global-max)))


(defclass isolation-leaf (sl.tp:fundamental-leaf-node)
  ())


(defclass isolation-training-state (sl.mp:fundamental-training-state)
  ((%parameters :initarg :parameters
                :reader sl.mp:training-parameters)
   (%global-min :initarg :global-min
                :reader global-min)
   (%global-max :initarg :global-max
                :reader global-max)
   (%split-point :initarg :split-point
                 :accessor sl.tp:split-point)
   (%depth :initarg :depth
           :reader sl.tp:depth)
   (%mins :initarg :mins
          :reader mins)
   (%normals :initarg :normals
             :reader normals)
   (%c :initarg :c
       :reader c)
   (%maxs :initarg :maxs
          :reader maxs)
   (%train-data :initarg :train-data
          :reader sl.mp:train-data)
   (%attributes :initarg :attributes
                :reader sl.tp:attribute-indexes)
   (%data-points :initarg :data-points
                 :reader sl.mp:data-points)
   (%gaussian-state :initarg :gaussian-state
                    :reader gaussian-state))
  (:default-initargs
   :gaussian-state (sl.common:make-gauss-random-state)))


(defclass isolation-prediction ()
  ((%trees-count :initarg :trees-count
                 :accessor trees-count)
   (%trees-sum :initarg :trees-sum
               :accessor trees-sum)
   (%indexes :initarg :indexes
             :reader sl.tp:indexes)
   (%c :initarg :c
       :reader c))
  (:default-initargs
   :trees-count 0))
