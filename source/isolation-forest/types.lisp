(cl:in-package #:statistical-learning.isolation-forest)


(defclass isolation-splitter (sl.tp:fundamental-splitter)
  ())


(defclass isolation (sl.tp:basic-tree-training-parameters)
  ((%maximal-depth :initarg :maximal-depth
                   :reader sl.tp:maximal-depth
                   :accessor access-maximal-depth)
   (%repeats :initarg :repeats
             :accessor access-repeats
             :reader repeats)
   (%parallel :initarg :parallel
              :accessor access-parallel
              :reader sl.tp:parallel)
   (%splitter :initarg :splitter
              :accessor access-splitter
              :reader sl.tp:splitter)
   (%minimal-size :initarg :minimal-size
                  :accessor access-minimal-size
                  :reader sl.tp:minimal-size))
  (:default-initargs
   :repeats 5
   :splitter (make 'isolation-splitter)
   :parallel nil))


(defclass isolation-model (sl.tp:tree-model)
  ((%attributes :initarg :attributes
                :reader attributes)
   (%c :initarg :c :reader c)))


(defclass isolation-tree (sl.tp:fundamental-tree-node)
  ((%size :initarg :size
          :accessor size))
  (:default-initargs :size 0))


(defclass isolation-leaf (sl.tp:fundamental-leaf-node)
  ((%size :initarg :size
          :accessor size))
  (:default-initargs :size 0))


(defclass isolation-training-state (sl.mp:fundamental-training-state)
  ((%parameters :initarg :parameters
                :reader sl.mp:training-parameters)
   (%split-point :initarg :split-point
                 :accessor sl.tp:split-point)
   (%depth :initarg :depth
           :reader sl.tp:depth)
   (%c :initarg :c
       :reader c)
   (%mins :accessor mins
          :documentation "Cache for point selection."
          :initarg :mins)
   (%maxs :accessor maxs
          :documentation "Cache for point selection."
          :initarg :maxs)
   (%train-data :initarg :train-data
          :reader sl.mp:train-data)
   (%attributes :initarg :attributes
                :reader sl.tp:attribute-indexes)
   (%data-points :initarg :data-points
                 :reader sl.mp:data-points))
  (:default-initargs :mins nil :maxs nil))


(defclass isolation-prediction ()
  ((%parameters :initarg :parameters
                :reader sl.mp:training-parameters)
   (%trees-count :initarg :trees-count
                 :accessor trees-count)
   (%trees-sum :initarg :trees-sum
               :accessor trees-sum)
   (%indexes :initarg :indexes
             :reader sl.tp:indexes)
   (%c :initarg :c
       :reader c))
  (:default-initargs
   :trees-count 0))


(defstruct split-point
  (normals (sl.data:make-data-matrix 0 0)
   :type sl.data:double-float-data-matrix)
  (dot-product 0.0d0 :type double-float))
