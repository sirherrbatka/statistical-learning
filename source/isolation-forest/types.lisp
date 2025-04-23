(cl:in-package #:statistical-learning.isolation-forest)


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
   :repeats 1
   :splitter (make 'sl.tp:hyperplane-splitter)
   :parallel nil))


(defclass isolation-model (sl.tp:tree-model)
  ((%c :initarg :c :reader c)))


(defclass isolation-leaf (sl.tp:fundamental-leaf-node)
  ())


(defclass isolation-training-state (sl.mp:fundamental-training-state)
  ((%split-point :initarg :split-point
                 :accessor sl.tp:split-point)
   (%loss :initarg :loss) ;; this is purely for the compatibility with the tree protocol
   (%depth :initarg :depth
           :reader sl.tp:depth)
   (%c :initarg :c
       :reader c)
   (%train-data :initarg :train-data
          :reader sl.mp:train-data)
   (%attributes :initarg :attributes
                :reader sl.tp:attribute-indexes)))


(defclass isolation-prediction ()
  ((%parameters :initarg :parameters
                :reader sl.mp:training-parameters)
   (%trees-count :initarg :trees-count
                 :accessor trees-count)
   (%trees-sum :initarg :trees-sum
               :accessor trees-sum)
   (%c :initarg :c
       :reader c))
  (:default-initargs
   :trees-count 0))


(defstruct split-point
  (normals (sl.data:make-data-matrix 0 0)
   :type sl.data:single-float-data-matrix)
  (dot-product 0.0 :type single-float))
