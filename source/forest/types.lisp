(cl:in-package #:cl-grf.forest)


(defclass random-forest-parameters ()
  ((%trees-count :initarg :trees-count
                 :reader trees-count
                 :type positive-integer)
   (%forest-class :initarg :forest-class
                  :reader forest-class)
   (%parallel :initarg :parallel
              :reader parallel)
   (%tree-attributes-count :initarg :tree-attributes-count
                           :reader tree-attributes-count)
   (%tree-sample-size :initarg :tree-sample-size
                      :reader tree-sample-size)
   (%tree-parameters :initarg :tree-parameters
                     :reader tree-parameters)))


(defclass fundamental-random-forest ()
  ((%trees :initarg :trees
           :reader trees
           :type simple-vector)
   (%tree-attributes-count :initarg :attributes-count
                           :reader tree-attributes-count)
   (%attributes :initarg :attributes
                :reader attributes
                :type simple-vector)))


(defclass classification-random-forest (fundamental-random-forest)
  ())
