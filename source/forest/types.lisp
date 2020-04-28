(cl:in-package #:cl-grf.forest)


(defclass random-forest-parameters ()
  ((%trees-count :initarg :trees-count
                 :accessor trees-count
                 :type positive-integer)
   (%tree-features-count :initarg :tree-features-count
                         :accessor tree-features-count)))


(defclass fundamental-random-forest ()
  ((%trees :initarg :trees
           :accessor trees
           :type simple-vector)
   (%features :initarg :features
              :accessor features
              :type simple-vector)))


(defclass classification-random-forest (fundamental-random-forest)
  ())
