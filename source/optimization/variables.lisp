(cl:in-package #:statistical-learning.optimization)


(def <squared-error> (make 'squared-error-function))
(defconstant right t)
(defconstant left nil)
(defconstant middle 'middle)
