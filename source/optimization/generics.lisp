(cl:in-package #:statistical-learning.optimization)


(defgeneric response (function expected function-output))
(defgeneric loss (function target-data weights
                  &optional split-array))
(defgeneric number-of-classes (function))
