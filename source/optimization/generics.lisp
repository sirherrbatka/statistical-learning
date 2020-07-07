(cl:in-package #:statistical-learning.optimization)


(defgeneric response (function expected function-output))
(defgeneric loss (function target-data weights data-points
                  &optional split-array))
(defgeneric number-of-classes (function))
