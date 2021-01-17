(cl:in-package #:sl.som)


(defgeneric alpha (decay initial iteration iterations))
(defgeneric sigma (decay initial iteration iterations))
(defgeneric unit-at (model location))
(defgeneric find-best-matching-unit (selector data sample units))
