(cl:in-package #:sl.som)


(defgeneric alpha (decay initial iteration iterations))
(defgeneric sigma (decay initial iteration iterations))
(defgeneric unit-at (model location))
(defgeneric find-best-matching-unit-with-selector
    (selector parameters data sample units))
(defgeneric find-best-matching-unit (parameters units-container))
(defgeneric forest (random-forest-self-organizing-map))
(defgeneric units-leafs (units-container))
(defgeneric fit (parameters state))
(defgeneric make-units-container (model data index))
