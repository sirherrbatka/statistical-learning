(cl:in-package #:cl-grf.forest)


(defgeneric leafs-for (random-forest data))
(defgeneric prediction-from-leafs (random-forest leafs))
