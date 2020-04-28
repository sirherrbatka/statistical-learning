(cl:in-package #:cl-grf.forest)


(defun predict (random-forest data)
  (~>> data
       (leafs-for random-forest)
       (prediction-from-leafs random-forest)))
