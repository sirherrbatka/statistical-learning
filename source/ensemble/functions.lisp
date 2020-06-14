(cl:in-package #:statistical-learning.ensemble)


(defun predict (random-forest data &optional parallel)
  (~>> (leafs-for random-forest data parallel)
       (predictions-from-leafs random-forest _ parallel)))
