(cl:in-package #:cl-grf.forest)


(defmethod leafs-for ((forest fundamental-random-forest)
                      data)
  (check-type data cl-grf.data:data-matrix)
  (map 'vector
       (lambda (tree features)
         (~>> (select-features data features)
              (cl-grf.tp:leafs-for tree)))
       (trees forest)
       (features forest)))
