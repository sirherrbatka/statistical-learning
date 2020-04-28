(cl:in-package #:cl-grf.forest)


(defmethod leafs-for ((forest fundamental-random-forest)
                      data)
  (map 'vector
       (lambda (tree features)
         (~>> (select-features data features)
              (cl-grf.tp:leaf-for tree)))
       (trees forest)
       (features forest)))
