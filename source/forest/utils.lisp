(cl:in-package #:cl-grf.forest)


(defun select-features (data features)
  (lret ((result (make-array (list
                              (cl-grf.data:data-points-count data)
                              (length features))
                             :element-type 'double-float)))
    (iterate
      (for i from 0 below (cl-grf.data:data-points-count data))
      (iterate
        (for k from 0 below (length features))
        (for j = (aref features k))
        (setf (aref result i k) (aref data i j))))))


(defun select-random-attributes (tree-attributes-count
                                 total-attributes-count)
  (check-type tree-attributes-count positive-integer)
  (check-type total-attributes-count positive-integer)
  (~>> total-attributes-count
       iota
       shuffle
       (take tree-attributes-count)
       (coerce _ '(vector fixnum))))


(defun selecting-random-attributes (tree-attributes-count
                                    total-attributes-count)
  (check-type tree-attributes-count positive-integer)
  (check-type total-attributes-count positive-integer)
  (curry #'select-random-attributes
         tree-attributes-count
         total-attributes-count))
