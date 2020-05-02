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


(defun select-random-indexes (selected-count
                              total-count)
  (check-type selected-count positive-integer)
  (check-type total-count positive-integer)
  (~>> total-count
       iota
       shuffle
       (take selected-count)
       (coerce _ '(vector fixnum))))


(defun selecting-random-indexes (selected-count
                                 total-count)
  (check-type selected-count positive-integer)
  (check-type total-count positive-integer)
  (curry #'select-random-indexes
         selected-count
         total-count))
