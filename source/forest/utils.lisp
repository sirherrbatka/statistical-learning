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
