(cl:in-package #:statistical-learning.omp)


(defun pseudoinversion (matrix)
  (metabang.math::svd-matrix-inverse matrix))


(defun matrix* (mat1 mat2)
  (if (= (array-dimension mat1 1)
         (array-dimension mat2 0))
      (let ((result (make-array (list (array-dimension mat1 0)
                                      (array-dimension mat2 1))
                                :element-type 'double-float)))
        (dotimes (row (array-dimension result 0))
          (dotimes (column (array-dimension result 1))
            (let ((terms 0))
              (dotimes (middle (array-dimension mat1 1))
                (setf terms (+ terms (* (or (aref mat1 row middle) 0)
                                        (or (aref mat2 middle column) 0)))))
              (setf (aref result row column) terms))))
        (return-from matrix* result))
      (error "not allowed!")))


(defun omp (results dictionaries iterations)
  (declare (optimize (speed 3) (safety 0))
           (type fixnum iterations)
           (type vector dictionaries results))
  (bind ((data-points-count (sl.data:data-points-count (first-elt results)))
         (atoms-count (sl.data:attributes-count (first-elt dictionaries))) ; trees-count
         (residuals (copy-array results))
         (selected-indexes (vect)))
    (declare (type fixnum atoms-count data-points-count)
             (type vector selected-indexes))
    (iterate
      (declare (type fixnum i max-d))
      (for i from 0 below iterations)
      (for max-d
           = (iterate outer
               (for dictionary in-vector dictionaries)
               (iterate
                 (declare (type fixnum d)
                          (type double-float d-val))
                 (for d from 0 below atoms-count)
                 (when (find d selected-indexes)
                   (next-iteration))
                 (for d-val = (sl.data:mref dictionary 0 d))
                 (in outer (finding d maximizing
                                    (abs (iterate inner
                                           (for residual in-vector residuals)
                                           (iterate
                                             (declare (type fixnum r)
                                                      (type double-float r-val))
                                             (for r from 0 below data-points-count)
                                             (for r-val = (sl.data:mref residual r 0))
                                             (in inner (summing (* d-val r-val)))))))))))
      (vector-push-extend max-d selected-indexes)
      (setf residuals
            (map 'vector
                 (lambda (dictionary result
                     &aux
                       (basis
                        (sl.data:sample dictionary
                                        :attributes selected-indexes))
                       (transposed
                        (metabang.math:transpose-matrix basis)))
                   (statistical-learning.data:map-data-matrix
                    #'-
                    (~> (matrix* transposed basis)
                        pseudoinversion
                        (matrix* transposed)
                        (matrix* result))))
                 dictionaries
                 results))
      (finally (return selected-indexes)))))


(defun extract-predictions (ensemble trees data parallel)
  (map 'vector
       (lambda (tree)
         (~> ensemble
             sl.ensemble:tree-parameters
             (sl.tp:contribute-predictions* tree
                                            data
                                            nil
                                            ensemble
                                            parallel)
             sl.tp:extract-predictions ))
       trees))


(defun extract-predictions-column (predictions column)
  (assert (not (emptyp predictions)))
  (assert (cl-ds.utils:homogenousp predictions :key #'sl.data:data-points-count))
  (bind ((number-of-trees (length predictions))
         (first-predictions (first-elt predictions))
         (result-data-points-count (sl.data:data-points-count first-predictions))
         (result (sl.data:make-data-matrix result-data-points-count
                                           number-of-trees)))
    (iterate
      (for i from 0 below result-data-points-count)
      (iterate
        (for j from 0 below number-of-trees)
        (for prediction = (~> (aref predictions j)
                              (sl.data:mref i column)))
        (setf (sl.data:mref result i j) prediction)))
    result))


(defun extract-results-column (results column)
  (sl.data:sample results :attributes (vector column)))
