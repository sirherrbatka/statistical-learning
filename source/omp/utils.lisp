(cl:in-package #:statistical-learning.omp)


(defun prune-trees-implementation (omp ensemble trees train-data target-data)
  (bind ((predictions (extract-predictions ensemble
                                           trees
                                           train-data
                                           nil))
         (result-columns (sl.data:attributes-count target-data))
         ((:flet map-columns (function))
          (iterate
            (with result = (make-array result-columns))
            (for i from 0 below result-columns)
            (setf (aref result i) (funcall function i))
            (finally (return result))))
         (dictionaries (map-columns (curry #'extract-predictions-column predictions)))
         (results (map-columns (curry #'extract-results-column target-data)))
         ((:values selected-trees)
          (omp results
               dictionaries
               (number-of-trees-selected omp)
               (threshold omp))))
    (map 'vector
         (lambda (index &aux (tree (aref trees index)) )
           tree)
         selected-trees)))


(defun inversion (u &optional (singularity-threshold 1.0d-10))
  (let* ((type (array-element-type u))
	       (m    (array-dimension u 0))
	       (n    (array-dimension u 1))
	       (w    (make-array n :element-type type))
	       (rv   (make-array n :element-type type))
	       (v    (make-array (list n n) :element-type type))
	       (a-1  (make-array (array-dimensions u) :element-type type)))
    (metabang.math::svdcmp-df u m n w v rv)
		(metabang.math::svzero-df w n singularity-threshold)
		(metabang.math::svd-inverse-fast-df u w v a-1 rv)
    a-1))


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


(defun norm2 (residuals)
  (flet ((impl (input)
           (sqrt (iterate
                   (for i from 0 below (array-total-size input))
                   (for val = (row-major-aref input i))
                   (summing (* val val))))))
    (reduce #'+ residuals :key #'impl)))


(defun matrix-difference (a b)
  (iterate
    (with result = (make-array (array-dimensions a)
                               :element-type 'double-float))
    (for i from 0 below (array-total-size a))
    (setf (row-major-aref result i)
          (- (row-major-aref a i)
             (row-major-aref b i)))
    (finally (return result))))


(defun omp (results dictionaries iterations threshold)
  (declare (type fixnum iterations)
           (type simple-vector dictionaries results))
  (bind ((atoms-count (sl.data:attributes-count (first-elt dictionaries))) ; trees-count
         (residuals (copy-array results))
         (selected-indexes (vect))
         ((:flet calculate-residual
            (dictionary
             result
             &aux
             (basis
              (sl.data:sample dictionary
                              :attributes selected-indexes))
             (transposed
              (metabang.math:transpose-matrix basis))))
          (matrix-difference
           result (~> (matrix* transposed basis)
                      inversion
                      (matrix* transposed)
                      (matrix* result)
                      (matrix* basis _)))))
    (declare (type fixnum atoms-count)
             (type vector selected-indexes))
    (iterate
      (declare (type fixnum i max-d))
      (for i from 0 below atoms-count)
      (while (and (or (null threshold)
                      (> (norm2 residuals) threshold))
                  (or (null iterations)
                      (< i iterations))))
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
                                             (for r from 0 below (sl.data:data-points-count residual))
                                             (for r-val = (sl.data:mref residual r 0))
                                             (in inner (summing (* d-val r-val)))))))))))
      (vector-push-extend max-d selected-indexes)
      (setf residuals
            (map 'vector #'calculate-residual dictionaries results))
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
