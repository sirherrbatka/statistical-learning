(cl:in-package #:sl.som)


(defun subscripts (grid)
  (let* ((array-dimensions (array-dimensions grid))
         (generator (cl-ds.utils:cycle-over-address array-dimensions))
         (result (make-array (array-total-size grid))))
    (map-into result generator)
    result))


(-> manhattan-distance (t t) double-float)
(defun manhattan-distance (a b)
  (iterate
    (declare (type double-float result))
    (with result = 0.0d0)
    (for ea in a)
    (for eb in b)
    (incf result (abs (- ea eb)) )
    (finally (return (the double-float result)))))


(defun all-manhattan-distances (grid)
  (cl-ds.utils:make-distance-matrix-from-vector 'double-float
                                                #'manhattan-distance
                                                (subscripts grid)
                                                :initial-element 0.0d0))


(defun units-data-matrix (units)
  (iterate
    (with data-points-count = (array-total-size units))
    (with attributes-count = (length (row-major-aref units 0)))
    (with result = (sl.data:make-data-matrix data-points-count
                                             attributes-count))
    (for i from 0 below data-points-count)
    (for unit = (row-major-aref units i))
    (iterate
      (for j from 0 below attributes-count)
      (setf (sl.data:mref result i j) (aref unit j)))
    (finally (return result))))


(defun jaccard-distance (leafs-1 leafs-2)
  (let ((length-1 (length leafs-1))
        (length-2 (length leafs-2)))
    (assert (= length-1 length-2))
    (iterate
      (with result = length-1)
      (for i from 0 below length-1)
      (when (eq (aref leafs-1 i)
                (aref leafs-2 i))
        (decf result))
      (finally (return result)))))


(defun update-units (state iteration units-container training-data)
  (let* ((training-parameters (sl.mp:training-parameters state))
         (parallel (parallel training-parameters))
         (distances (all-distances state))
         (weights (weights state))
         (sample (index units-container))
         (weight (sl.opt:weight-at weights sample))
         (decay (~> state sl.mp:training-parameters decay))
         (iterations (number-of-iterations training-parameters))
         (alpha (alpha decay (initial-alpha training-parameters) iteration iterations))
         (sigma (sigma decay (initial-sigma state) iteration iterations))
         (best-matching-unit
           (find-best-matching-unit training-parameters
                                    units-container))
         (units (units units-container))
         (all-indexes (all-indexes state)))
    (declare (type sl.data:data-matrix training-data)
             (type double-float alpha sigma weight)
             (type fixnum best-matching-unit)
             (type grid units))
    (flet ((update-weight (unit-index)
             (declare (type array-index unit-index))
             (iterate
               (declare (type fixnum i)
                        (type unit unit)
                        (type double-float h distance))
               (with distance = (if (= unit-index best-matching-unit)
                                    0.0d0
                                    (cl-ds.utils:mref distances
                                                      unit-index
                                                      best-matching-unit)))
               (with h = (~> (/ distance sigma)
                             cl-ds.utils:square
                             -
                             exp
                             (* alpha)
                             (* weight)))
               (with unit = (row-major-aref units unit-index))
               (for i from 0 below (length unit))
               (for v = (sl.data:mref training-data sample i))
               (decf #1=(aref unit i)
                     (* h (- #1# v))))))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               #'update-weight
               all-indexes))))


(defun make-unit (attributes-count random-ranges)
  (lret ((result (make-array attributes-count
                             :element-type 'double-float)))
    (map-into result (curry #'random-in-range -1.0d0 1.0d0))
    (unless (null random-ranges)
      (map-into result
                (compose (rcurry #'coerce 'double-float)
                         (curry #'apply #'random-in-range))
                random-ranges))
    result))


(defun make-grid (grid-dimensions attributes-count random-ranges)
  (lret ((result (make-array grid-dimensions :element-type 'unit)))
    (map-into (cl-ds.utils:unfold-table result)
              (curry #'make-unit attributes-count random-ranges))))
