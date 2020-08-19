(cl:in-package #:sl.som)


(defun subscripts (grid)
  (let* ((array-dimensions (array-dimensions grid))
         (generator (cl-ds.utils:cycle-over-address array-dimensions))
         (result (make-array (array-total-size grid))))
    (map-into result generator)
    result))


(defun manhattan-distance (a b)
  (iterate
    (for ea in a)
    (for eb in b)
    (sum (abs (- ea eb)) into result)
    (finally (return (coerce result 'double-float)))))


(defun all-manhattan-distances (grid)
  (cl-ds.utils:make-distance-matrix-from-vector 'double-float
                                                #'manhattan-distance
                                                (subscripts grid)
                                                :initial-element 0.0d0))


(-> find-best-matching-unit (sl.data:double-float-data-matrix fixnum grid) fixnum)
(defun find-best-matching-unit (data sample units)
  (iterate
    (declare (type fixnum i))
    (for i from 0 below (array-total-size units))
    (for unit = (row-major-aref units i))
    (for distance = (iterate
                      (declare (type fixnum i)
                               (type double-float result))
                      (with result = 0.0d0)
                      (for i from 0 below (length unit))
                      (incf result (~> (- (sl.data:mref data sample i)
                                          (aref unit i))
                                       cl-ds.utils:square))
                      (finally (return result))))
    (finding i minimizing distance)))


(defun update-units (state sample iteration)
  (let* ((data (data state))
         (units (units state))
         (training-parameters (sl.mp:training-parameters state))
         (parallel (parallel training-parameters))
         (distances (all-distances state))
         (weights (weights state))
         (weight (sl.opt:weight-at weights sample))
         (decay (~> state sl.mp:training-parameters decay))
         (iterations (number-of-iterations training-parameters))
         (alpha (alpha decay (initial-alpha training-parameters) iteration iterations))
         (sigma (sigma decay (initial-sigma training-parameters) iteration iterations))
         (best-matching-unit (find-best-matching-unit data sample units))
         (all-indexes (all-indexes state)))
    (declare (type sl.data:double-float-data-matrix data)
             (type double-float alpha sigma weight)
             (type fixnum best-matching-unit)
             (type grid units))
    (flet ((update-weight (unit-index)
             (declare (type array-index unit-index)
                      (optimize (speed 3)))
             (iterate
               (declare (type fixnum i)
                        (type unit unit)
                        (type double-float h v distance))
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
               (for v = (sl.data:mref data sample i))
               (decf #1=(aref unit i)
                     (* h (- #1# v))))))
      (funcall (if parallel #'lparallel:pmap #'map)
               nil
               #'update-weight
               all-indexes))))


(defun fit (state)
  (iterate
    (with data = (data state))
    (with data-points-count = (sl.data:data-points-count data))
    (for i from 1 to (~> state sl.mp:training-parameters number-of-iterations))
    (for random = (random data-points-count))
    (update-units state random i)))


(defun make-unit (attributes-count)
  (lret ((result (make-array attributes-count
                             :element-type 'double-float)))
    (map-into result (curry #'random-in-range -1.0d0 1.0d0))))


(defun make-grid (grid-dimensions attributes-count)
  (lret ((result (make-array grid-dimensions :element-type 'unit)))
    (map-into (cl-ds.utils:unfold-table result)
              (curry #'make-unit attributes-count))))
