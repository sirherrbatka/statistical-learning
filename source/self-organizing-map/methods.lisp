(cl:in-package #:sl.som)


(defmethod sl.mp:sample-training-state-info/proxy append
    (parameters/proxy
     (parameters self-organizing-map)
     state
     &key data-points)
  (list :data (sl.data:sample (data state)
                              :data-points data-points)
        :units (cl-ds.utils:transform (units state) #'copy-array)
        :weights (let ((weights (weights state)))
                   (if (null weights)
                       nil
                       (sl.data:sample weights
                                       :data-points data-points)))))


(defmethod cl-ds.utils:cloning-information append ((state self-organizing-map-training-state))
  `((:data data)
    (:units units)
    (:all-distances all-distances)
    (:weights weights)
    (:all-indexes all-indexes)))


(defmethod initialize-instance :after ((object self-organizing-map)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((number-of-iterations (number-of-iterations object))
        (parallel (parallel object))
        (grid-dimensions (grid-dimensions object))
        (decay (decay object))
        (alpha (initial-alpha object))
        (sigma (initial-sigma object)))
    (declare (ignore parallel decay))
    (check-type alpha double-float)
    (check-type sigma double-float)
    (when (emptyp grid-dimensions)
      (error 'cl-ds:invalid-argument-value
             :argument :grid-dimensions
             :value grid-dimensions
             :format-control "Empty grid-dimensions list passed."))
    (iterate
      (for dimension in grid-dimensions)
      (check-type dimension integer)
      (unless (> dimension 0)
        (error (error 'cl-ds:invalid-argument-value
                      :argument :grid-dimensions
                      :value grid-dimensions
                      :format-control "Each dimension passed in the grid-dimensions should be positive."))))
    (check-type number-of-iterations integer)
    (unless (> number-of-iterations 0)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :iterations
             :value number-of-iterations
             :bounds '(> number-of-iterations 0)))))


(defmethod sl.mp:make-training-state/proxy
    (parameters/proxy
     (parameters self-organizing-map)
     &rest initargs
     &key data
       weights)
  (declare (ignore initargs))
  (let* ((attributes-count (sl.data:attributes-count data))
         (grid (~> parameters grid-dimensions
                   (make-grid attributes-count))))
    (make 'self-organizing-map-training-state
          :data data
          :training-parameters parameters
          :units grid
          :all-indexes (~> grid array-total-size sl.data:iota-vector)
          :all-distances (all-manhattan-distances grid)
          :data data
          :weights (if (null weights)
                       nil
                       (copy-array weights)))))


(defmethod sl.mp:make-model*/proxy (parameters/proxy
                                    (parameters self-organizing-map)
                                    training-state)
  (fit training-state)
  (make 'self-organizing-map-model
        :parameters parameters
        :units (units training-state)))


(defmethod sl.mp:predict ((model self-organizing-map-model)
                          data
                          &optional parallel)
  (let* ((all-indexes (~> data sl.data:data-points-count sl.data:iota-vector))
         (units (units model))
         (result (sl.data:make-data-matrix (sl.data:data-points-count data)
                                           (array-rank units))))
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (i)
               (iterate
                 (for j from 0)
                 (for value in (~>> (find-best-matching-unit data i units)
                                    (cl-ds.utils:row-major-index-to-subscripts data)))
                 (setf (sl.data:mref result i j) (coerce value 'double-float))))
             all-indexes)
    result))


(defmethod alpha ((decay linear-decay) initial iteration iterations)
  (check-type iterations positive-integer)
  (check-type iteration non-negative-integer)
  (check-type initial double-float)
  (+ initial
     (* (/ iteration iterations)
        (- initial +linear-decay-final-alpha+))))


(defmethod sigma ((decay linear-decay) initial iteration iterations)
  (check-type iterations positive-integer)
  (check-type iteration non-negative-integer)
  (check-type initial double-float)
  (+ initial
     (* (/ iteration iterations)
        (- initial +linear-decay-final-sigma+))))


(defmethod alpha ((decay hill-decay) initial iteration iterations)
  (check-type iterations positive-integer)
  (check-type iteration non-negative-integer)
  (check-type initial double-float)
  (/ initial
     (1+ (expt (* (/ iteration iterations) 2.0d0) 4))))


(defmethod sigma ((decay hill-decay) initial iteration iterations)
  (check-type iterations positive-integer)
  (check-type iteration non-negative-integer)
  (check-type initial double-float)
  (/ initial
     (1+ (expt (* (/ iteration iterations) 2.0d0) 4))))


(defmethod unit-at ((model self-organizing-map-model) location)
  (apply #'aref (units model) location))
