(cl:in-package #:sl.som)


(defmethod sl.mp:sample-training-state-info/proxy append
    (parameters/proxy
     (parameters abstract-self-organizing-map)
     state
     &key data-points)
  (list :data (sl.data:sample (sl.mp:train-data state)
                              :data-points data-points)
        :units (cl-ds.utils:transform (units state) #'copy-array)
        :weights (if-let ((weights (weights state)))
                   (sl.data:sample weights :data-points data-points)
                   nil)))


(defmethod initialize-instance :after ((object abstract-self-organizing-map)
                                       &rest initargs)
  (declare (ignore initargs))
  (let ((number-of-iterations (number-of-iterations object))
        (parallel (parallel object))
        (grid-dimensions (grid-dimensions object))
        (decay (decay object))
        (alpha (initial-alpha object))
        (random-ranges (random-ranges object)))
    (declare (ignore parallel decay))
    (check-type alpha double-float)
    (check-type random-ranges (or null sequence))
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


(defmethod initialize-instance :after ((object self-organizing-map)
                                       &rest initargs)
  (declare (ignore initargs))
  ; this is there simply to ensure that that the matching-unit-selector has been passed
  (assert (matching-unit-selector object)))


(defmethod sl.mp:make-training-state/proxy
    (parameters/proxy
     (parameters abstract-self-organizing-map)
     &rest initargs
     &key data
       weights)
  (declare (ignore initargs))
  (let* ((attributes-count (sl.data:attributes-count data))
         (grid-dimensions (grid-dimensions parameters))
         (random-ranges (random-ranges parameters))
         (grid (make-grid grid-dimensions
                          attributes-count
                          random-ranges)))
    (make 'self-organizing-map-training-state
          :data data
          :initial-sigma (~> parameters grid-dimensions first (/ 2.0d0))
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
  (fit parameters training-state)
  (make 'self-organizing-map-model
        :parameters parameters
        :units (units training-state)))


(defmethod sl.mp:make-model*/proxy (parameters/proxy
                                    (parameters random-forest-self-organizing-map)
                                    training-state)
  (fit parameters training-state)
  (let* ((parallel (parallel parameters))
         (forest (forest parameters))
         (units (units training-state))
         (units-data-matrix (units-data-matrix units))
         (unit-leafs (sl.ensemble:leafs forest
                                         units-data-matrix
                                         parallel)))
    (make-instance 'random-forest-self-organizing-map-model
                   :parameters parameters
                   :unit-leafs unit-leafs
                   :units units)))


(defmethod make-units-container ((model self-organizing-map-model) data index)
  (make 'units-container
        :data data
        :index index
        :units (units model)
        :parameters (sl.mp:parameters model)))


(defmethod make-units-container ((model random-forest-self-organizing-map-model)
                                 data index)
  (make 'units-container-with-unit-leafs
        :unit-leafs (unit-leafs model)
        :leafs (sl.ensemble:leafs (~> model sl.mp:parameters forest)
                                  data
                                  (~> model
                                      sl.mp:parameters
                                      parallel))
        :data data
        :index index
        :units (units model)
        :parameters (sl.mp:parameters model)))


(defmethod sl.mp:predict ((model self-organizing-map-model)
                          data
                          &optional parallel)
  (let* ((all-indexes (~> data sl.data:data-points-count sl.data:iota-vector))
         (units (units model))
         (parameters (sl.mp:parameters model))
         (result (sl.data:make-data-matrix (sl.data:data-points-count data)
                                           (array-rank units)))
         (units-container (make-units-container model data 0)))
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (i)
               (iterate
                 (for j from 0)
                 (for value in (~>> (cl-ds.utils:quasi-clone units-container :index i)
                                    (find-best-matching-unit parameters)
                                    (serapeum:array-index-row-major units)))
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


(defmethod cl-ds.utils:cloning-information
    append ((object self-organizing-map-training-state))
  '((:data sl.mp:train-data)
    (:all-distances all-distances)
    (:all-indexes all-indexes)
    (:units units)
    (:initial-sigma initial-sigma)
    (:weights weights)))


(defmethod find-best-matching-unit-with-selector
    ((selector euclid-matching-unit-selector)
     parameters
     data
     sample
     units)
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


(defmethod find-best-matching-unit ((parameters self-organizing-map)
                                    (units-container units-container))
  (find-best-matching-unit-with-selector (matching-unit-selector parameters)
                                         parameters
                                         (data units-container)
                                         (index units-container)
                                         (units units-container)))


(defmethod unit-leafs ((container units-container))
  (let* ((parameters (sl.mp:parameters container))
         (parallel (parallel parameters))
         (forest (forest parameters))
         (units (units container))
         (units-data-matrix (units-data-matrix units)))
    (sl.ensemble:leafs forest
                       units-data-matrix
                       parallel)))


(defmethod cl-ds.utils:cloning-information append ((container units-container))
  `((:data data)
    (:index index)
    (:units units)
    (:parameters sl.mp:parameters)))


(defmethod cl-ds.utils:cloning-information append ((container units-container-with-unit-leafs))
  `((:leafs leafs)
    (:unit-leafs unit-leafs)))


(defmethod leafs ((units-container units-container))
  (data units-container))


(defmethod find-best-matching-unit ((parameters random-forest-self-organizing-map)
                                    (units-container units-container))
  (iterate
    (declare (type fixnum i))
    (with unit-leafs = (unit-leafs units-container))
    (with leafs = (leafs units-container))
    (with sample = (index units-container))
    (with sample-leafs = (sl.data:mref leafs sample 0))
    (for i from 0 below (sl.data:data-points-count unit-leafs))
    (for unit = (sl.data:mref unit-leafs i 0))
    (for distance = (jaccard-distance sample-leafs unit))
    (finding i minimizing distance)))


(defmethod fit ((parameters self-organizing-map) state)
  (iterate
    (with data = (sl.mp:train-data state))
    (with data-points-count = (sl.data:data-points-count data))
    (for i from 1 to (~> state sl.mp:training-parameters number-of-iterations))
    (for random = (random data-points-count))
    (for container = (make-instance 'units-container
                                    :data data
                                    :index random
                                    :parameters parameters
                                    :units (units state)))
    (update-units state i container data)))


(defmethod fit ((parameters random-forest-self-organizing-map) state)
  (iterate
    (with data = (sl.mp:train-data state))
    (with data-points-count = (sl.data:data-points-count data))
    (with parallel = (parallel parameters))
    (with forest = (forest parameters))
    (with leafs = (sl.ensemble:leafs forest data parallel))
    (for i from 1 to (~> state sl.mp:training-parameters number-of-iterations))
    (for random = (random data-points-count))
    (for container = (make-instance 'units-container
                                    :data leafs
                                    :index random
                                    :parameters parameters
                                    :units (units state)))
    (update-units state i container data)))
