(cl:in-package #:statistical-learning.tree-protocol)


(defmethod (setf training-parameters) :before (new-value state)
  (check-type new-value fundamental-tree-training-parameters))


(defmethod make-node (node-class &rest arguments)
  (apply #'make node-class arguments))


(defmethod (setf maximal-depth) :before (new-value
                                         training-parameters)
  (check-type new-value (integer 1 *)))


(defmethod (setf depth) :before (new-value training-parameters)
  (check-type new-value (integer 0 *)))


(defmethod treep ((node fundamental-tree-node))
  t)


(defmethod treep ((node fundamental-leaf-node))
  nil)


(defmethod force-tree* ((node fundamental-tree-node))
  (setf (left-node node) (~> node left-node lparallel:force)
        (right-node node) (~> node right-node lparallel:force))
  (unless (null (left-node node))
    (~> node left-node force-tree*))
  (unless (null (right-node node))
    (~> node right-node force-tree*))
  node)


(defmethod force-tree* ((node fundamental-leaf-node))
  node)


(defmethod cl-ds.utils:cloning-information append
    ((object tree-training-state))
  `((:depth depth)
    (:split-point split-point)
    (:optimal-split-point optimal-split-point)
    (:loss loss)
    (:attributes attribute-indexes)
    (:data-points sl.mp:data-points)
    (:weights sl.mp:weights)
    (:target-data sl.mp:target-data)
    (:train-data sl.mp:train-data)))


(defmethod split-training-state-info/proxy append
    (parameters/proxy
     (splitter fundamental-splitter)
     (parameters basic-tree-training-parameters)
     state
     split-array
     position
     size
     point)
  (list
   :data-points (iterate
                  (declare (type (simple-array fixnum (*))
                                 old-indexes new-indexes))
                  (with old-indexes = (sl.mp:data-points state))
                  (with new-indexes = (make-array size :element-type 'fixnum))
                  (with j = 0)
                  (for i from 0 below (length old-indexes))
                  (when (eql position (aref split-array i))
                    (setf (aref new-indexes j) (aref old-indexes i))
                    (incf j))
                  (finally
                   (assert (= j size))
                   (return new-indexes)))))


(defmethod split-training-state-info/proxy append
    (parameters/proxy
     (splitter random-attribute-splitter)
     (parameters basic-tree-training-parameters)
     state
     split-array
     position
     size
     point)
  (bind ((attribute-index (car point))
         (attributes (attribute-indexes state)))
    (list
     :attributes (if (null attribute-index)
                     attributes
                     (iterate
                       (with size = (~> attributes length 1-))
                       (with result = (make-array size :element-type 'fixnum))
                       (with j = 0)
                       (for i from 0 below (length attributes))
                       (for value = (aref attributes i))
                       (unless (= value attribute-index)
                         (setf (aref result j) value)
                         (incf j))
                       (finally
                        (assert (= j size))
                        (return result)))))))


(defmethod split*/proxy :around (parameters/proxy
                                 (training-parameters fundamental-tree-training-parameters)
                                 training-state)
  (if (requires-split-p (splitter training-parameters)
                        training-parameters
                        training-state)
      (call-next-method)
      nil))


(defmethod contribute-predictions*/proxy
    :before (parameters/proxy
             (parameters fundamental-tree-training-parameters)
             (model tree-model)
             data
             state
             context
             parallel
             &optional leaf-key)
  (declare (ignore leaf-key))
  (unless (forced model)
    (force-tree model)))


(defmethod statistical-learning.mp:predict ((model tree-model)
                                            data
                                            &optional parallel)
  (~> (contribute-predictions model data nil parallel)
      extract-predictions))


(defmethod initialize-instance :after ((instance tree-training-state)
                                       &rest initargs)
  (declare (ignore initargs))
  (ensure (attribute-indexes instance)
    (~> instance
        sl.mp:train-data
        sl.data:attributes-count
        sl.data:iota-vector))
  (ensure (sl.mp:data-points instance)
    (~> instance
        sl.mp:train-data
        sl.data:data-points-count
        sl.data:iota-vector)))


(defmethod initialize-instance :after ((instance distance-splitter)
                                       &rest initargs)
  (declare (ignore initargs))
  (ensure-function (distance-function instance))
  (let ((iterations (iterations instance))
        (repeats (repeats instance)))
    (check-type iterations integer)
    (check-type repeats integer)
    (unless (>= iterations 0)
      (error 'cl-ds:argument-value-out-of-bounds
             :format-control "Iterations is supposed to be non-negative."
             :argument :iterations
             :value iterations
             :bounds '(>= iterations)))
    (unless (>= repeats 0)
      (error 'cl-ds:argument-value-out-of-bounds
             :format-control "Repeats is supposed to be non-negative."
             :argument :repeats
             :value repeats
             :bounds '(>= repeats)))))


(defmethod initialize-instance :after
    ((instance standard-tree-training-parameters)
     &rest initargs)
  (declare (ignore initargs))
  (let ((maximal-depth (maximal-depth instance))
        (minimal-size (minimal-size instance))
        (minimal-difference (minimal-difference instance))
        (trials-count (trials-count instance)))
    (parallel instance) ; here just to check if slot is bound
    (check-type maximal-depth integer)
    (check-type minimal-difference double-float)
    (unless (< 0 maximal-depth)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :maximal-depth
             :bounds '(< 0 :maximal-depth)
             :value maximal-depth))
    (check-type minimal-size integer)
    (unless (< 0 minimal-size)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :minimal-size
             :bounds '(<= 0 :minimal-size)
             :value minimal-size))
    (unless (integerp trials-count)
      (error 'type-error :expected 'integer
                         :datum trials-count))
    (unless (< 0 trials-count)
      (error 'cl-ds:argument-value-out-of-bounds
             :argument :trials-count
             :bounds '(< 0 :trials-count)
             :value trials-count))))


(defmethod make-leaf*/proxy (parameters/proxy
                             (parameters fundamental-tree-training-parameters))
  (make 'standard-leaf-node))


(defmethod split*/proxy
    (parameters/proxy
     (training-parameters basic-tree-training-parameters)
     training-state)
  (declare (optimize (debug 3) (speed 0) (safety 3)))
  (bind ((trials-count (trials-count training-parameters))
         (minimal-difference (minimal-difference training-parameters))
         (score (loss training-state))
         (minimal-size (minimal-size training-parameters))
         (parallel (parallel training-parameters)))
    (declare (type fixnum trials-count)
             (type double-float score minimal-difference)
             (type boolean parallel))
    (iterate
      (declare (type fixnum attempt left-length right-length
                     optimal-left-length optimal-right-length data-size)
               (type double-float left-score right-score minimal-score))
      (with optimal-left-length = -1)
      (with optimal-right-length = -1)
      (with optimal-point = nil)
      (with minimal-score = most-positive-double-float)
      (with minimal-left-score = most-positive-double-float)
      (with minimal-right-score = most-positive-double-float)
      (with data-size = (~> training-state
                            sl.mp:data-points
                            length))
      (with split-array = (sl.opt:make-split-array data-size))
      (with optimal-array = (sl.opt:make-split-array data-size))
      (for attempt from 0 below (min data-size trials-count))
      (for point = (pick-split training-state))
      (setf (split-point training-state) point)
      (for (values left-length right-length) = (fill-split-vector
                                                training-state
                                                split-array))
      (when (or (< left-length minimal-size)
                (< right-length minimal-size))
        (next-iteration))
      (for (values left-score right-score) =
           (calculate-loss* training-parameters
                            training-state
                            split-array))
      (for split-score = (+ (* (/ left-length data-size)
                               left-score)
                            (* (/ right-length data-size)
                               right-score)))
      (when (< split-score minimal-score)
        (setf minimal-score split-score
              optimal-point point
              (optimal-split-point training-state) optimal-point
              optimal-left-length left-length
              optimal-right-length right-length
              minimal-left-score left-score
              minimal-right-score right-score)
        (rotatef split-array optimal-array))
      (finally
       (let ((difference (- (the double-float score)
                            (the double-float minimal-score))))
         (declare (type double-float difference))
         (when (< difference minimal-difference)
           (return nil))
         (bind ((new-depth (~> training-state depth 1+))
                ((:flet new-state (position size loss))
                 (split-training-state*/proxy
                  parameters/proxy
                  training-parameters
                  training-state
                  optimal-array
                  position
                  size
                  `(:depth ,new-depth :loss ,loss)
                  optimal-point))
                ((:flet subtree-impl (position
                                      size
                                      loss
                                      &aux (state (new-state position size loss))))
                 (~>> (make-leaf*/proxy parameters/proxy
                                        training-parameters)
                      (split state _ parameters/proxy)))
                ((:flet subtree (position size loss &optional parallel))
                 (if (and parallel (< new-depth 10))
                     (lparallel:future (subtree-impl position size loss))
                     (subtree-impl position size loss))))
           (return (make-node 'fundamental-tree-node
                              :left-node (subtree sl.opt:left
                                                  optimal-left-length
                                                  minimal-left-score
                                                  parallel)
                              :right-node (subtree sl.opt:right
                                                   optimal-right-length
                                                   minimal-right-score)
                              :point optimal-point))))))))


(defmethod split-training-state*/proxy
    (parameters/proxy
     (parameters basic-tree-training-parameters)
     state split-array
     position size initargs
     point)
  (bind ((cloning-list (cl-ds.utils:cloning-list state)))
    (apply #'make (class-of state)
           (append initargs
                   (split-training-state-info/proxy
                    parameters/proxy
                    (splitter parameters)
                    parameters
                    state
                    split-array
                    position
                    size
                    point)
                   cloning-list))))


(defmethod sl.mp:sample-training-state-info/proxy append
    (parameters/proxy
     (parameters basic-tree-training-parameters)
     state
     &key train-attributes data-points)
  (list :attributes (if (null train-attributes)
                        (attribute-indexes state)
                        train-attributes)
        :data-points (if (null data-points)
                         (sl.mp:data-points state)
                         data-points)))


(defmethod leaf-for/proxy (proxy
                           (splitter random-attribute-splitter)
                           (node fundamental-node)
                           data index context)
  (declare (type sl.data:double-float-data-matrix data)
           (type fixnum index))
  (labels ((impl (node depth &aux (new-depth (the fixnum (1+ depth))))
             (declare (optimize (speed 3) (safety 0)))
             (if (treep node)
                 (bind (((attribute-index . attribute-value) (point node)))
                   (declare (type fixnum attribute-index)
                            (type double-float attribute-value))
                   (if (> (sl.data:mref data index attribute-index)
                          attribute-value)
                       (~> node right-node (impl new-depth))
                       (~> node left-node (impl new-depth))))
                 (values node depth))))
    (impl node 0)))


(defmethod pick-split*/proxy (splitter/proxy
                              (splitter random-attribute-splitter)
                              parameters state)
  (declare (optimize (speed 3) (safety 0))
           (ignore parameters))
  (bind ((attributes (the (simple-array fixnum (*))
                          (attribute-indexes state)))
         (data (sl.mp:train-data state))
         (attributes-count (length attributes))
         (attribute-index (aref attributes (random attributes-count)))
         ((:values min max) (sl.data:data-min/max data
                                                  attribute-index
                                                  (sl.mp:data-points state)))
         (threshold (if (= min max) min (sl.common:random-uniform min max))))
    (list* attribute-index (if (= threshold max) min threshold))))


(defmethod fill-split-vector*/proxy
    (splitter/proxy
     (splitter random-attribute-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector)
           (type cons point)
           (optimize (speed 3) (safety 0) (debug 0)))
  (bind ((attribute (car point))
         (threshold (cdr point))
         (data (sl.mp:train-data state))
         (data-point-indexes (sl.mp:data-points state))
         (length (length split-vector)))
    (declare (type sl.data:double-float-data-matrix data)
             (type (simple-array fixnum (*)) data-point-indexes)
             (type double-float threshold)
             (type fixnum length attribute))
    (assert (< attribute (sl.data:attributes-count data)))
    (iterate
      (declare (type fixnum right-count left-count i j)
               (type boolean rightp))
      (with right-count = 0)
      (with left-count = 0)
      (for j from 0 below length)
      (for i = (aref data-point-indexes j))
      (for rightp = (> (sl.data:mref data i attribute) threshold))
      (setf (aref split-vector j) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count))))))


(defmethod requires-split-p/proxy
    and (parameters/proxy
         (splitter random-attribute-splitter)
         (training-parameters basic-tree-training-parameters)
         training-state)
  (~> training-state attribute-indexes emptyp not))


(defmethod requires-split-p/proxy
    and (parameters/proxy
         (splitter fundamental-splitter)
         (training-parameters standard-tree-training-parameters)
         training-state)
  (let* ((indexes (sl.mp:data-points training-state))
         (depth (depth training-state))
         (loss (loss training-state))
         (maximal-depth (maximal-depth training-parameters))
         (minimal-size (minimal-size training-parameters)))
    (declare (type (integer 1 *) minimal-size))
    (nor (< (length indexes) (* 2 minimal-size))
         (>= depth maximal-depth)
         (<= loss (minimal-difference training-parameters)))))


(defmethod pick-split*/proxy (splitter/proxy
                              (splitter distance-splitter)
                              parameters
                              state)
  (let* ((data-points (sl.mp:data-points state))
         (train-data (sl.mp:train-data state))
         (length (length data-points))
         (distance-function (ensure-function (distance-function splitter)))
         (first-index (random length))
         (repeats (repeats splitter))
         (second-index (iterate
                         (for r = (random length))
                         (while (= r first-index))
                         (finally (return r)))))
    (declare (type (simple-array fixnum (*)) data-points)
             (type sl.data:universal-data-matrix train-data)
             (type fixnum repeats first-index second-index))
    (iterate
      (declare (type fixnum iterations))
      (with iterations = (iterations splitter))
      (repeat iterations)
      (for first-data-point = (aref data-points first-index))
      (for object = (sl.data:mref train-data first-data-point 0))
      (for result =
           (iterate
             (repeat repeats)
             (for i = (random length))
             (when (= i first-index) (next-iteration))
             (for second-data-point = (aref data-points i))
             (for other = (sl.data:mref train-data second-data-point 0))
             (for distance = (funcall distance-function object other))
             (finding i maximizing distance)))
      (when (= result second-index) (finish))
      (setf second-index result)
      (rotatef first-index second-index))
    (cons (~> (aref data-points first-index)
              (sl.data:mref train-data _ 0))
          (~> (aref data-points second-index)
              (sl.data:mref train-data _ 0)))))


(defmethod fill-split-vector*/proxy
    (splitter/proxy
     (splitter distance-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type cons point)
           (type sl.data:split-vector split-vector)
           (optimize (speed 3) (safety 0)))
  (bind (((left-pivot . right-pivot) point)
         (data-points (sl.mp:data-points state))
         (distance-function (ensure-function (distance-function splitter)))
         (train-data (sl.mp:train-data state)))
    (declare (type (simple-array fixnum (*)) data-points)
             (type sl.data:universal-data-matrix train-data))
    (iterate
      (declare (type fixnum j i left-length right-length))
      (with left-length = 0)
      (with right-length = 0)
      (for j from 0 below (length data-points))
      (for i = (aref data-points j))
      (for object = (sl.data:mref train-data i 0))
      (for left-distance = (funcall distance-function
                                    left-pivot
                                    object))
      (for right-distance = (funcall distance-function
                                     right-pivot
                                     object))
      (for rightp = (< right-distance left-distance))
      (setf (aref split-vector j) rightp)
      (if rightp
          (incf right-length)
          (incf left-length))
      (finally (return (values left-length right-length))))))


(defmethod leaf-for/proxy (splitter/proxy
                           (splitter distance-splitter)
                           node
                           data
                           index
                           context)
  (declare (type fixnum index))
  (let ((object (sl.data:mref data index 0))
        (distance-function (ensure-function (distance-function splitter))))
    (labels ((impl (node depth &aux (new-depth (the fixnum (1+ depth))))
               (declare (optimize (speed 3) (safety 0)))
               (if (treep node)
                   (bind (((left-pivot . right-pivot) (point node))
                          (left-distance (funcall distance-function
                                                  left-pivot
                                                  object))
                          (right-distance (funcall distance-function
                                                   right-pivot
                                                   object)))
                     (if (< right-distance left-distance)
                         (~> node right-node (impl new-depth))
                         (~> node left-node (impl new-depth))))
                   (values node depth))))
      (impl node 0))))


(defmethod requires-split-p/proxy
    and (parameters/proxy
         (splitter distance-splitter)
         parameters
         training-state)
  (> (~> training-state sl.mp:data-points length) 2))


(defmethod sl.mp:make-model*/proxy
    (parameters-proxy
     (parameters basic-tree-training-parameters)
     state)
  (let* ((protoroot (make-leaf* parameters))
         (root (split state protoroot parameters-proxy)))
    (make 'tree-model
          :parameters parameters
          :root root)))
