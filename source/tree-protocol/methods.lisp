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
    (:loss loss)
    (:attributes attribute-indexes)
    (:data-point-indexes sl.mp:data-points)
    (:weights sl.mp:weights)
    (:target-data sl.mp:target-data)
    (:train-data sl.mp:train-data)))


(defmethod split-training-state-info append
    ((splitter random-attribute-splitter)
     (parameters standard-tree-training-parameters)
     state
     split-array
     position
     size
     point)
  (bind ((attribute-index (car point))
         (attributes (attribute-indexes state))
         (new-attributes (if (null attribute-index)
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
                                (return result)))))
         (new-data-points (iterate
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
    (list :data-point-indexes new-data-points
          :attributes new-attributes)))


(defmethod split* :around ((training-parameters fundamental-tree-training-parameters)
                           training-state)
  (if (requires-split-p (splitter training-parameters)
                        training-parameters
                        training-state)
      (call-next-method)
      nil))


(defmethod contribute-predictions* :before ((parameters fundamental-tree-training-parameters)
                                            (model tree-model)
                                            data
                                            state
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


(defmethod initialize-instance :after
    ((instance fundamental-tree-training-parameters)
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


(defmethod make-leaf* ((parameters fundamental-tree-training-parameters))
  (make 'standard-leaf-node))


(defmethod split*
    ((training-parameters fundamental-tree-training-parameters)
     training-state)
  (declare (optimize (speed 3) (safety 0)))
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
               (type double-float
                     left-score right-score minimal-score))
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
      (for attempt from 0 below trials-count)
      (for point = (pick-split training-state))
      (for (values left-length right-length) = (fill-split-vector
                                                training-state
                                                point
                                                split-array))
      (when (or (< left-length minimal-size)
                (< right-length minimal-size))
        (next-iteration))
      (for (values left-score right-score) = (calculate-loss*
                                              training-parameters
                                              training-state
                                              split-array))
      (for split-score = (+ (* (/ left-length data-size) left-score)
                            (* (/ right-length data-size) right-score)))
      (when (< split-score minimal-score)
        (setf minimal-score split-score
              optimal-point point
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
                 (split-training-state* training-parameters
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
                 (~>> state make-leaf (split state)))
                ((:flet subtree (position size loss &optional parallel))
                 (if (and parallel (< new-depth 5))
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


(defmethod split-training-state* ((parameters standard-tree-training-parameters)
                                  state split-array
                                  position size initargs
                                  point)
  (bind ((cloning-list (cl-ds.utils:cloning-list state)))
    (apply #'make (class-of state)
           (append (split-training-state-info (splitter parameters)
                                              parameters
                                              state
                                              split-array
                                              position
                                              size
                                              point)
                   initargs
                   cloning-list))))


(defmethod sl.mp:sample-training-state-info append
    ((parameters fundamental-tree-training-parameters)
     state
     &key train-attributes data-points)
  (list :attributes (if (null train-attributes)
                        (attribute-indexes state)
                        train-attributes)
        :data-point-indexes (if (null data-points)
                                (sl.mp:data-points state)
                                data-points)))


(defmethod leaf-for ((splitter random-attribute-splitter)
                     (node fundamental-node)
                     data index)
  (declare (type sl.data:double-float-data-matrix data)
           (type fixnum index))
  (labels ((impl (node)
             (declare (optimize (speed 3) (safety 0)))
             (if (treep node)
                 (bind (((attribute-index . attribute-value) (point node)))
                   (declare (type fixnum attribute-index)
                            (type double-float attribute-value))
                   (if (> (sl.data:mref data index attribute-index)
                          attribute-value)
                       (~> node right-node impl)
                       (~> node left-node impl)))
                 node)))
    (impl node)))


(defmethod pick-split* ((splitter random-attribute-splitter) parameters state)
  "Uses ExtraTree approach."
  (declare (optimize (speed 3) (safety 0))
           (ignore parameters))
  (bind ((attributes (the (simple-array fixnum (*))
                          (attribute-indexes state)))
         (data (sl.mp:train-data state))
         (attributes-count (length attributes))
         (attribute-index (aref attributes (random attributes-count)))
         ((:values min max) (data-min/max data
                                          attribute-index
                                          (sl.mp:data-points state)))
         (threshold (if (= min max) min (random-uniform min max))))
    (list* attribute-index (if (= threshold max) min threshold))))


(defmethod fill-split-vector* ((splitter random-attribute-splitter)
                               parameters
                               state
                               point
                               split-vector)
  (declare (type sl.data:split-vector split-vector)
           (type cons point)
           (optimize (speed 3) (safety 0)))
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
      (declare (type fixnum right-count left-count i)
               (type boolean rightp))
      (with right-count = 0)
      (with left-count = 0)
      (for j from 0 below length)
      (for i = (aref data-point-indexes j))
      (for rightp = (> (sl.data:mref data i attribute) threshold))
      (setf (aref split-vector j) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count))))))


(defmethod requires-split-p and ((splitter random-attribute-splitter)
                                 (training-parameters standard-tree-training-parameters)
                                 training-state)
  (~> training-state attribute-indexes emptyp not))


(defmethod requires-split-p and ((splitter fundamental-splitter)
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
