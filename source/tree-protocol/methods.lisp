(cl:in-package #:statistical-learning.tree-protocol)


(defmethod print-object ((object split-result) stream)
  (print-unreadable-object (object stream)
    (format stream "~a" (split-point object))))


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
    (:attributes attribute-indexes)
    (:loss loss)
    (:data-points sl.mp:data-points)
    (:weights sl.mp:weights)
    (:target-data sl.mp:target-data)
    (:splitter-state splitter-state)
    (:train-data sl.mp:train-data)))


(defmethod split-training-state-info/proxy append
    (parameters/proxy
     (splitter data-point-oriented-splitter)
     (parameters basic-tree-training-parameters)
     state
     split-array
     position
     size
     point)
  (let ((old-indexes (sl.mp:data-points state)))
    (assert (= (length split-array) (length old-indexes)))
    (list
     :data-points (iterate
                    (declare (type (simple-array fixnum (*))
                                   old-indexes new-indexes))
                    (with new-indexes = (make-array size :element-type 'fixnum))
                    (with j = 0)
                    (for i from 0 below (length old-indexes))
                    (when (eql position (aref split-array i))
                      (setf (aref new-indexes j) (aref old-indexes i))
                      (incf j))
                    (finally
                     (assert (= j size)
                             (j size))
                     (return new-indexes))))))


(defmethod split-training-state-info/proxy append
    (parameters/proxy
     (splitter random-attribute-splitter)
     (parameters basic-tree-training-parameters)
     state
     split-array
     position
     size
     point)
  (bind ((attribute-index (first point))
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
        (minimal-difference (minimal-difference instance)))
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
             :value minimal-size))))


(defmethod initialize-instance :after
    ((instance random-splitter)
     &rest
       initargs
     &aux
       (trials-count (trials-count instance)))
  (declare (ignore initargs))
  (unless (integerp trials-count)
    (error 'type-error :expected 'integer
                       :datum trials-count))
  (unless (< 0 trials-count)
    (error 'cl-ds:argument-value-out-of-bounds
           :argument :trials-count
           :bounds '(< 0 :trials-count)
           :value trials-count)))


(defmethod make-leaf*/proxy (parameters/proxy
                             (parameters fundamental-tree-training-parameters)
                             state)
  (make 'standard-leaf-node))


(defmethod split-result-accepted-p/proxy (parameters/proxy
                                          (parameters standard-tree-training-parameters)
                                          state
                                          result)
  (let ((minimal-size (minimal-size parameters))
        (left-length (left-length result))
        (right-length (right-length result)))
    (if (or (< left-length minimal-size)
            (< right-length  minimal-size)
            (< (- (loss state) (split-result-loss result state))
               (minimal-difference parameters)))
        nil
        t)))


(defmethod split-result-improved-p/proxy (parameters/proxy
                                          (parameters standard-tree-training-parameters)
                                          state
                                          new-result
                                          old-result)
  (if (null old-result)
      t
      (< (split-result-loss new-result state)
         (split-result-loss old-result state))))


(defmethod split-using-splitter/proxy (splitter/proxy
                                       (splitter fundamental-splitter)
                                       (training-parameters fundamental-tree-training-parameters)
                                       training-state)
  (bind ((data-size (~> training-state
                        sl.mp:data-points
                        length))
         (split-array (sl.opt:make-split-array data-size))
         (point (pick-split training-state))
         ((:values left-length right-length)
          (progn
            (setf (split-point training-state) point)
            (fill-split-vector training-state
                               split-array)))
         ((:values left-score right-score)
          (calculate-loss* training-parameters
                           training-state
                           split-array
                           left-length
                           right-length)))
    (assert (= (+ left-length right-length) data-size))
    (let ((result (make 'split-result
                        :split-point point
                        :split-vector split-array
                        :left-score left-score
                        :right-score right-score
                        :left-length left-length
                        :right-length right-length)))
      (if (split-result-accepted-p training-parameters training-state result)
          result
          nil))))


(defmethod split-using-splitter/proxy ((splitter/proxy random-splitter)
                                       (splitter fundamental-splitter)
                                       (training-parameters fundamental-tree-training-parameters)
                                       training-state)
  (iterate
    (declare (type fixnum attempt left-length right-length data-size
                   trials-count))
    (with trials-count = (trials-count splitter/proxy))
    (with data-size = (~> training-state sl.mp:data-points length))
    (with optimal-split-result = nil)
    (for attempt from 0 below (min data-size trials-count))
    (for split-result = (call-next-method))
    (when (null split-result) (next-iteration))
    (for point = (split-point split-result))
    (setf (split-point training-state) point)
    (for left-length = (left-length split-result))
    (for right-length = (right-length split-result))
    (assert (= (+ left-length right-length) data-size))
    (when (and split-result
               (split-result-improved-p training-parameters
                                        training-state
                                        split-result
                                        optimal-split-result))
      (setf optimal-split-result split-result))
    (finally (return optimal-split-result))))


(defmethod split*
    ((training-parameters fundamental-tree-training-parameters)
     training-state)
  (bind ((split-result (split-using-splitter (splitter training-parameters)
                                             training-parameters
                                             training-state))
         (parallel (parallel training-parameters))
         (new-depth (~> training-state depth 1+))
         ((:flet new-state (position size loss))
          (split-training-state
           training-state
           (split-vector split-result)
           position
           size
           `(:depth ,new-depth :loss ,loss)
           (split-point split-result)))
         ((:flet subtree-impl (position
                               size
                               loss
                               &aux (state (new-state position size loss))))
          (make-tree state))
         ((:flet subtree (position size loss &optional parallel))
          (if (and parallel (< new-depth 10))
              (lparallel:future (subtree-impl position size loss))
              (subtree-impl position size loss))))
    (if (null split-result)
        nil
        (make-node 'fundamental-tree-node
                   :left-node (subtree sl.opt:left
                                       (left-length split-result)
                                       (left-score split-result)
                                       parallel)
                   :right-node (subtree sl.opt:right
                                        (right-length split-result)
                                        (right-score split-result))
                   :point (split-point split-result)))))


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
             (declare (optimize (speed 3) (safety 0)
                                (space 0) (debug 0)
                                (compilation-speed 0)))
             (setf node (lparallel:force node))
             (assert (not (null node)))
             (if (treep node)
                 (bind (((attribute-index attribute-value) (point node)))
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
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0))
           (ignore parameters))
  (bind ((attributes (the (simple-array fixnum (*))
                          (attribute-indexes state)))
         (data (the sl.data:double-float-data-matrix (sl.mp:train-data state)))
         (data-points (sl.mp:data-points state))
         ((mins . maxs) (ensure (sl.mp:cache state 'mins/maxs)
                          (sl.data:mins/maxs data :data-points data-points
                                                  :attributes attributes)))
         (attributes-count (length attributes))
         (attribute-index (random attributes-count))
         (attribute (aref attributes attribute-index))
         (min (sl.data:mref (the sl.data:double-float-data-matrix mins)
                            0 attribute-index))
         (max (sl.data:mref (the sl.data:double-float-data-matrix maxs)
                            0 attribute-index))
         (threshold (if (= min max) min (sl.random:random-uniform min max))))
    (list attribute (if (= threshold max) min threshold))))


(defmethod fill-split-vector*/proxy
    (splitter/proxy
     (splitter random-attribute-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector)
           (type list point)
           (optimize (speed 3) (space 0)
                     (safety 0) (debug 0)
                     (compilation-speed 0)))
  (bind ((attribute (first point))
         (threshold (second point))
         (data (sl.mp:train-data state))
         (data-point-indexes (sl.mp:data-points state))
         (length (length split-vector)))
    (declare (type sl.data:double-float-data-matrix data)
             (type (simple-array fixnum (*)) data-point-indexes)
             (type double-float threshold)
             (type fixnum length attribute))
    (assert (< attribute (sl.data:attributes-count data)))
    (assert (= (length data-point-indexes) length))
    (iterate
      (declare (type fixnum
                     left-count1 left-count2 left-count3 left-count4
                     right-count1 right-count2 right-count3 right-count4
                     j1 j2 j3 j4))
      (with right-count1 = 0)
      (with right-count2 = 0)
      (with right-count3 = 0)
      (with right-count4 = 0)
      (with left-count1 = 0)
      (with left-count2 = 0)
      (with left-count3 = 0)
      (with left-count4 = 0)
      (for j1 from 0 below length by 4)
      (for i1 = (aref data-point-indexes j1))
      (for j2 = (+ j1 1))
      (for j3 = (+ j1 2))
      (for j4 = (+ j1 3))
      (for rightp1 = (> (sl.data:mref data i1 attribute) threshold))
      (setf (aref split-vector j1) rightp1)
      (if rightp1 (incf right-count1) (incf left-count1))
      ;; this code is micro-optimized
      (cond ((< j4 length)
             (let* ((i4 (aref data-point-indexes j4))
                    (rightp4 (> (sl.data:mref data i4 attribute) threshold)))
               (setf (aref split-vector j4) rightp4)
               (if rightp4 (incf right-count4) (incf left-count4)))
             #1=(let* ((i3 (aref data-point-indexes j3))
                       (rightp3 (> (sl.data:mref data i3 attribute) threshold)))
                  (setf (aref split-vector j3) rightp3)
                  (if rightp3 (incf right-count3) (incf left-count3)))
             #2=(let* ((i2 (aref data-point-indexes j2))
                       (rightp2 (> (sl.data:mref data i2 attribute) threshold)))
                  (setf (aref split-vector j2) rightp2)
                  (if rightp2 (incf right-count2) (incf left-count2))))
            ((< j3 length) #1# #2#)
            ((< j2 length) #2#))
      (finally
       (let ((left (the fixnum (+ left-count1 left-count2 left-count3 left-count4)))
             (right (the fixnum (+ right-count1 right-count2 right-count3 right-count4))))
         (assert (= (the fixnum (+ left right)) length))
         (return (values left right)))))))


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
           (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0)))
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
               (declare (optimize (speed 3) (safety 0)
                                  (debug 0) (space 0)
                                  (compilation-speed 0)))
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
  (let* ((root (make-tree state)))
    (make 'tree-model
          :parameters parameters
          :attribute-indexes (attribute-indexes state)
          :root root)))


(defmethod cl-ds.utils:cloning-information
    append ((object standard-tree-training-parameters))
  '((:maximal-depth maximal-depth)
    (:minimal-difference minimal-difference)
    (:minimal-size minimal-size)
    (:parallel parallel)
    (:splitter splitter)))


(defmethod sl.tp:pick-split*/proxy (splitter/proxy
                                    (splitter hyperplane-splitter)
                                    parameters
                                    state)
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0)))
  (iterate
    (declare (type sl.data:double-float-data-matrix normals data max min)
             (type fixnum i attributes-count)
             (type double-float dot-product)
             (type (simple-array fixnum (*)) attributes samples))
    (with dot-product = 0.0d0)
    (with data = (sl.mp:train-data state))
    (with samples = (sl.mp:data-points state))
    (with attributes = (sl.tp:attribute-indexes state))
    (with attributes-count = (length attributes))
    (with max = (ensure (sl.mp:cache state 'maxs)
                  (sl.data:maxs data
                              :data-points samples
                              :attributes attributes)))
    (with min = (ensure (sl.mp:cache state 'mins)
                  (sl.data:mins data
                                :data-points samples
                                :attributes attributes)))
    (with normals = (sl.data:make-data-matrix 1 attributes-count))
    (for i from 0 below attributes-count)
    (setf (sl.data:mref normals 0 i) (sl.random:random-gauss 0.0d0 1.0d0))
    (incf dot-product (* (sl.data:mref normals 0 i)
                         (if (= (sl.data:mref min 0 i)
                                (sl.data:mref max 0 i))
                             (sl.data:mref max 0 i)
                             (sl.random:random-uniform (sl.data:mref min 0 i)
                                                       (sl.data:mref max 0 i)))))
    (finally (return (cons normals dot-product)))))


(defmethod sl.tp:fill-split-vector*/proxy
    (splitter/proxy
     (splitter hyperplane-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector))
  (bind ((data (sl.mp:train-data state))
         (data-points (sl.mp:data-points state))
         ((normals . dot-product) point)
         (attributes (sl.tp:attribute-indexes state)))
    (declare (type (simple-array fixnum (*)) data-points attributes))
    (iterate
      (declare (type fixnum right-count left-count i j))
      (with right-count = 0)
      (with left-count = 0)
      (for j from 0 below (length data-points))
      (for i = (aref data-points j))
      (for rightp = (< (wdot data normals i 0 attributes)
                       (the double-float dot-product)))
      (setf (aref split-vector j) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count))))))


(defmethod sl.tp:leaf-for/proxy (splitter/proxy
                                 (splitter hyperplane-splitter)
                                 node
                                 data
                                 index
                                 context)
  (declare (type sl.data:double-float-data-matrix data)
           (type fixnum index))
  (bind ((attributes (attribute-indexes context))
         ((:labels impl (node depth
                              &aux (next-depth (the fixnum (1+ depth)))))
          (declare (optimize (speed 3) (safety 0)
                             (debug 0) (space 0)
                             (compilation-speed 0)))
          (if (sl.tp:treep node)
              (bind (((normals . dot-product) (point node)))
                (if (< (wdot data normals index 0 attributes)
                       (the double-float dot-product))
                    (~> node right-node (impl next-depth))
                    (~> node left-node (impl next-depth))))
              (values node depth))))
    (impl node 0)))
