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
    ((object tree-model))
  '((:root root)
    (:attribute-indexes attribute-indexes)
    (:forced forced)
    (:weight weight)))


(defmethod cl-ds.utils:cloning-information append
    ((object tree-training-state))
  `((:depth depth)
    (:split-point split-point)
    (:attributes attribute-indexes)
    (:loss loss)
    (:weights sl.mp:weights)
    (:target-data sl.mp:target-data)
    (:splitter-state splitter-state)
    (:train-data sl.mp:train-data)
    (:parent-state parent-state)))


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


(defmethod split-training-state-info/proxy append
    (parameters/proxy
     (splitter fundamental-splitter)
     (parameters basic-tree-training-parameters)
     state
     split-array
     position
     size
     point)
  (data-matrix-split-list size
                          split-array
                          position
                          :train-data (sl.mp:train-data state)
                          :target-data (sl.mp:target-data state)))


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
  (~> (contribute-predictions model (sl.data:wrap data) nil parallel)
      extract-predictions))


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
    (check-type minimal-difference single-float)
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
            (< (abs (- (loss state) (split-result-loss result state)))
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
  (bind ((data-size (~> training-state sl.mp:train-data sl.data:data-points-count))
         (split-array (sl.opt:make-split-array data-size))
         (point (pick-split training-state))
         ((:values left-length right-length middle-length)
          (progn
            (setf (split-point training-state) point)
            (let ((results (multiple-value-list (fill-split-vector training-state
                                                                   split-array)))
                  (middle-strategy (middle-strategy splitter)))
              (apply #'handle-split-middle
                     middle-strategy
                     split-array
                     results))))
         ((:values left-score right-score)
          (calculate-loss* training-parameters
                           training-state
                           split-array
                           left-length
                           right-length)))
    (ensure middle-length 0)
    (let ((result (make 'split-result
                        :split-point point
                        :split-vector split-array
                        :left-score left-score
                        :right-score right-score
                        :left-length left-length
                        :right-length right-length
                        :middle-length middle-length)))
      (setf (split-point training-state) point)
      (if (split-result-accepted-p training-parameters training-state result)
          result
          nil))))


(defmethod split-using-splitter/proxy ((splitter/proxy random-splitter)
                                       (splitter fundamental-splitter)
                                       (training-parameters fundamental-tree-training-parameters)
                                       training-state)
  (iterate
    (declare (type fixnum attempt data-size trials-count))
    (with trials-count = (trials-count splitter/proxy))
    (with data-size = (~> training-state sl.mp:train-data sl.data:data-points-count))
    (with optimal-split-result = nil)
    (for attempt from 0 below (min data-size trials-count))
    (for split-result = (call-next-method))
    (when (null split-result) (next-iteration))
    (when (and split-result
               (split-result-improved-p training-parameters
                                        training-state
                                        split-result
                                        optimal-split-result))
      (setf optimal-split-result split-result))
    (finally
     (when optimal-split-result
       (setf (split-point training-state) (split-point optimal-split-result)))
     (return optimal-split-result))))


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
           (+ size (middle-length split-result))
           `(:depth ,new-depth :loss ,loss)
           (split-point split-result)))
         ((:flet subtree-impl (position
                               size
                               loss
                               &aux (state (new-state position size loss))))
          (make-tree state))
         ((:flet subtree (position size loss &optional parallel))
          (if (zerop size)
              nil
              (if (and parallel (< new-depth 8))
                  (lparallel:future (subtree-impl position size loss))
                  (subtree-impl position size loss)))))
    (if (null split-result)
        nil
        (make-node (tree-node-class training-parameters)
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
           :parent-state state
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
     &key train-attributes)
  (list :attributes (if (null train-attributes)
                        (attribute-indexes state)
                        train-attributes)))


(defmethod leaf-for/proxy (proxy
                           (splitter random-attribute-splitter)
                           (node fundamental-node)
                           data index context)
  (declare (type sl.data:single-float-data-matrix data)
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
                            (type single-float attribute-value))
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
         (data (the sl.data:single-float-data-matrix (sl.mp:train-data state)))
         ((mins . maxs) (ensure (sl.mp:cache state 'mins/maxs)
                          (sl.data:mins/maxs data :attributes attributes)))
         (attributes-count (length attributes))
         (attribute-index (random attributes-count))
         (attribute (aref attributes attribute-index))
         (min (sl.data:mref (the sl.data:single-float-data-matrix mins)
                            0 attribute-index))
         (max (sl.data:mref (the sl.data:single-float-data-matrix maxs)
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
         (length (length split-vector)))
    (declare (type sl.data:single-float-data-matrix data)
             (type single-float threshold)
             (type fixnum length attribute))
    (assert (< attribute (sl.data:attributes-count data)))
    (iterate
      (declare (type fixnum i left-count right-count middle-count)
               (type single-float value)
               (type boolean present))
      (with left-count = 0)
      (with right-count = 0)
      (with middle-count = 0)
      (for i from 0 below length)
      (for (values value present) = (sl.data:mref data i attribute))
      (if present
          (let ((right (> (sl.data:mref data i attribute) threshold)))
            (if right
                (incf right-count)
                (incf left-count))
            (setf (aref split-vector i) right))
          (setf (aref split-vector i) sl.opt:middle
                middle-count (1+ middle-count)))
      (finally (return (values left-count right-count middle-count))))))


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
  (let* ((depth (depth training-state))
         (loss (loss training-state))
         (maximal-depth (maximal-depth training-parameters))
         (minimal-size (minimal-size training-parameters)))
    (declare (type (integer 1 *) minimal-size))
    (nor (< (sl.data:data-points-count (sl.mp:train-data training-state))
            (* 2 minimal-size))
         (>= depth maximal-depth)
         (<= loss (minimal-difference training-parameters)))))


(defmethod pick-split*/proxy (splitter/proxy
                              (splitter distance-splitter)
                              parameters
                              state)
  (let* ((train-data (sl.mp:train-data state))
         (length (sl.data:data-points-count train-data))
         (distance-function (ensure-function (distance-function splitter)))
         (first-index 0)
         (repeats (repeats splitter))
         (second-index (iterate
                         (for r = (random length))
                         (while (= r first-index))
                         (finally (return (or r 0))))))
    (declare (type sl.data:universal-data-matrix train-data)
             (type fixnum repeats first-index second-index))
    (iterate
      (declare (type fixnum iterations))
      (with iterations = (iterations splitter))
      (repeat iterations)
      (for first-data-point = first-index)
      (for object = (sl.data:mref train-data first-data-point 0))
      (for result =
           (iterate
             (with repeat = 0)
             (for i = (random length))
             (when (= i first-index) (next-iteration))
             (for other = (sl.data:mref train-data i 0))
             (for distance = (funcall distance-function object other))
             (incf repeat)
             (until (= repeat repeats))
             (finding i maximizing distance)))
      (when (= result second-index) (finish))
      (setf second-index result)
      (rotatef first-index second-index))
    (cons (sl.data:mref train-data first-index 0)
          (sl.data:mref train-data second-index 0))))


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
         (distance-function (ensure-function (distance-function splitter)))
         (train-data (sl.mp:train-data state)))
    (declare (type sl.data:universal-data-matrix train-data))
    (iterate
      (declare (type fixnum i left-length right-length))
      (with left-length = 0)
      (with right-length = 0)
      (for i from 0 below (sl.data:data-points-count train-data))
      (for object = (sl.data:mref train-data i 0))
      (for left-distance = (funcall distance-function
                                    left-pivot
                                    object))
      (for right-distance = (funcall distance-function
                                     right-pivot
                                     object))
      (for rightp = (< right-distance left-distance))
      (setf (aref split-vector i) rightp)
      (if rightp
          (incf right-length)
          (incf left-length))
      (finally (return (values left-length right-length 0))))))


(defmethod leaf-for/proxy (splitter/proxy
                           (splitter distance-splitter)
                           node
                           data
                           index
                           context)
  (declare (type fixnum index)
           (type sl.data:universal-data-matrix data))
  (let ((object (sl.data:mref data index 0))
        (distance-function (ensure-function (distance-function splitter))))
    (labels ((impl (node depth &aux (new-depth (the fixnum (1+ depth))))
               (declare (optimize (speed 3) (safety 0)
                                  (debug 0) (space 0)
                                  (compilation-speed 0)))
               (setf node (lparallel:force node))
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
  (> (~> training-state sl.mp:train-data sl.data:data-points-count)
     2))


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


(defmethod pick-split*/proxy (splitter/proxy
                               (splitter hyperplane-splitter)
                               parameters
                               state)
  (declare (optimize (speed 3) (safety 0)
                     (debug 0) (space 0)
                     (compilation-speed 0)))
  (iterate
    (declare (type sl.data:single-float-data-matrix normals data max min)
             (type fixnum i attributes-count)
             (type single-float dot-product)
             (type (simple-array fixnum (*)) attributes))
    (with dot-product = 0.0)
    (with data = (sl.mp:train-data state))
    (with attributes = (sl.tp:attribute-indexes state))
    (with attributes-count = (length attributes))
    (with max = (ensure (sl.mp:cache state 'maxs)
                  (sl.data:maxs data :attributes attributes)))
    (with min = (ensure (sl.mp:cache state 'mins)
                  (sl.data:mins data :attributes attributes)))
    (with normals = (sl.data:make-data-matrix 1 attributes-count))
    (for i from 0 below attributes-count)
    (setf (sl.data:mref normals 0 i) (sl.random:random-gauss 0.0 1.0))
    (incf dot-product (* (sl.data:mref normals 0 i)
                         (if (= (sl.data:mref min 0 i)
                                (sl.data:mref max 0 i))
                             (sl.data:mref max 0 i)
                             (sl.random:random-uniform (sl.data:mref min 0 i)
                                                       (sl.data:mref max 0 i)))))
    (finally (return (cons normals dot-product)))))


(defmethod fill-split-vector*/proxy
    (splitter/proxy
     (splitter hyperplane-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector))
  (bind ((data (sl.mp:train-data state))
         ((normals . dot-product) point)
         (attributes (sl.tp:attribute-indexes state)))
    (declare (type (simple-array fixnum (*)) attributes))
    (iterate
      (declare (type fixnum right-count left-count i))
      (with right-count = 0)
      (with left-count = 0)
      (for i from 0 below (sl.data:data-points-count data))
      (for rightp = (< (wdot data normals i 0 attributes)
                       (the single-float dot-product)))
      (setf (aref split-vector i) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count 0))))))


(defmethod leaf-for/proxy (splitter/proxy
                           (splitter hyperplane-splitter)
                           node
                           data
                           index
                           context)
  (declare (type (simple-array single-float (* *)) data)
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
                       (the single-float dot-product))
                    (~> node right-node (impl next-depth))
                    (~> node left-node (impl next-depth))))
              (values node depth))))
    (impl node 0)))


(defmethod pick-split*/proxy (splitter/proxy
                              (splitter set-splitter)
                              parameters
                              state)
  (bind ((parent-state (parent-state state))
         (previous-points (if (null parent-state)
                              '()
                              (split-point parent-state)))
         (train-data (sl.mp:train-data state))
         (data-points-count (sl.data:data-points-count train-data))
         (attributes-count (sl.data:attributes-count train-data))
         (second-random (random attributes-count))
         (first-random (random data-points-count))
         (set (sl.data:mref train-data
                            first-random
                            second-random))
         (third-random (random (length set))))
    (set-splitter-split-point (aref set third-random)
                              second-random
                              (random (length (aref set third-random)))
                              previous-points)))


(defmethod fill-split-vector*/proxy (splitter/proxy
                                     (splitter set-splitter)
                                     parameters
                                     state
                                     split-point
                                     split-vector)
  (bind ((train-data (sl.mp:train-data state))
         (left-count 0)
         (right-count 0)
         (middle-count 0)
         (attributes-count (sl.data:attributes-count train-data)))
    (iterate
      (for i from 0 below (sl.data:data-points-count train-data))
      (setf (aref split-vector i)
            (iterate outer
              (for index from 0 below attributes-count)
              (for set = (sl.data:mref train-data i index))
              (iterate
                (for tuple in-vector set)
                (when (set-splitter-split-point-side index
                                                     tuple
                                                     split-point)
                  (in outer (leave sl.opt:right))))
              (finally (return-from outer sl.opt:left))))
      (switch-direction ((aref split-vector i))
                        (incf left-count)
                        (incf right-count))
      (finally (return (values left-count right-count middle-count))))))


(defmethod leaf-for/proxy (splitter/proxy
                           (splitter set-splitter)
                           node
                           data
                           index
                           context)
  (declare (type fixnum index))
  (let ((attributes-count (sl.data:attributes-count data)))
    (labels ((impl (node depth &aux (new-depth (the fixnum (1+ depth))))
               (setf node (lparallel:force node))
               (if (treep node)
                   (bind ((split-point (sl.tp:point node)))
                     (switch-direction ((iterate outer
                                           (for attribute from 0 below attributes-count)
                                           (for set = (sl.data:mref data index attribute))
                                           (iterate
                                             (for tuple in-vector set)
                                             (when (set-splitter-split-point-side attribute
                                                                                  tuple
                                                                                  split-point)
                                               (in outer (leave sl.opt:right))))
                                          (finally (return-from outer sl.opt:left))))
                                       (~> node left-node (impl new-depth))
                                       (~> node right-node (impl new-depth))))
                   (values node depth))))
      (impl node 0))))


(defmethod make-tree (training-state)
  (split training-state))


(defmethod handle-split-middle/proxy (middle-strategy/proxy
                                      (middle-strategy proportional-middle-strategy)
                                      split-vector
                                      left-length
                                      right-length
                                      middle-length)
  (ensure middle-length 0)
  (if (zerop middle-length)
      (values left-length right-length middle-length)
      (iterate
        (with sum = (+ left-length right-length))
        (with new-left-length = left-length)
        (with new-right-length = right-length)
        (for i from 0 below (length split-vector))
        (unless (eq (svref split-vector i) sl.opt:middle)
          (next-iteration))
        (setf (svref split-vector i) (if (< (random sum) left-length)
                                         (progn (incf new-left-length)
                                                sl.opt:left)
                                         (progn (incf new-right-length)
                                                sl.opt:right)))
        (finally (return (values new-left-length new-right-length 0))))))


(defmethod handle-split-middle/proxy (middle-strategy/proxy
                                      (middle-strategy naive-middle-strategy)
                                      split-vector
                                      left-length
                                      right-length
                                      middle-length)
  (values (+ left-length middle-length)
          (+ right-length middle-length)
          0))
