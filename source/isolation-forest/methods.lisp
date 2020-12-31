(cl:in-package #:statistical-learning.isolation-forest)


(defmethod sl.tp:split*
    ((training-parameters isolation)
     training-state)
  (bind ((split-array #1=(~> training-state sl.mp:data-points
                             length sl.opt:make-split-array))
         (optimal-split-array #1#)
         (optimal-point nil)
         (new-depth (1+ (sl.tp:depth training-state)))
         (parallel (sl.tp:parallel training-parameters))
         ((:flet new-state (point position size))
          (sl.tp:split-training-state*
           training-parameters
           training-state
           optimal-split-array
           position
           size
           `(:depth ,new-depth)
           point))
         ((:flet subtree-impl (point position size))
          (sl.tp:split (new-state point position size)))
         ((:flet subtree (point position size &optional parallel))
          (if (and parallel (< new-depth 10))
              (lparallel:future (subtree-impl point position size))
              (subtree-impl point position size))))
    (iterate
      (with optimal-left-length = 0)
      (with optimal-right-length = 0)
      (repeat (repeats training-parameters))
      (for point = (sl.tp:pick-split training-state))
      (setf (sl.tp:split-point training-state) point)
      (for (values left-length right-length) =
           (sl.tp:fill-split-vector training-state
                                    split-array))
      (when (or (zerop left-length) (zerop right-length))
        (continue))
      (for score = (abs (- left-length right-length)))
      (minimize score into min)
      (when (= score min)
        (setf optimal-left-length left-length
              optimal-right-length right-length)
        (rotatef optimal-split-array split-array)
        (rotatef point optimal-point))
      (finally
       (if optimal-point
           (return (sl.tp:make-node
                    'sl.tp:fundamental-tree-node
                    :left-node (subtree optimal-point
                                        sl.opt:left
                                        optimal-left-length
                                        parallel)
                    :right-node (subtree optimal-point
                                         sl.opt:right
                                         optimal-right-length)
                    :point optimal-point))
           (return nil))))))


(defmethod sl.tp:leaf-for/proxy (splitter/proxy
                                 (splitter isolation-splitter)
                                 node
                                 data
                                 index
                                 context)
  (declare (type sl.data:double-float-data-matrix data)
           (type fixnum index))
  (bind ((attributes (attributes context))
         ((:labels impl
            (node depth &aux (next-depth (the fixnum (1+ depth)))))
          (declare (optimize (speed 3) (safety 0)))
          (if (sl.tp:treep node)
              (if (rightp (sl.tp:point node)
                          attributes
                          index data)
                  (~> node sl.tp:right-node (impl next-depth))
                  (~> node sl.tp:left-node (impl next-depth)))
              (values node depth))))
    (impl node 0)))


(defmethod sl.tp:requires-split-p/proxy and
    (parameters/proxy
     (splitter isolation-splitter)
     (isolation isolation)
     training-state)
  (and (> (~> training-state sl.mp:data-points length)
          (sl.tp:minimal-size isolation))
       (< (sl.tp:depth training-state)
          (sl.tp:maximal-depth isolation))))


(defmethod sl.mp:make-training-state/proxy (parameters/proxy
                                            (parameters isolation)
                                            &rest initargs
                                            &key
                                              data
                                              c
                                              data-points
                                              attributes)
  (declare (ignore initargs))
  (make 'isolation-training-state
        :parameters parameters
        :depth 0
        :c c
        :train-data data
        :data-points data-points
        :attributes attributes))


(defmethod sl.tp:make-leaf*/proxy (training-parameters/proxy
                                   (training-parameters isolation)
                                   state)
  (make 'isolation-leaf))


(defmethod sl.tp:initialize-leaf/proxy (training-parameters/proxy
                                        (training-parameters isolation)
                                        training-state
                                        leaf)
  leaf)


(defmethod sl.tp:pick-split*/proxy (splitter/proxy
                                    (splitter isolation-splitter)
                                    parameters
                                    state)
  (iterate
    (with data = (sl.mp:train-data state))
    (with samples = (sl.mp:data-points state))
    (with attributes = (sl.tp:attribute-indexes state))
    (with attributes-count = (length attributes))
    (with min = (ensure (mins state)
                  (calculate-mins data samples attributes)))
    (with max = (ensure (maxs state)
                  (calculate-maxs data samples attributes)))
    (with normals = (sl.data:make-data-matrix 1 attributes-count))
    (for i from 0 below attributes-count)
    (for avg = (/ (+ (sl.data:mref min 0 i)
                     (sl.data:mref max 0 i))
                  2))
    (setf (sl.data:mref normals 0 i) (sl.common:gauss-random 0.0d0
                                                             avg))
    (sum (* (sl.data:mref normals 0 i)
            (if (= (sl.data:mref min 0 i)
                   (sl.data:mref max 0 i))
                (sl.data:mref max 0 i)
                (random-in-range (sl.data:mref min 0 i)
                                 (sl.data:mref max 0 i))))
         into dot-product)
    (finally (return (make-split-point
                      :normals normals
                      :dot-product dot-product)))))


(defmethod sl.tp:fill-split-vector*/proxy
    (splitter/proxy
     (splitter isolation-splitter)
     parameters
     state
     point
     split-vector)
  (declare (type sl.data:split-vector split-vector)
           (type split-point point)
           (optimize (speed 3) (safety 0) (debug 0)))
  (bind ((data (sl.mp:train-data state))
         (data-points (sl.mp:data-points state))
         (attributes (sl.tp:attribute-indexes state)))
    (declare (type (simple-array fixnum (*)) data-points attributes))
    (iterate
      (declare (type fixnum right-count left-count i j))
      (with right-count = 0)
      (with left-count = 0)
      (for j from 0 below (length data-points))
      (for i = (aref data-points j))
      (for rightp = (rightp point attributes i data))
      (setf (aref split-vector j) rightp)
      (if rightp (incf right-count) (incf left-count))
      (finally (return (values left-count right-count))))))


(defmethod cl-ds.utils:cloning-information append
    ((object isolation-training-state))
  '((:parameters sl.mp:training-parameters)
    (:split-point sl.tp:split-point)
    (:depth sl.tp:depth)
    (:c c)
    (:train-data sl.mp:train-data)
    (:attributes sl.tp:attribute-indexes)
    (:data-points sl.mp:data-points)))


(defmethod sl.tp:extract-predictions*/proxy (parameters/proxy
                                             (parameters isolation)
                                             state)
  (let* ((trees-count (trees-count state))
         (trees-sum (trees-sum state))
         (c (c state))
         (result (sl.data:make-data-matrix-like trees-sum)))
    (iterate
      (for i from 0 below (sl.data:data-points-count trees-sum))
      (for avg = (/ (sl.data:mref trees-sum i 0)
                    trees-count))
      (for value = (/ 1 (expt 2 (/ avg c))))
      (setf (sl.data:mref result i 0) value)
      (finally (return result)))))


(defmethod sl.tp:contribute-predictions*/proxy
    (parameters/proxy
     (parameters isolation)
     tree-model
     data
     state
     context
     parallel
     &optional (leaf-key #'identity))
  (declare (ignore leaf-key))
  (when (null state)
    (let ((c (c tree-model))
          (data-points-count (sl.data:data-points-count data)))
      (check-type c double-float)
      (setf state (make 'isolation-prediction
                        :indexes (sl.data:iota-vector data-points-count)
                        :trees-sum (sl.data:make-data-matrix
                                    data-points-count
                                    1)
                        :parameters parameters
                        :c c))))
  (let* ((splitter (sl.tp:splitter parameters))
         (sums (trees-sum state))
         (root (sl.tp:root tree-model)))
    (funcall (if parallel #'lparallel:pmap #'map)
             nil
             (lambda (data-point)
               (bind (((:values leaf depth)
                       (sl.tp:leaf-for splitter root
                                       data data-point
                                       tree-model)))
                 (assert depth)
                 (assert leaf)
                 (incf (sl.data:mref sums data-point 0)
                       depth)))
             (sl.tp:indexes state))
    (incf (trees-count state))
    state))


(defmethod sl.mp:make-model*/proxy
    (parameters/proxy
     (parameters isolation)
     state)
  (let* ((root (sl.tp:make-tree state)))
    (make-instance 'isolation-model
                   :parameters parameters
                   :root root
                   :attributes (sl.tp:attribute-indexes state)
                   :c (c state))))


(defmethod initialize-instance :after ((object isolation)
                                       &rest all)
  (declare (ignore all))
  (bind (((:accessors (maximal-depth access-maximal-depth)
                      (repeats access-repeats)
                      (parallel access-parallel)
                      (minimal-size access-minimal-size))
          object))
    (check-type maximal-depth integer)
    (check-type repeats integer)
    (check-type parallel boolean)
    (check-type minimal-size integer)
    (assert (> maximal-depth 0)
            (maximal-depth)
            'cl-ds:argument-value-out-of-bounds
            :value maximal-depth
            :argument :minimal-size
            :bounds '(> minimal-size 0)
            :format-control "MAXIMAL-DEPTH should be greater then zero.")
    (assert (> minimal-size 0)
            (minimal-size)
            'cl-ds:argument-value-out-of-bounds
            :value minimal-size
            :argument :minimal-size
            :bounds '(> minimal-size 0)
            :format-control "MINIMAL-SIZE should be greater then zero.")))


(defmethod cl-ds.utils:cloning-information
    append ((object isolation))
  '((:maximal-depth sl.tp:maximal-depth)
    (:repeats repeats)
    (:splitter sl.tp:splitter)
    (:minimal-size sl.tp:minimal-size)
    (:parallel sl.tp:parallel)))
