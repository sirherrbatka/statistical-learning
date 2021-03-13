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


(defmethod sl.tp:requires-split-p/proxy and
    (parameters/proxy
     splitter
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
  (make-instance 'isolation-model
                 :parameters parameters
                 :root (sl.tp:make-tree state)
                 :attribute-indexes (sl.tp:attribute-indexes state)
                 :c (c state)))


(defmethod initialize-instance :after ((object isolation)
                                       &rest all)
  (declare (ignore all))
  (bind (((:accessors (maximal-depth access-maximal-depth)
                      (repeats access-repeats)
                      (parallel access-parallel)
                      (minimal-size access-minimal-size))
          object))
    (check-type parallel boolean)
    (cl-ds.utils:check-value repeats
      (check-type repeats integer)
      (assert (> repeats 0)
              (repeats)
              'cl-ds:argument-value-out-of-bounds
              :value repeats
              :argument :repeat
              :bounds '(> repeats 0)
              :format-control "REPEATS should be greater then zero."))
    (cl-ds.utils:check-value maximal-depth
      (check-type maximal-depth integer)
      (assert (> maximal-depth 0)
              (maximal-depth)
              'cl-ds:argument-value-out-of-bounds
              :value maximal-depth
              :argument :maximal-depth
              :bounds '(> maximal-depth 0)
              :format-control "MAXIMAL-DEPTH should be greater then zero."))
    (cl-ds.utils:check-value minimal-size
      (check-type minimal-size integer)
      (assert (> minimal-size 0)
              (minimal-size)
              'cl-ds:argument-value-out-of-bounds
              :value minimal-size
              :argument :minimal-size
              :bounds '(> minimal-size 0)
              :format-control "MINIMAL-SIZE should be greater then zero."))))


(defmethod cl-ds.utils:cloning-information
    append ((object isolation))
  '((:maximal-depth sl.tp:maximal-depth)
    (:repeats repeats)
    (:splitter sl.tp:splitter)
    (:minimal-size sl.tp:minimal-size)
    (:parallel sl.tp:parallel)))
